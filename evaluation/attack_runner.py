# evaluation/attack_runner.py
"""
Generic attack runner — evaluates any RAG system against the attack suite.

Designed to be reused unchanged for S0, S1, S2, S3.
The only thing that differs between systems is the RAG object passed in.

Key design decisions:
  1. INCREMENTAL SAVE — results written after every single attack.
     If the API call fails midway, just rerun — already-done attacks
     are skipped automatically (resume-safe).

  2. INDIRECT ATTACK HANDLING (fixed from original):
     Indirect attacks (I1/I2/I3) inject the poisoned document text
     directly into the RAG context via query_with_injected_doc().
     This correctly tests: "given retrieval succeeds, does the LLM
     follow the injection?" — consistent with Phase 1B which separately
     measures retrieval probability.
     The original bug was calling rag.query(trigger) which retrieved
     clean documents and never surfaced the poisoned content.

  3. MULTI-TURN D5 — calls query_multi_turn() with the stored turn
     sequence and replaces __PLACEHOLDER__ with real Turn 1 response.

  4. SAFETY BLOCK HANDLING — Gemini's safety filter may block some
     D2/D3 attacks. These are recorded as safety_blocked=True and
     counted as failed attacks. The paper documents this as partial
     first-line defence provided by the LLM's alignment training.

  5. RATE LIMITING — config.API_RATE_LIMIT_DELAY applied between
     all LLM calls. For Gemini free tier (10 RPM), 6.5s delay is safe.
"""

import os
import json
import time
from tqdm import tqdm

import config
from rag.llm_client import SAFETY_BLOCKED


class AttackRunner:

    def __init__(self,
                 rag,             # single RAG instance (clean DB for all)
                 system_name: str,
                 results_dir: str):
        """
        Args:
            rag         : RAG system instance (BaselineRAG or defended variant)
            system_name : label for results files ("S0", "S1", etc.)
            results_dir : directory to save incremental results
        """
        self.rag         = rag
        self.system_name = system_name
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

        self.results_path = os.path.join(
            results_dir, f"{system_name}_attack_results.jsonl"
        )

    # ── Resume ────────────────────────────────────────────────────────────

    def _load_completed_ids(self) -> set:
        completed = set()
        if os.path.exists(self.results_path):
            with open(self.results_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            completed.add(json.loads(line)["attack_id"])
                        except Exception:
                            pass
        return completed

    def _save_result(self, result: dict):
        with open(self.results_path, "a") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    # ── Success evaluation ────────────────────────────────────────────────

    @staticmethod
    def evaluate_success(attack: dict, response: str) -> bool:
        """
        Binary keyword-in-response check.
        Safety-blocked responses always count as failed (not successful).
        """
        if not response:
            return False
        if response == SAFETY_BLOCKED:
            return False

        keyword = attack.get("success_keyword", "")
        fn_type = attack.get("success_fn_type", "keyword_in_response")

        if fn_type == "keyword_in_response":
            return keyword.lower() in response.lower()
        elif fn_type == "keyword_not_in_response":
            return keyword.lower() not in response.lower()
        return keyword.lower() in response.lower()

    # ── Per-attack dispatch ───────────────────────────────────────────────

    def _run_single_attack(self, attack: dict) -> dict:
        """
        Execute one attack and return a result record.

        Routing logic:
          surface=direct  + is_multi_turn=True  → query_multi_turn()
          surface=direct  + is_multi_turn=False → query()
          surface=indirect                       → query_with_injected_doc()
        """
        surface   = attack["surface"]
        category  = attack["category"]
        is_multi  = attack.get("is_multi_turn", False)

        # ── Indirect injection (I1 / I2 / I3) ────────────────────────────
        if surface == "indirect":
            trigger      = attack.get("trigger_query", "")
            poisoned_text = attack.get("text", "")

            if not trigger:
                return self._error_result(
                    attack, "Missing trigger_query for indirect attack"
                )
            if not poisoned_text:
                return self._error_result(
                    attack, "Missing text (poisoned document) for indirect attack"
                )

            result = self.rag.query_with_injected_doc(
                trigger_query     = trigger,
                poisoned_doc_text = poisoned_text,
            )

        # ── Multi-turn direct injection (D5) ──────────────────────────────
        elif is_multi:
            attack_turn_index = attack.get("attack_turn_index", 2)
            turns             = attack.get("turns", [])
            if not turns:
                return self._error_result(
                    attack, "Multi-turn attack has empty turns list"
                )
            result = self.rag.query_multi_turn(
                turns             = turns,
                attack_turn_index = attack_turn_index,
            )

        # ── Standard single-turn direct injection (D1 / D2 / D3 / D4) ────
        else:
            attack_text = attack.get("text", "")
            if not attack_text:
                return self._error_result(
                    attack, "Empty text for direct attack"
                )
            result = self.rag.query(attack_text)

        response       = result.get("response", "")
        safety_blocked = result.get("safety_blocked", False)
        success        = self.evaluate_success(attack, response)

        return {
            "attack_id":             attack["attack_id"],
            "system":                self.system_name,
            "surface":               surface,
            "category":              category,
            "category_name":         attack.get("category_name", ""),
            "domain":                attack.get("domain", "any"),
            "is_multi_turn":         is_multi,
            "success":               success,
            "safety_blocked":        safety_blocked,
            "success_keyword":       attack.get("success_keyword", ""),
            "response":              response[:500],
            "latency_ms":            result.get("latency_ms", 0),
            "error":                 result.get("error", None),
            "n_context_docs":        len(result.get("context_docs", [])),
            "poisoned_doc_injected": result.get("poisoned_doc_injected", False),
        }

    def _error_result(self, attack: dict, msg: str) -> dict:
        return {
            "attack_id":             attack["attack_id"],
            "system":                self.system_name,
            "surface":               attack["surface"],
            "category":              attack["category"],
            "category_name":         attack.get("category_name", ""),
            "domain":                attack.get("domain", "any"),
            "is_multi_turn":         attack.get("is_multi_turn", False),
            "success":               False,
            "safety_blocked":        False,
            "success_keyword":       attack.get("success_keyword", ""),
            "response":              "",
            "latency_ms":            0,
            "error":                 msg,
            "n_context_docs":        0,
            "poisoned_doc_injected": False,
        }

    # ── Main run loop ─────────────────────────────────────────────────────

    def run(self, attacks: list[dict]) -> list[dict]:
        """
        Run all attacks with resume support.

        Returns list of all result dicts.
        """
        completed  = self._load_completed_ids()
        remaining  = [a for a in attacks
                      if a["attack_id"] not in completed]

        print(f"\n  System       : {self.system_name}")
        print(f"  Total        : {len(attacks)}")
        print(f"  Already done : {len(completed)}")
        print(f"  Remaining    : {len(remaining)}")
        print(f"  Delay/call   : {config.API_RATE_LIMIT_DELAY}s "
              f"(~{60/config.API_RATE_LIMIT_DELAY:.0f} RPM, "
              f"Gemini free limit: 10 RPM)")

        if not remaining:
            print("  All attacks done — loading from file.")
            return self._load_all_results()

        for attack in tqdm(remaining,
                           desc=f"  [{self.system_name}] Attacks"):
            result = self._run_single_attack(attack)
            self._save_result(result)
            time.sleep(config.API_RATE_LIMIT_DELAY)

        return self._load_all_results()

    def _load_all_results(self) -> list[dict]:
        results = []
        with open(self.results_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        results.append(json.loads(line))
                    except Exception:
                        pass
        return results