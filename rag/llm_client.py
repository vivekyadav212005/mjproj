# rag/llm_client.py
"""
LLM client — local Ollama backend with Gemini-compatible wrapper names.

Interface:
    client = GeminiClient()
    response_str = client.chat(messages, system_prompt)

    messages: list of {"role": "user"/"assistant", "content": str}
    system_prompt: accepted for compatibility with the existing pipeline
"""

import json
from urllib import error as urllib_error
from urllib import request as urllib_request

import config

# Sentinel returned when Gemini's safety filter blocks a response.
SAFETY_BLOCKED = "[GEMINI_SAFETY_BLOCKED]"


class LLMError(Exception):
    """Retained for compatibility with the existing pipeline."""
    pass


class GeminiClient:
    """
    Ollama-backed client that preserves the existing GeminiClient name.
    """

    def __init__(self):
        self.model_name = "llama3:latest"
        self.api_url = "http://localhost:11434/api/generate"
        self.last_error = None

    @staticmethod
    def _build_prompt(messages: list[dict], system_prompt: str = "") -> str:
        prompt_parts = []

        if system_prompt:
            prompt_parts.append(f"System: {system_prompt.strip()}")

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "assistant":
                label = "Assistant"
            elif role == "system":
                label = "System"
            else:
                label = "User"

            prompt_parts.append(f"{label}: {content}")

        prompt_parts.append("Assistant:")
        return "\n".join(prompt_parts)

    def chat(self,
             messages: list[dict],
             system_prompt: str = config.RAG_SYSTEM_PROMPT) -> str:
        """
        Send a non-streaming generate request to the local Ollama server.

        Args:
            messages    : list of {"role": "user"/"assistant", "content": str}
            system_prompt: accepted for compatibility; folded into the prompt

        Returns:
            str: the model's response text, or empty string on failure
        """
        self.last_error = None
        prompt = self._build_prompt(messages, system_prompt=system_prompt)
        payload = json.dumps(
            {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
            }
        ).encode("utf-8")

        req = urllib_request.Request(
            self.api_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib_request.urlopen(req, timeout=120) as response:
                body = response.read().decode("utf-8")
            data = json.loads(body)
            return data.get("response", "") or ""
        except (urllib_error.URLError, urllib_error.HTTPError, json.JSONDecodeError) as e:
            self.last_error = str(e)
            return ""
        except Exception as e:
            self.last_error = str(e)
            return ""


def get_llm_client() -> GeminiClient:
    """Factory — returns the local Ollama-backed client."""
    if config.LLM_PROVIDER == "gemini":
        return GeminiClient()
    else:
        raise ValueError(
            f"LLM_PROVIDER='{config.LLM_PROVIDER}' not supported. "
            "Only 'gemini' is supported. Set LLM_PROVIDER='gemini' in config.py"
        )