# rag/llm_client.py
"""
LLM client — Google Gemini 2.5 Flash using the new google-genai SDK.

The old `google.generativeai` package (google-generativeai on PyPI) is
deprecated and no longer receives updates. This file uses the new
`google.genai` package (google-genai on PyPI) which is the current
official SDK.

Interface:
    client = GeminiClient()
    response_str = client.chat(messages, system_prompt)

    messages: list of {"role": "user"/"assistant", "content": str}
    system_prompt: str — passed as system_instruction
"""

import time
import config

try:
    from google import genai
    from google.genai import types as genai_types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Sentinel returned when Gemini's safety filter blocks a response.
SAFETY_BLOCKED = "[GEMINI_SAFETY_BLOCKED]"


class LLMError(Exception):
    """Raised when an LLM call fails after all retries."""
    pass


class GeminiClient:
    """
    Google Gemini 2.5 Flash using the official google-genai SDK.

    Free tier limits (April 2026):
        gemini-2.5-flash: 10 RPM, 250 RPD
    With API_RATE_LIMIT_DELAY=6.5s we send ~9 RPM — safely within limits.
    """

    def __init__(self):
        if not GEMINI_AVAILABLE:
            raise ImportError(
                "google-genai not installed.\n"
                "Run: pip install google-genai>=1.0.0\n"
                "(Note: this is 'google-genai', NOT 'google-generativeai')"
            )
        if not config.GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY not set.\n"
                "Get a free key at https://aistudio.google.com\n"
                "Then: export GEMINI_API_KEY='AIza...'"
            )
        # Create client once — reused for all calls
        self._client    = genai.Client(api_key=config.GEMINI_API_KEY)
        self.model_name = config.GEMINI_MODEL

    def chat(self,
             messages: list[dict],
             system_prompt: str = config.RAG_SYSTEM_PROMPT) -> str:
        """
        Send a chat request to Gemini 2.5 Flash.

        Args:
            messages    : list of {"role": "user"/"assistant", "content": str}
            system_prompt: passed as system_instruction to the model

        Returns:
            str: the model's response text, or SAFETY_BLOCKED sentinel

        Raises:
            LLMError: if all retries exhausted
        """
        # Convert to Gemini Content format
        # New SDK uses "model" role instead of "assistant"
        gemini_contents = []
        for msg in messages:
            role    = "model" if msg["role"] == "assistant" else msg["role"]
            content = msg["content"]
            gemini_contents.append(
                genai_types.Content(
                    role  = role,
                    parts = [genai_types.Part(text=content)],
                )
            )

        gen_config = genai_types.GenerateContentConfig(
            system_instruction = system_prompt,
            temperature        = config.LLM_TEMPERATURE,
            max_output_tokens  = config.LLM_MAX_TOKENS,
        )

        for attempt in range(1, config.API_MAX_RETRIES + 1):
            try:
                response = self._client.models.generate_content(
                    model    = self.model_name,
                    contents = gemini_contents,
                    config   = gen_config,
                )

                # Handle safety block
                if not response.candidates:
                    return SAFETY_BLOCKED

                candidate   = response.candidates[0]
                finish_name = candidate.finish_reason.name \
                              if hasattr(candidate.finish_reason, 'name') \
                              else str(candidate.finish_reason)

                if finish_name == "SAFETY":
                    return SAFETY_BLOCKED

                # response.text raises if no valid parts — handle gracefully
                try:
                    text = response.text
                    return text if text else ""
                except Exception:
                    return SAFETY_BLOCKED

            except Exception as e:
                err_str = str(e).lower()

                # Rate limit — back off
                if "429" in str(e) or "quota" in err_str \
                        or "resource_exhausted" in err_str:
                    wait = 60 * attempt    # 60s → 120s → 180s
                    print(f"\n    [Rate limit] Waiting {wait}s "
                          f"(attempt {attempt}/{config.API_MAX_RETRIES})...")
                    time.sleep(wait)
                    continue

                if attempt == config.API_MAX_RETRIES:
                    raise LLMError(
                        f"Gemini call failed after "
                        f"{config.API_MAX_RETRIES} attempts: {e}"
                    ) from e

                time.sleep(attempt * 3)

        return ""


def get_llm_client() -> GeminiClient:
    """Factory — returns configured Gemini client."""
    if config.LLM_PROVIDER == "gemini":
        return GeminiClient()
    else:
        raise ValueError(
            f"LLM_PROVIDER='{config.LLM_PROVIDER}' not supported. "
            "Only 'gemini' is supported. Set LLM_PROVIDER='gemini' in config.py"
        )