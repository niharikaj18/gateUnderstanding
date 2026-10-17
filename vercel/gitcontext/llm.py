"""LLM adapters. Anthropic is the default; any OpenAI-compatible server (incl. Ollama/vLLM) also works."""
from __future__ import annotations

from typing import Protocol

from .config import Settings


class LLM(Protocol):
    name: str

    def complete(self, system: str, prompt: str, max_tokens: int = 1200) -> str: ...


class AnthropicLLM:
    def __init__(self, model: str, api_key: str | None):
        import anthropic

        self.name = f"anthropic:{model}"
        self._c, self._m = anthropic.Anthropic(api_key=api_key), model

    def complete(self, system, prompt, max_tokens=1200):
        r = self._c.messages.create(model=self._m, max_tokens=max_tokens, temperature=0, system=system,
                                    messages=[{"role": "user", "content": prompt}])
        return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")


class OpenAILLM:
    def __init__(self, model: str, api_key: str | None, base_url: str | None):
        from openai import OpenAI

        self.name = f"openai:{model}"
        self._c, self._m = OpenAI(api_key=api_key or "none", base_url=base_url), model

    def complete(self, system, prompt, max_tokens=1200):
        r = self._c.chat.completions.create(model=self._m, max_tokens=max_tokens, temperature=0, messages=[
            {"role": "system", "content": system}, {"role": "user", "content": prompt}])
        return r.choices[0].message.content or ""


def make_llm(s: Settings) -> LLM | None:
    """Returns None when no LLM is configured; callers fall back to extractive answers."""
    if s.llm_provider == "anthropic" and s.anthropic_api_key:
        return AnthropicLLM(s.llm_model, s.anthropic_api_key)
    if s.llm_provider == "openai" and (s.openai_api_key or s.openai_base_url):
        return OpenAILLM(s.llm_model, s.openai_api_key, s.openai_base_url)
    return None
