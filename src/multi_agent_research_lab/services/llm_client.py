"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

from __future__ import annotations

import logging
import pathlib
from dataclasses import dataclass
from typing import Any

import yaml
from tenacity import retry, stop_after_attempt, wait_exponential

from multi_agent_research_lab.core.config import get_settings

logger = logging.getLogger(__name__)

_CONFIGS_DIR = pathlib.Path(__file__).parents[4] / "configs"

def load_agent_config(agent_name: str) -> dict[str, Any]:
    """Load agent-specific config (model, temperature) from lab_default.yaml.

    Args:
        agent_name: One of 'supervisor', 'researcher', 'analyst', 'writer'.

    Returns:
        Dict with keys like 'model' and 'temperature'.
    """
    config_file = _CONFIGS_DIR / "lab_default.yaml"
    with config_file.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    agents: dict[str, Any] = data.get("agents", {})
    if agent_name not in agents:
        raise KeyError(f"Agent '{agent_name}' not found in {config_file}")
    return agents[agent_name]


# Cost per 1M tokens (USD) for common OpenAI models
_COST_PER_1M: dict[str, dict[str, float]] = {
    "gpt-4o": {"input": 5.0, "output": 15.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.0, "output": 30.0},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "gpt-oss-120b": {"input": 0.09, "output": 0.36}, # Google Vertex AI Pricing
}


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Return estimated USD cost for a completion, or None if model is unknown."""
    for key, prices in _COST_PER_1M.items():
        if model.startswith(key):
            return (input_tokens * prices["input"] + output_tokens * prices["output"]) / 1_000_000
    return None


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class LLMClient:
    """Provider-agnostic LLM client backed by OpenAI.

    All retry, timeout, and token-logging concerns live here so that
    individual agents stay focused on their own logic.
    """

    def __init__(self) -> None:
        settings = get_settings()
        if not settings.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. "
                "Add it to your .env file or set the environment variable."
            )
        
        if not settings.base_url:
            raise ValueError(
                "BASE_URL is not set. "
                "Add it to your .env file or set the environment variable."
            )
        try:
            from openai import OpenAI  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "openai package is not installed. Run: pip install -e '.[llm]'"
            ) from exc

        self._client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.base_url,
            timeout=float(settings.timeout_seconds),
        )

        # ── Wrap with LangSmith tracer if active ──────────────────────────────
        # This makes every LLM call (prompt, response, tokens) visible in
        # LangSmith alongside the LangGraph node traces.
        import os
        if os.environ.get("LANGCHAIN_TRACING_V2") == "true":
            try:
                from langsmith.wrappers import wrap_openai  # type: ignore[import]
                self._client = wrap_openai(self._client)
                logger.debug("LLMClient: OpenAI client wrapped with LangSmith tracer")
            except ImportError:
                logger.debug("LangSmith not installed — OpenAI calls will not be traced")

        self._model = settings.openai_model

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """Return a model completion with retry, timeout, and token logging.

        Args:
            system_prompt: The system instruction for the model.
            user_prompt: The user's input/question.
            temperature: Sampling temperature (0.0 = deterministic).
                         Load from lab_default.yaml via load_agent_config() for consistency.
        """
        logger.debug("LLMClient.complete | model=%s temperature=%s", self._model, temperature)

        response = self._client.chat.completions.create(
            model=self._model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        choice = response.choices[0]
        content = choice.message.content or ""

        input_tokens = response.usage.prompt_tokens if response.usage else None
        output_tokens = response.usage.completion_tokens if response.usage else None
        cost = (
            _estimate_cost(self._model, input_tokens, output_tokens)
            if input_tokens is not None and output_tokens is not None
            else None
        )

        logger.info(
            "LLMClient.complete | model=%s input_tokens=%s output_tokens=%s cost_usd=%s",
            self._model,
            input_tokens,
            output_tokens,
            f"{cost:.6f}" if cost is not None else "unknown",
        )

        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )
