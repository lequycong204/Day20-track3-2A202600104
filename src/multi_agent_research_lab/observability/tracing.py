"""Tracing hooks with LangSmith integration.

Strategy:
  - If LANGSMITH_API_KEY is set in settings, send spans to LangSmith via RunTree.
  - Otherwise, fall back silently to a local timing-only span.

Agents call trace_span() — they never need to know which backend is active.
LangGraph / LangChain automatic tracing is also enabled via environment variables
set in configure_tracing().
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter
from typing import Any

logger = logging.getLogger(__name__)


def configure_tracing() -> None:
    """Enable LangSmith tracing if API key is available.

    Sets the environment variables that LangChain and LangGraph read to
    automatically send traces to LangSmith. Call this once at startup
    (e.g. from cli._init()).
    """
    from multi_agent_research_lab.core.config import get_settings

    settings = get_settings()
    if not settings.langsmith_api_key:
        logger.info("LangSmith tracing disabled — LANGSMITH_API_KEY not set")
        return

    # Environment variables picked up by LangChain / LangGraph automatically
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project

    logger.info(
        "LangSmith tracing enabled | project=%s", settings.langsmith_project
    )


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Timed span that optionally sends data to LangSmith.

    Always yields a local span dict so callers can read timing and
    attributes regardless of whether LangSmith is configured.

    Usage:
        with trace_span("researcher.search", {"query": q}) as span:
            results = search(q)
            span["attributes"]["num_results"] = len(results)
        # span["duration_seconds"] is set automatically
    """
    started = perf_counter()
    span: dict[str, Any] = {
        "name": name,
        "attributes": dict(attributes or {}),
        "duration_seconds": None,
    }

    # ── Try LangSmith RunTree span ────────────────────────────────────────────
    langsmith_run = _try_start_langsmith_run(name, span["attributes"])

    try:
        yield span
        error: BaseException | None = None
    except BaseException as exc:
        error = exc
        raise
    finally:
        span["duration_seconds"] = perf_counter() - started
        _try_end_langsmith_run(langsmith_run, span, error=error if "error" in dir() else None)
        logger.debug(
            "trace_span | name=%s duration=%.3fs",
            name,
            span["duration_seconds"],
        )


# ── LangSmith helpers (graceful no-op if langsmith not installed) ─────────────

def _try_start_langsmith_run(name: str, inputs: dict[str, Any]) -> Any | None:
    """Start a LangSmith RunTree, returning it or None on any failure."""
    if not os.environ.get("LANGCHAIN_TRACING_V2"):
        return None
    try:
        from langsmith.run_trees import RunTree  # type: ignore[import]

        run = RunTree(
            name=name,
            run_type="chain",
            inputs=inputs,
        )
        run.post()
        return run
    except Exception as exc:  # noqa: BLE001
        logger.debug("LangSmith span start failed (non-fatal): %s", exc)
        return None


def _try_end_langsmith_run(
    run: Any | None,
    span: dict[str, Any],
    error: BaseException | None = None,
) -> None:
    """Patch and end the LangSmith RunTree, or no-op if run is None."""
    if run is None:
        return
    try:
        outputs = {
            "duration_seconds": span["duration_seconds"],
            **span["attributes"],
        }
        if error is not None:
            run.end(outputs=outputs, error=str(error))
        else:
            run.end(outputs=outputs)
        run.patch()
    except Exception as exc:  # noqa: BLE001
        logger.debug("LangSmith span end failed (non-fatal): %s", exc)
