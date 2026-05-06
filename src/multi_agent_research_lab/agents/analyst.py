"""Analyst agent.

Responsibilities:
  1. Read state.research_notes from shared state.
  2. Use the LLM to extract key claims, compare viewpoints, and flag weak evidence.
  3. Identify knowledge gaps and contradictions across sources.
  4. Write structured analytical insights to state.analysis_notes.
"""

from __future__ import annotations

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient, load_agent_config

logger = logging.getLogger(__name__)

_ANALYST_SYSTEM = """\
You are a rigorous research analyst. Given research notes compiled from web sources,
your job is to produce structured analytical insights.

Your analysis MUST include the following sections:

## Key Claims
List the most important factual claims found in the research notes.
Mark each claim's evidence strength: [Strong] / [Moderate] / [Weak].

## Comparative Analysis
Compare different viewpoints, approaches, or findings across sources.
Highlight agreements and contradictions.

## Knowledge Gaps
Identify questions the research notes could NOT fully answer.
Explain why this matters for the user's query.

## Critical Assessment
Flag any claims that seem overstated, biased, or poorly evidenced.
Note the overall reliability of the source material.

## Synthesis
In 2-3 sentences, summarise the most defensible conclusion the evidence supports.

Be concise, precise, and analytical. Do not repeat the research notes verbatim.
"""


class AnalystAgent(BaseAgent):
    """Turns research notes into structured analytical insights."""

    name = "analyst"

    def __init__(self) -> None:
        cfg = load_agent_config("analyst")
        self._llm = LLMClient()
        self._temperature: float = cfg.get("temperature", 0.1)

    def run(self, state: ResearchState) -> ResearchState:
        """Populate ``state.analysis_notes`` from ``state.research_notes``."""

        with trace_span("analyst.run", {"query": state.request.query}) as span:
            # ── Guard: researcher must run first ─────────────────────────────
            if not state.research_notes:
                logger.warning("AnalystAgent: no research_notes available — skipping analysis")
                state.errors.append(
                    "AnalystAgent: called before ResearcherAgent produced research_notes."
                )
                return state

            # ── 1. Compose analysis prompt ────────────────────────────────────
            source_list = "\n".join(
                f"[{i + 1}] {src.title} — {src.url or 'no url'}"
                for i, src in enumerate(state.sources)
            )

            user_prompt = (
                f"User query: {state.request.query}\n"
                f"Audience: {state.request.audience}\n\n"
                f"## Research Notes\n{state.research_notes}\n\n"
                + (f"## Sources\n{source_list}\n\n" if source_list else "")
                + "Produce your structured analytical insights now."
            )

            # ── 2. Generate analysis ──────────────────────────────────────────
            with trace_span("analyst.analyse"):
                response = self._llm.complete(
                    system_prompt=_ANALYST_SYSTEM,
                    user_prompt=user_prompt,
                    temperature=self._temperature,
                )

            state.analysis_notes = response.content
            span["attributes"]["analysis_length"] = len(response.content)

            logger.info(
                "AnalystAgent: analysis complete "
                "(input_tokens=%s output_tokens=%s cost_usd=%s)",
                response.input_tokens,
                response.output_tokens,
                f"{response.cost_usd:.6f}" if response.cost_usd is not None else "unknown",
            )

            # ── 3. Record agent result ────────────────────────────────────────
            state.agent_results.append(
                AgentResult(
                    agent=AgentName.ANALYST,
                    content=response.content,
                    metadata={
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "cost_usd": response.cost_usd,
                    },
                )
            )
            state.add_trace_event(
                "analyst_complete",
                {"analysis_length": len(response.content)},
            )

        return state
