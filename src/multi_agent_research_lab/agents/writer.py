"""Writer agent.

Responsibilities:
  1. Read state.research_notes and state.analysis_notes from shared state.
  2. Use the LLM to draft a well-structured final answer with citations.
  3. Write the result to state.final_answer.
"""

from __future__ import annotations

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient, load_agent_config

logger = logging.getLogger(__name__)

_WRITER_SYSTEM = """\
You are a professional technical writer. Your job is to produce a clear,
well-structured response for the user based on research notes and analysis.

Guidelines:
- Write for the specified audience (technical or non-technical).
- Use headings (##) to organise sections.
- Include inline citations where relevant, e.g. [1], [2].
- Append a "## Sources" section at the end listing all cited URLs.
- Be accurate; do not add information not present in the notes.
- Target length: 400-800 words.
"""


class WriterAgent(BaseAgent):
    """Produces the final answer from research and analysis notes."""

    name = "writer"

    def __init__(self) -> None:
        cfg = load_agent_config("writer")
        self._llm = LLMClient()
        self._temperature: float = cfg.get("temperature", 0.4)

    def run(self, state: ResearchState) -> ResearchState:
        """Populate ``state.final_answer`` from research and analysis notes."""

        with trace_span("writer.run", {"query": state.request.query}) as span:
            # ── 1. Gather inputs from shared state ────────────────────────────
            research_notes = state.research_notes or ""
            analysis_notes = state.analysis_notes or ""

            if not research_notes and not analysis_notes:
                logger.warning("WriterAgent: no notes available — writing fallback answer")
                state.final_answer = (
                    "Insufficient research data to generate a complete answer. "
                    "Please ensure the Researcher and Analyst agents have run first."
                )
                return state

            # ── 2. Build source citation list ─────────────────────────────────
            source_lines = "\n".join(
                f"[{i + 1}] {src.title} — {src.url or 'no url'}"
                for i, src in enumerate(state.sources)
            )

            # ── 3. Compose the user prompt ────────────────────────────────────
            user_prompt = (
                f"User query: {state.request.query}\n"
                f"Audience: {state.request.audience}\n\n"
                f"## Research Notes\n{research_notes}\n\n"
                + (f"## Analysis Notes\n{analysis_notes}\n\n" if analysis_notes else "")
                + (f"## Available Sources\n{source_lines}\n\n" if source_lines else "")
                + "Write the final answer now."
            )

            # ── 4. Generate final answer ──────────────────────────────────────
            with trace_span("writer.generate"):
                response = self._llm.complete(
                    system_prompt=_WRITER_SYSTEM,
                    user_prompt=user_prompt,
                    temperature=self._temperature,
                )

            state.final_answer = response.content
            span["attributes"]["answer_length"] = len(response.content)

            logger.info(
                "WriterAgent: final answer generated "
                "(input_tokens=%s output_tokens=%s cost_usd=%s)",
                response.input_tokens,
                response.output_tokens,
                f"{response.cost_usd:.6f}" if response.cost_usd is not None else "unknown",
            )

            # ── 5. Record agent result ────────────────────────────────────────
            state.agent_results.append(
                AgentResult(
                    agent=AgentName.WRITER,
                    content=response.content,
                    metadata={
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "cost_usd": response.cost_usd,
                    },
                )
            )
            state.add_trace_event(
                "writer_complete",
                {"answer_length": len(response.content)},
            )

        return state
