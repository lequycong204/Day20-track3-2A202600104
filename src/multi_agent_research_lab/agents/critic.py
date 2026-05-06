"""Critic agent.

Responsibilities:
  1. Read state.final_answer and state.sources.
  2. Use the LLM to fact-check claims against the source snippets.
  3. Detect potential hallucinations (claims not backed by sources).
  4. Check citation coverage (sources mentioned but not cited, or vice-versa).
  5. Produce a quality score (0-10) and optional revision notes.
  6. Append findings to state; optionally clear final_answer to trigger re-write.
"""

from __future__ import annotations

import json
import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient, load_agent_config

logger = logging.getLogger(__name__)

# Score threshold below which the final_answer is cleared for re-writing
_MIN_ACCEPTABLE_SCORE = 6.0

_CRITIC_SYSTEM = """\
You are a strict fact-checker and quality reviewer for AI-generated research answers.

Given:
- The user's original query
- The research source snippets (numbered)
- The generated final answer

Your task:
1. Check every factual claim in the final answer against the source snippets.
2. Flag any claim that cannot be verified from the provided snippets as a potential hallucination.
3. Check citation coverage: are citations [1], [2]... present and correct?
4. Assign an overall quality score from 0 to 10.

Return ONLY a valid JSON object with these keys:
{
  "score": <float 0-10>,
  "hallucinations": ["claim 1 that is unverified", ...],
  "missing_citations": ["fact without citation", ...],
  "revision_notes": "Short paragraph with concrete suggestions for improvement.",
  "approved": <true if score >= 6, false otherwise>
}
"""


class CriticAgent(BaseAgent):
    """Optional fact-checking and safety-review agent."""

    name = "critic"

    def __init__(self) -> None:
        cfg = load_agent_config("critic")
        self._llm = LLMClient()
        self._temperature: float = cfg.get("temperature", 0.0)

    def run(self, state: ResearchState) -> ResearchState:
        """Validate final_answer and append quality findings to state."""

        with trace_span("critic.run", {"query": state.request.query}) as span:
            # ── Guard: nothing to review yet ─────────────────────────────────
            if not state.final_answer:
                logger.warning("CriticAgent: no final_answer to review — skipping")
                state.errors.append("CriticAgent: called before WriterAgent produced an answer.")
                return state

            # ── 1. Build source evidence string ──────────────────────────────
            source_evidence = "\n\n".join(
                f"[{i + 1}] {src.title}\n{src.snippet}"
                for i, src in enumerate(state.sources)
            ) or "No sources available."

            # ── 2. Compose review prompt ──────────────────────────────────────
            user_prompt = (
                f"User query: {state.request.query}\n\n"
                f"## Source snippets\n{source_evidence}\n\n"
                f"## Final answer to review\n{state.final_answer}\n\n"
                "Perform your fact-check and quality review now."
            )

            # ── 3. Call LLM ───────────────────────────────────────────────────
            with trace_span("critic.review"):
                response = self._llm.complete(
                    system_prompt=_CRITIC_SYSTEM,
                    user_prompt=user_prompt,
                    temperature=self._temperature,
                )

            # ── 4. Parse JSON response ────────────────────────────────────────
            try:
                review: dict = json.loads(response.content)
            except json.JSONDecodeError:
                logger.warning("CriticAgent: LLM returned non-JSON — storing raw response")
                review = {
                    "score": 5.0,
                    "hallucinations": [],
                    "missing_citations": [],
                    "revision_notes": response.content,
                    "approved": False,
                }

            score: float = float(review.get("score", 5.0))
            approved: bool = bool(review.get("approved", score >= _MIN_ACCEPTABLE_SCORE))
            hallucinations: list[str] = review.get("hallucinations", [])
            revision_notes: str = review.get("revision_notes", "")

            span["attributes"]["score"] = score
            span["attributes"]["approved"] = approved
            span["attributes"]["hallucination_count"] = len(hallucinations)

            logger.info(
                "CriticAgent: score=%.1f approved=%s hallucinations=%d",
                score,
                approved,
                len(hallucinations),
            )

            # ── 5. If quality too low → clear final_answer for re-write ──────
            if not approved:
                logger.warning(
                    "CriticAgent: score %.1f below threshold %.1f — clearing final_answer "
                    "so WriterAgent can revise",
                    score,
                    _MIN_ACCEPTABLE_SCORE,
                )
                state.final_answer = None
                # Append revision hints to analysis_notes so Writer can improve
                hint = f"\n\n---\n**Critic revision request (score {score}/10):**\n{revision_notes}"
                state.analysis_notes = (state.analysis_notes or "") + hint

            # ── 6. Record result in state ─────────────────────────────────────
            state.agent_results.append(
                AgentResult(
                    agent=AgentName.CRITIC,
                    content=revision_notes or "Approved.",
                    metadata={
                        "score": score,
                        "approved": approved,
                        "hallucinations": hallucinations,
                        "missing_citations": review.get("missing_citations", []),
                    },
                )
            )
            state.add_trace_event(
                "critic_complete",
                {
                    "score": score,
                    "approved": approved,
                    "hallucination_count": len(hallucinations),
                },
            )

        return state
