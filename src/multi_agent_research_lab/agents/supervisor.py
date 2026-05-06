"""Supervisor / router agent.

The Supervisor is the only node that decides which worker runs next.
It pre-processes the user query (splitting multi-part requests into sub-tasks)
and routes through researcher → analyst → writer → critic in order.

Routing table
─────────────
  researcher  → if sources are still missing
  analyst     → if research_notes exist but analysis_notes is still missing
  writer      → if analysis_notes exists but final_answer is still missing
  critic      → if final_answer exists but has not yet been reviewed this cycle
  writer      → if critic rejected AND rewrite_count < 2
  done        → if critic approved OR rewrite_count >= 2 OR max_iterations exceeded
"""

from __future__ import annotations

import json
import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import AgentName
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient, load_agent_config

logger = logging.getLogger(__name__)

_MAX_REWRITES = 2

# ──────────────────── System prompts ──────────────────────────

_DECOMPOSE_SYSTEM = """\
You are a research orchestrator. Your job is to analyse a user query and
decide whether it contains multiple distinct sub-questions or requirements.

Return a JSON object with exactly two keys:
  "sub_tasks": a list of strings, each a focused, self-contained question.
               If the query is already atomic, return a list with ONE item.
  "reasoning": a one-sentence explanation of your decomposition.

Return ONLY valid JSON. Do not include markdown fences.
"""

_ROUTE_SYSTEM = """\
You are a research supervisor deciding which specialist agent should run next.

Available agents: researcher, analyst, writer, critic, done.
- researcher : fetches web sources and writes research_notes.
- analyst    : interprets research_notes and writes analysis_notes.
- writer     : drafts the final_answer from analysis_notes.
- critic     : fact-checks the final_answer and assigns a quality score.
- done       : signals that the workflow is complete.

Rules:
1. Always run researcher before analyst, analyst before writer, writer before critic.
2. If the critic has approved the answer, return "done".
3. If max_iterations is close, skip straight to writer then done.
4. Return ONLY one of the five strings above, nothing else.
"""


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop.

    Pre-processes the incoming query with an LLM to detect multi-part
    requests and stores the decomposed sub-tasks on the state so that
    downstream agents can iterate over them.

    Critic is mandatory. Writer may be re-invoked up to 2 times if
    the critic rejects the answer.
    """

    name = "supervisor"

    def __init__(self) -> None:
        cfg = load_agent_config("supervisor")
        self._llm = LLMClient()
        self._temperature: float = cfg.get("temperature", 0.0)
        self._max_iterations: int = get_settings().max_iterations

    # ── Public interface ──────────────────────────────────────────────────────

    def run(self, state: ResearchState) -> ResearchState:
        """Pre-process query (first call) then route to the next agent."""

        # ── Step 1: Query decomposition (only on first call) ─────────────────
        if state.iteration == 0:
            state = self._decompose_query(state)

        # ── Step 2: Guard against runaway loops ──────────────────────────────
        if state.iteration >= self._max_iterations:
            logger.warning(
                "SupervisorAgent: max_iterations (%d) reached — forcing done",
                self._max_iterations,
            )
            state.record_route("done")
            state.add_trace_event(
                "supervisor_route",
                {"route": "done", "reason": "max_iterations_exceeded"},
            )
            return state

        # ── Step 3: Determine next route ─────────────────────────────────────
        route = self._decide_route(state)
        state.record_route(route)
        state.add_trace_event(
            "supervisor_route",
            {
                "route": route,
                "iteration": state.iteration,
                "rewrite_count": state.rewrite_count,
                "has_sources": bool(state.sources),
                "has_research_notes": state.research_notes is not None,
                "has_analysis_notes": state.analysis_notes is not None,
                "has_final_answer": state.final_answer is not None,
            },
        )
        logger.info(
            "SupervisorAgent.run | iteration=%d route=%s rewrite_count=%d",
            state.iteration,
            route,
            state.rewrite_count,
        )
        return state

    # ── Private helpers ───────────────────────────────────────────────────────

    def _decompose_query(self, state: ResearchState) -> ResearchState:
        """Use the LLM to split a multi-part query into atomic sub-tasks.

        The decomposed sub-tasks are stored as a JSON string in
        ``state.trace`` so downstream agents can read them.  The original
        query is never modified.
        """
        user_prompt = (
            f"User query: {state.request.query}\n\n"
            f"Audience: {state.request.audience}\n"
            "Decompose this query into focused sub-tasks."
        )
        try:
            response = self._llm.complete(
                system_prompt=_DECOMPOSE_SYSTEM,
                user_prompt=user_prompt,
                temperature=self._temperature,
            )
            parsed: dict = json.loads(response.content)
            sub_tasks: list[str] = parsed.get("sub_tasks", [state.request.query])
            reasoning: str = parsed.get("reasoning", "")
        except Exception as exc:  # noqa: BLE001
            logger.warning("SupervisorAgent: decomposition failed (%s) — using raw query", exc)
            sub_tasks = [state.request.query]
            reasoning = "Fallback: LLM decomposition unavailable."

        state.add_trace_event(
            "query_decomposition",
            {"sub_tasks": sub_tasks, "reasoning": reasoning},
        )
        logger.info(
            "SupervisorAgent: decomposed into %d sub-task(s): %s",
            len(sub_tasks),
            sub_tasks,
        )
        return state

    def _decide_route(self, state: ResearchState) -> str:
        """Apply a deterministic routing policy, with LLM as a tiebreaker.

        Routing order: researcher → analyst → writer → critic → done.
        Critic can send back to writer (up to _MAX_REWRITES times).
        """
        last_route = state.route_history[-1] if state.route_history else None

        # ── 1. No sources yet → researcher ───────────────────────────────────
        if not state.sources and state.research_notes is None:
            return AgentName.RESEARCHER

        # ── 2. Has research, no analysis → analyst ───────────────────────────
        if state.research_notes is not None and state.analysis_notes is None:
            return AgentName.ANALYST

        # ── 3. Has analysis, no final_answer → writer ────────────────────────
        if state.analysis_notes is not None and state.final_answer is None:
            return AgentName.WRITER

        # ── 4. Has final_answer → send to critic for review ──────────────────
        if state.final_answer is not None and last_route != AgentName.CRITIC:
            # Only send to critic if it wasn't just reviewed
            return AgentName.CRITIC

        # ── 5. After critic: check if approved or max rewrites reached ───────
        if state.final_answer is not None:
            # Critic approved (final_answer still intact)
            return "done"

        if state.final_answer is None and state.rewrite_count >= _MAX_REWRITES:
            # Max rewrites exhausted — force done with whatever we have
            logger.warning(
                "SupervisorAgent: max rewrites (%d) reached — forcing done",
                _MAX_REWRITES,
            )
            # Use the last writer output from agent_results as fallback
            for result in reversed(state.agent_results):
                if result.agent == AgentName.WRITER:
                    state.final_answer = result.content
                    break
            if state.final_answer is None:
                state.final_answer = state.analysis_notes or state.research_notes or "No answer."
            return "done"

        if state.final_answer is None and state.rewrite_count < _MAX_REWRITES:
            # Critic rejected — send back to writer for revision
            state.rewrite_count += 1
            logger.info(
                "SupervisorAgent: critic rejected — sending to writer (rewrite %d/%d)",
                state.rewrite_count,
                _MAX_REWRITES,
            )
            return AgentName.WRITER

        # ── Fallback: ask the LLM to decide ──────────────────────────────────
        return self._llm_route(state)

    def _llm_route(self, state: ResearchState) -> str:
        """Ask the LLM to choose the next route when deterministic rules are ambiguous."""
        user_prompt = (
            f"Query: {state.request.query}\n"
            f"Iteration: {state.iteration}/{self._max_iterations}\n"
            f"Rewrite count: {state.rewrite_count}/{_MAX_REWRITES}\n"
            f"Has sources: {bool(state.sources)}\n"
            f"Has research_notes: {state.research_notes is not None}\n"
            f"Has analysis_notes: {state.analysis_notes is not None}\n"
            f"Has final_answer: {state.final_answer is not None}\n"
            f"Route history: {state.route_history}\n\n"
            "Which agent should run next?"
        )
        valid = {
            AgentName.RESEARCHER, AgentName.ANALYST, AgentName.WRITER,
            AgentName.CRITIC, "done",
        }
        try:
            response = self._llm.complete(
                system_prompt=_ROUTE_SYSTEM,
                user_prompt=user_prompt,
                temperature=self._temperature,
            )
            route = response.content.strip().lower()
            if route not in valid:
                raise ValueError(f"Unexpected route from LLM: {route!r}")
            return route
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "SupervisorAgent: LLM routing failed (%s) — defaulting to done", exc
            )
            return "done"
