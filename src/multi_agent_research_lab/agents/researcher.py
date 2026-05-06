"""Researcher agent.

Responsibilities:
  1. Read sub-tasks from the supervisor's decomposition (stored in state.trace).
  2. Search the web for each sub-task using SearchClient.
  3. De-duplicate and filter sources.
  4. Use the LLM to synthesise a concise set of research notes from the snippets.
  5. Write sources → state.sources and notes → state.research_notes.
"""

from __future__ import annotations

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient, load_agent_config
from multi_agent_research_lab.services.search_client import SearchClient

logger = logging.getLogger(__name__)

_NOTES_SYSTEM = """\
You are a research assistant. Given a set of web snippets, write concise and
structured research notes that directly address the user's query.

Format:
- Use bullet points grouped by sub-topic.
- Cite sources by number, e.g. [1], [2].
- Be factual; do not invent information not present in the snippets.
- Aim for 200-400 words.
"""


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(self) -> None:
        cfg = load_agent_config("researcher")
        self._llm = LLMClient()
        self._search = SearchClient()
        self._temperature: float = cfg.get("temperature", 0.2)

    def run(self, state: ResearchState) -> ResearchState:
        """Populate ``state.sources`` and ``state.research_notes``."""

        with trace_span("researcher.run", {"query": state.request.query}) as span:
            # ── 1. Determine which sub-tasks to search ────────────────────────
            sub_tasks = self._get_sub_tasks(state)
            span["attributes"]["sub_tasks"] = sub_tasks

            # ── 2. Search for each sub-task ───────────────────────────────────
            all_sources: list[SourceDocument] = []
            for task in sub_tasks:
                with trace_span("researcher.search", {"task": task}):
                    results = self._search.search(
                        query=task,
                        max_results=state.request.max_sources,
                    )
                    all_sources.extend(results)
                    logger.debug("ResearcherAgent: '%s' → %d results", task, len(results))

            # ── 3. De-duplicate by URL ────────────────────────────────────────
            seen_urls: set[str] = set()
            unique_sources: list[SourceDocument] = []
            for src in all_sources:
                key = src.url or src.title
                if key not in seen_urls:
                    seen_urls.add(key)
                    unique_sources.append(src)

            state.sources = unique_sources
            span["attributes"]["num_sources"] = len(unique_sources)
            logger.info("ResearcherAgent: collected %d unique sources", len(unique_sources))

            # ── 4. Synthesise research notes via LLM ─────────────────────────
            notes = self._synthesise_notes(state.request.query, unique_sources)
            state.research_notes = notes

            # ── 5. Record agent result ────────────────────────────────────────
            state.agent_results.append(
                AgentResult(
                    agent=AgentName.RESEARCHER,
                    content=notes,
                    metadata={"num_sources": len(unique_sources)},
                )
            )
            state.add_trace_event(
                "researcher_complete",
                {"num_sources": len(unique_sources), "notes_length": len(notes)},
            )

        return state

    # ── Private helpers ───────────────────────────────────────────────────────

    def _get_sub_tasks(self, state: ResearchState) -> list[str]:
        """Extract sub-tasks from the supervisor's decomposition trace.

        Falls back to the original query if the supervisor hasn't decomposed yet.
        """
        for event in state.trace:
            if event.get("name") == "query_decomposition":
                tasks: list[str] = event["payload"].get("sub_tasks", [])
                if tasks:
                    return tasks
        return [state.request.query]

    def _synthesise_notes(self, query: str, sources: list[SourceDocument]) -> str:
        """Ask the LLM to write structured research notes from the source snippets."""
        if not sources:
            return "No sources found. Unable to generate research notes."

        # Build numbered snippet list for the LLM
        snippets = "\n\n".join(
            f"[{i + 1}] {src.title}\n{src.url or 'no url'}\n{src.snippet}"
            for i, src in enumerate(sources)
        )

        user_prompt = (
            f"Research query: {query}\n\n"
            f"Web snippets:\n{snippets}\n\n"
            "Write structured research notes addressing the query."
        )

        with trace_span("researcher.synthesise"):
            response = self._llm.complete(
                system_prompt=_NOTES_SYSTEM,
                user_prompt=user_prompt,
                temperature=self._temperature,
            )

        logger.info(
            "ResearcherAgent: notes synthesised (input_tokens=%s output_tokens=%s)",
            response.input_tokens,
            response.output_tokens,
        )
        return response.content
