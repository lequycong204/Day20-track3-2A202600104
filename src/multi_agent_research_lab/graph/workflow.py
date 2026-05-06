"""LangGraph workflow for the multi-agent research system.

Topology (hub-and-spoke):
  All worker nodes route back to the supervisor.
  The supervisor is the ONLY node with conditional edges.

  START → supervisor ←→ researcher
                     ←→ analyst
                     ←→ writer
                     ←→ critic
                     → END
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, StateGraph

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.critic import CriticAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import SupervisorAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span

logger = logging.getLogger(__name__)


# ── Node functions ────────────────────────────────────────────────────────────
# Each node receives the full Pydantic state, runs its agent, and returns it.

_supervisor = SupervisorAgent()
_researcher = ResearcherAgent()
_analyst = AnalystAgent()
_writer = WriterAgent()
_critic = CriticAgent()


def supervisor_node(state: ResearchState) -> ResearchState:
    """Run the supervisor to decide the next route."""
    with trace_span("node.supervisor"):
        return _supervisor.run(state)


def researcher_node(state: ResearchState) -> ResearchState:
    """Run the researcher to collect sources and write research notes."""
    with trace_span("node.researcher"):
        return _researcher.run(state)


def analyst_node(state: ResearchState) -> ResearchState:
    """Run the analyst to produce analytical insights."""
    with trace_span("node.analyst"):
        return _analyst.run(state)


def writer_node(state: ResearchState) -> ResearchState:
    """Run the writer to draft the final answer."""
    with trace_span("node.writer"):
        return _writer.run(state)


def critic_node(state: ResearchState) -> ResearchState:
    """Run the critic to fact-check and score the final answer."""
    with trace_span("node.critic"):
        return _critic.run(state)


# ── Routing function ─────────────────────────────────────────────────────────

def _route_after_supervisor(state: ResearchState) -> str:
    """Read the last route decision from the supervisor and return the next node.

    Returns one of: "researcher", "analyst", "writer", "critic", or END.
    """
    if not state.route_history:
        logger.warning("_route_after_supervisor: empty route_history — ending")
        return END

    last_route = state.route_history[-1]
    if last_route == "done":
        return END

    valid_nodes = {"researcher", "analyst", "writer", "critic"}
    if last_route in valid_nodes:
        return last_route

    logger.warning("_route_after_supervisor: unknown route %r — ending", last_route)
    return END


# ── Workflow class ────────────────────────────────────────────────────────────

class MultiAgentWorkflow:
    """Builds and runs the multi-agent LangGraph graph.

    Keep orchestration here; keep agent internals in ``agents/``.
    """

    def __init__(self) -> None:
        self._app = self.build()

    def build(self) -> Any:
        """Create and compile the LangGraph state graph.

        Graph topology (hub-and-spoke):
          - Entry point: supervisor
          - Supervisor → conditional edges → {researcher, analyst, writer, critic, END}
          - All workers → fixed edge → supervisor
        """
        graph = StateGraph(ResearchState)

        # ── Add nodes ─────────────────────────────────────────────────────────
        graph.add_node("supervisor", supervisor_node)
        graph.add_node("researcher", researcher_node)
        graph.add_node("analyst", analyst_node)
        graph.add_node("writer", writer_node)
        graph.add_node("critic", critic_node)

        # ── Entry point ──────────────────────────────────────────────────────
        graph.set_entry_point("supervisor")

        # ── Supervisor → conditional routing ──────────────────────────────────
        graph.add_conditional_edges(
            "supervisor",
            _route_after_supervisor,
            {
                "researcher": "researcher",
                "analyst": "analyst",
                "writer": "writer",
                "critic": "critic",
                END: END,
            },
        )

        # ── All workers → back to supervisor ──────────────────────────────────
        graph.add_edge("researcher", "supervisor")
        graph.add_edge("analyst", "supervisor")
        graph.add_edge("writer", "supervisor")
        graph.add_edge("critic", "supervisor")

        logger.info("MultiAgentWorkflow: graph built and compiled")
        return graph.compile()

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the compiled graph and return the final ResearchState.

        Args:
            state: The initial state containing at least `request`.

        Returns:
            The final state after the workflow completes.
        """
        logger.info("MultiAgentWorkflow.run | query=%r", state.request.query)

        with trace_span("workflow.run", {"query": state.request.query}) as span:
            result = self._app.invoke(state)

            # LangGraph returns a Pydantic model when schema is Pydantic
            if isinstance(result, ResearchState):
                final_state = result
            elif isinstance(result, dict):
                final_state = ResearchState(**result)
            else:
                raise TypeError(f"Unexpected result type: {type(result)}")

            span["attributes"]["iterations"] = final_state.iteration
            span["attributes"]["rewrite_count"] = final_state.rewrite_count
            span["attributes"]["has_final_answer"] = final_state.final_answer is not None

        logger.info(
            "MultiAgentWorkflow.run complete | iterations=%d rewrites=%d",
            final_state.iteration,
            final_state.rewrite_count,
        )
        return final_state
