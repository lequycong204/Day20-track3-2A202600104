
from multi_agent_research_lab.agents import SupervisorAgent
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState


def test_supervisor_routes_to_researcher_first() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    result = SupervisorAgent().run(state)

    assert result is state
    assert state.route_history == ["researcher"]
    assert state.iteration == 1
    assert state.trace[-1]["payload"]["route"] == "researcher"
