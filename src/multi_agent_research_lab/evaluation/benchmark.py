"""Benchmark skeleton for single-agent vs multi-agent."""

from collections.abc import Callable
from time import perf_counter

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState


Runner = Callable[[str], ResearchState]


def run_benchmark(
    run_name: str,
    query: str,
    runner: Runner,
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Measure latency and return a placeholder metric object.

    Includes heuristic quality scoring, estimated token cost, citation coverage, and error rate tracking.
    """

    started = perf_counter()
    state = runner(query)
    latency = perf_counter() - started
    
    # 1. Quality scoring heuristic
    quality_score = 0.0
    if state.final_answer:
        quality_score += 5.0
        if state.sources:
            quality_score += min(5.0, len(state.sources) * 1.5)  # Up to 5 points for sources
            
    # 2. Estimated token cost heuristic (rough estimate based on length)
    total_length = len(state.final_answer or "") + len(state.research_notes or "") + len(state.analysis_notes or "")
    estimated_cost_usd = total_length * 0.000005  # Placeholder cost per character
    
    # 3. Citation coverage and error rate
    error_count = len(state.errors)
    citation_count = len(state.sources)
    notes = f"Iter: {state.iteration} | Refs: {citation_count} | Errs: {error_count}"
    
    if error_count > 0:
        notes += f" | {state.errors[0][:20]}..."

    metrics = BenchmarkMetrics(
        run_name=run_name, 
        latency_seconds=latency,
        estimated_cost_usd=estimated_cost_usd,
        quality_score=quality_score,
        notes=notes
    )
    return state, metrics
