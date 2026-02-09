"""
Crew orchestration: run BullwhipAnalyst, ResilienceAssessor, SurvivalPredictor, HumanCoordinationAgent in sequence.
Shared context (df, model) passed through; results aggregated for Streamlit UI.
"""

from typing import Any, Dict
from . import bullwhip_analyst, resilience_assessor, survival_predictor, human_coordination_agent


def run_crew(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run theory-governed agents in order. Context must include 'df'.
    Optionally 'model' (pre-fitted Cox) to avoid refit.
    """
    results = {}
    results["bullwhip"] = bullwhip_analyst.run(context)
    results["resilience"] = resilience_assessor.run(context)
    results["survival"] = survival_predictor.run(context)
    context["hazard_ratios"] = results["survival"].get("hazard_ratios") or {}
    results["human_coordination"] = human_coordination_agent.run(context)
    return results
