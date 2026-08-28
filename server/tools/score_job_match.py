"""MCP-facing adapter for deterministic candidate-to-job matching.

The adapter is deliberately protocol-thin: ``MatchingService`` owns the frozen
scoring algorithm and result contract, while this module only exposes it to MCP.
"""

from typing import Any

if __package__ == "server.tools":
    from ..services.matching_service import MatchingService
else:
    from services.matching_service import MatchingService


_matching_service = MatchingService()


def score_job_match(job_id: str) -> dict[str, Any]:
    """Evaluate a known JobPilot job using deterministic weighted scoring.

    Returns component scores and evidence for skills, role, experience level, and
    location against the candidate profile.
    """
    # Forward the identifier unchanged and return the frozen domain result without
    # adding protocol-specific fields or duplicating matching decisions.
    return _matching_service.score_job_match(job_id)
