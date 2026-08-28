"""Deterministic candidate-to-job scoring in the application-service layer.

This module composes ``ProfileService`` and ``JobService`` rather than reading
JSON itself. It intentionally contains no MCP or model logic: the same explainable
score can later be exposed through any interface without changing its meaning.
"""

import re
from typing import Any

from server.services.job_service import JobService
from server.services.profile_service import ProfileService


_WEIGHTS = {
    "skills": 0.50,
    "role": 0.20,
    "experience_level": 0.20,
    "location": 0.10,
}

# These aliases express spelling equivalence only. Related technologies are not
# collapsed together, which keeps matching predictable and interview-explainable.
_SKILL_ALIASES = {
    "ml": "machine learning",
    "machine learning": "machine learning",
    "llm": "large language model",
    "llms": "large language model",
    "large language model": "large language model",
    "large language models": "large language model",
    "js": "javascript",
    "javascript": "javascript",
}

_ROLE_ALIASES = {
    "ml engineer": "machine learning engineer",
}

# The current fixtures justify one explicit family: application-oriented AI/ML
# engineering. Other engineer titles remain distinct unless meaningful domain
# tokens overlap with a preferred role.
_ROLE_FAMILIES = (
    frozenset(
        {
            "ai engineer",
            "applied ai engineer",
            "machine learning engineer",
        }
    ),
)

# Generic seniority and occupation words do not provide meaningful alignment on
# their own; otherwise every role containing "engineer" would receive points.
_GENERIC_ROLE_TOKENS = {
    "applied",
    "developer",
    "engineer",
    "junior",
    "level",
    "mid",
    "senior",
    "scientist",
}

_SEPARATORS = re.compile(r"[-_/\s,]+")


class MatchingService:
    """Calculate reproducible match components and human-readable evidence."""

    def __init__(
        self,
        profile_service: ProfileService | None = None,
        job_service: JobService | None = None,
    ) -> None:
        """Accept service dependencies so scoring can be tested with controlled data."""
        self._profile_service = profile_service or ProfileService()
        self._job_service = job_service or JobService()

    def score_job_match(self, job_id: str) -> dict[str, Any]:
        """Score one known job using the approved four weighted components."""
        profile = self._profile_service.get_profile()
        # JobNotFoundError deliberately propagates: an unknown job is an invalid
        # lookup, not a real job with a zero-percent match.
        job = self._job_service.get_job(job_id)

        skills_score, matched_skills, missing_skills = _score_skills(
            profile["skills"], job["required_skills"]
        )
        role_score, role_match = _score_role(
            job["title"], profile["preferred_roles"]
        )
        experience_score, experience_match = _score_experience(
            job["experience_level"], profile["preferred_experience_levels"]
        )
        location_score, location_match = _score_location(
            job["location"], profile["preferred_locations"]
        )

        components = {
            "skills": skills_score,
            "role": role_score,
            "experience_level": experience_score,
            "location": location_score,
        }
        # Calculate from the rounded, reported components so consumers can
        # reproduce the final score exactly from the returned evidence.
        score = round(
            components["skills"] * _WEIGHTS["skills"]
            + components["role"] * _WEIGHTS["role"]
            + components["experience_level"] * _WEIGHTS["experience_level"]
            + components["location"] * _WEIGHTS["location"],
            2,
        )

        return {
            "job_id": job["id"],
            "job_title": job["title"],
            "company": job["company"],
            "score": score,
            # Return a copy so consumers can inspect the formula without sharing
            # mutable module-level configuration.
            "weights": dict(_WEIGHTS),
            "components": components,
            "evidence": {
                "matched_required_skills": matched_skills,
                "missing_required_skills": missing_skills,
                "role_match": role_match,
                "experience_match": experience_match,
                "location_match": location_match,
            },
        }


def _normalize_text(value: str) -> str:
    """Normalize case, surrounding whitespace, and obvious separator variants."""
    return _SEPARATORS.sub(" ", value.strip().casefold()).strip()


def _canonicalize_skill(skill: str) -> str:
    """Map only explicitly approved spelling aliases to a canonical skill name."""
    normalized_skill = _normalize_text(skill)
    return _SKILL_ALIASES.get(normalized_skill, normalized_skill)


def _normalize_role(role: str) -> str:
    """Normalize a role and expand the small explicit role alias table."""
    normalized_role = _normalize_text(role)
    return _ROLE_ALIASES.get(normalized_role, normalized_role)


def _score_skills(
    candidate_skills: list[str], required_skills: list[str]
) -> tuple[float, list[str], list[str]]:
    """Score required-skill coverage while preserving job labels as evidence."""
    if not required_skills:
        # No stated requirements provide no positive matching evidence; scoring
        # zero avoids granting an unearned perfect skills component.
        return 0.0, [], []

    normalized_candidate_skills = {
        _canonicalize_skill(skill) for skill in candidate_skills
    }
    matched_skills = []
    missing_skills = []

    for required_skill in required_skills:
        if _canonicalize_skill(required_skill) in normalized_candidate_skills:
            matched_skills.append(required_skill)
        else:
            missing_skills.append(required_skill)

    score = round(len(matched_skills) / len(required_skills) * 100, 2)
    return score, matched_skills, missing_skills


def _score_role(
    job_title: str, preferred_roles: list[str]
) -> tuple[float, dict[str, Any]]:
    """Return role points plus stable, machine-readable match evidence."""
    normalized_title = _normalize_role(job_title)
    normalized_preferences = [
        (preferred_role, _normalize_role(preferred_role))
        for preferred_role in preferred_roles
    ]

    for preferred_role, normalized_preference in normalized_preferences:
        if normalized_title == normalized_preference:
            return 100.0, {
                "job_role": job_title,
                "matched_preference": preferred_role,
                "match_type": "exact",
            }

    for preferred_role, normalized_preference in normalized_preferences:
        if any(
            normalized_title in role_family
            and normalized_preference in role_family
            for role_family in _ROLE_FAMILIES
        ):
            return 70.0, {
                "job_role": job_title,
                "matched_preference": preferred_role,
                "match_type": "family",
            }

    title_tokens = set(normalized_title.split()) - _GENERIC_ROLE_TOKENS
    for preferred_role, normalized_preference in normalized_preferences:
        preferred_tokens = (
            set(normalized_preference.split()) - _GENERIC_ROLE_TOKENS
        )
        if title_tokens & preferred_tokens:
            return 40.0, {
                "job_role": job_title,
                "matched_preference": preferred_role,
                "match_type": "token_overlap",
            }

    return 0.0, {
        "job_role": job_title,
        "matched_preference": None,
        "match_type": "none",
    }


def _score_experience(
    job_experience_level: str, preferred_experience_levels: list[str]
) -> tuple[float, dict[str, Any]]:
    """Return binary experience points with the source preferences preserved."""
    normalized_job_level = _normalize_text(job_experience_level)

    for preferred_level in preferred_experience_levels:
        if normalized_job_level == _normalize_text(preferred_level):
            return 100.0, {
                "job_level": job_experience_level,
                "candidate_preferences": list(preferred_experience_levels),
                "matched": True,
            }

    return 0.0, {
        "job_level": job_experience_level,
        "candidate_preferences": list(preferred_experience_levels),
        "matched": False,
    }


def _score_location(
    job_location: str, preferred_locations: list[str]
) -> tuple[float, dict[str, Any]]:
    """Return location points and the actual preference responsible for a match."""
    normalized_job_location = _normalize_text(job_location)

    for preferred_location in preferred_locations:
        normalized_preference = _normalize_text(preferred_location)
        if normalized_preference in normalized_job_location:
            return 100.0, {
                "job_location": job_location,
                "candidate_preferences": list(preferred_locations),
                "matched_preference": preferred_location,
                "matched": True,
            }

    return 0.0, {
        "job_location": job_location,
        "candidate_preferences": list(preferred_locations),
        "matched_preference": None,
        "matched": False,
    }
