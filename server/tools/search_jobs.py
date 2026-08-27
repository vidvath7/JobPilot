if __package__ == "server.tools":
    from ..services.job_service import JobService
else:
    from services.job_service import JobService


_job_service = JobService()


def search_jobs(
    role: str | None = None,
    location: str | None = None,
    experience_level: str | None = None,
) -> list[dict[str, str]]:
    """Search available JobPilot jobs using optional role, location, and experience-level filters.

    Returns concise job summaries for matching jobs.
    """
    return _job_service.search_jobs(
        role=role,
        location=location,
        experience_level=experience_level,
    )
