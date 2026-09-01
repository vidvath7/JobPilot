"""MCP Prompt for a reusable, grounded application-preparation workflow.

Unlike Tools and Resources, this capability returns model-facing instructions.
It intentionally performs no job lookup, profile read, scoring, persistence, LLM
call, or orchestration; a future Host will decide how to execute the workflow.
"""

from mcp import types


PREPARE_APPLICATION_NAME = "prepare_application"
PREPARE_APPLICATION_DESCRIPTION = (
    "Provides a reusable application-preparation workflow for a selected "
    "JobPilot job."
)


def prepare_application(job_id: str) -> list[types.PromptMessage]:
    """Return grounded preparation instructions for the selected job ID.

    ``job_id`` is interpolated only as workflow context. Returning instructions
    rather than executing capabilities preserves the MCP distinction between a
    Prompt and server-side Tool orchestration.
    """
    instructions = f"""Prepare application guidance for JobPilot job {job_id}.

Follow this grounded workflow:
1. Retrieve the complete job details for {job_id}.
2. Retrieve the candidate profile.
3. Use deterministic job-match evidence where useful.
4. Identify strong candidate-job matches and the genuine candidate experience and skills that should be emphasized.
5. Identify gaps and missing requirements, clearly distinguishing them from supported candidate evidence.
6. Prepare resume-tailoring guidance based only on supported evidence.
7. Prepare cover-letter guidance describing which genuine qualifications and motivations to emphasize.

Do not fabricate skills, employment history, education, certifications, achievements, or any other candidate evidence."""

    # One user message is sufficient for this reusable workflow. Additional
    # conversation turns would imply orchestration that belongs in the Host.
    return [
        types.PromptMessage(
            role="user",
            content=types.TextContent(type="text", text=instructions),
        )
    ]
