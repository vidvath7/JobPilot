"""MCP Prompt for a reusable, grounded application-preparation workflow.

Unlike Tools and Resources, this capability returns model-facing instructions.
It intentionally performs no job lookup, profile read, scoring, persistence, LLM
call, or orchestration; the Host decides how to execute the workflow.
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
    # Separate candidate evidence from job requirements before drafting guidance:
    # plausible suggestions must never become unsupported candidate claims.
    instructions = f"""Prepare application guidance for JobPilot job {job_id}.

Follow this grounded workflow:
1. Retrieve the complete job details for {job_id}.
2. Retrieve the candidate profile.
3. Use deterministic job-match evidence to explain alignment; scores do not establish additional candidate facts.
4. Build an Evidence Ledger before giving resume or cover-letter guidance:
   A. SUPPORTED: facts explicitly present in retrieved candidate evidence. Identify the supporting profile field or experience for each claim.
   B. GAP: job requirements/preferences and missing requirements with no supporting candidate evidence. Missing evidence is not proof the candidate lacks a qualification.
   C. UNKNOWN / VERIFY: potentially useful information not established by retrieved candidate evidence; ask the candidate to verify it.
   Never convert GAP or UNKNOWN information into SUPPORTED candidate evidence.
5. Identify strong candidate-job matches using A, and discuss gaps and verification questions using B and C.
6. Prepare resume-tailoring guidance using A only for candidate claims.
7. Prepare cover-letter guidance using A only for candidate claims.

Evidence rules:
- Do not fabricate skills, employment history, or any other candidate facts. Do not infer or assert citizenship, visa/work authorization, relocation willingness, languages, salary expectations, certifications, academic coursework/modules, education details, achievements, employment responsibilities, stakeholder collaboration, domain experience, or motivations/preferences unless explicitly present in retrieved candidate evidence.
- A preferred location does not establish relocation willingness or work eligibility. A degree does not establish unlisted coursework. A skill does not establish its use in every project.
- Job-description facts describe the JOB, not the CANDIDATE. For example, "Collaborate with product stakeholders" does not imply the candidate "Worked closely with product stakeholders" without explicit candidate evidence.

Resume guidance:
- Emphasize only supported candidate evidence from ledger A. Never recommend adding an unsupported skill, course, certification, or experience to the resume.
- Keep absent required skills as gaps; do not invent related experience or rewrite unsupported claims into plausible-sounding resume bullets.
- Label any unknown useful qualification "Verify with the candidate before using"; it must not appear as a candidate claim before verification.

Cover-letter guidance:
- Use only supported candidate qualifications from ledger A. Do not invent motivations, relocation willingness, work authorization, collaboration, leadership, or domain experience.
- Unknown facts may only be questions to verify with the candidate, not statements in the cover letter. Discuss ledger B and C as gaps/questions, never as candidate claims."""

    # One user message is sufficient for this reusable workflow. Additional
    # conversation turns would imply orchestration that belongs in the Host.
    return [
        types.PromptMessage(
            role="user",
            content=types.TextContent(type="text", text=instructions),
        )
    ]
