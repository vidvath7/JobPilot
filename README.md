# JobPilot

JobPilot is a learning-focused project that explores the Model Context Protocol (MCP) through an incremental job-application workflow.

## Problem statement

Job discovery, shortlisting, resume tailoring, cover-letter preparation, and application tracking are often fragmented across job platforms, documents, and spreadsheets. Repeating these manual activities makes the overall application process slow and difficult to manage consistently.

## Primary objective

The primary goal is to build a deep, practical understanding of MCP through an inspectable end-to-end project. Success means understanding and explaining how each MCP component works and why it is used—not maximizing feature breadth or automating the entire application process at once.

## Why MCP

JobPilot uses MCP to expose structured business capabilities and readable context to an AI Host. Tools represent operations, Resources expose context, and Prompts provide reusable model-facing instructions.

## Architecture

```text
User
  ↓
Host + LLM
  ↓
MCP Client
  ↓
JobPilot MCP Server
  ↓
Tools / Resources / Prompts
  ↓
Application Services
  ↓
Local Data
```

The Host uses NVIDIA NIM through a provider-independent LLM boundary and connects to one local MCP server over stdio. The Host owns bounded orchestration and approval; the server owns business capabilities. `read_mcp_resource` is a Host-local bridge, not a server Tool. Prompt retrieval returns instructions; the Host executes explicitly selected workflows.

## V1 milestones

1. **Milestone 0 — Foundation:** Establish the Python project, dependencies, repository structure, synthetic data, baseline testing, and documentation.
2. **Milestone 1 — Minimal MCP Server and `search_jobs` Tool:** Build the first independently testable MCP vertical slice.
3. **Milestone 2 — MCP Resources:** Introduce candidate-profile and job-detail Resources.
4. **Milestone 3 — Deterministic job matching:** Add reproducible candidate-to-job scoring with supporting evidence.
5. **Milestone 4 — Application tracking:** Introduce validated persistence and a state-changing Tool.
6. **Milestone 5 — MCP Prompt:** Add a reusable application-preparation Prompt.
7. **Milestone 6 — Host, MCP Client, and LLM:** Add capability discovery and model-driven orchestration.
8. **Milestone 7 — End-to-end V1:** Connect job discovery, profile context, matching, recommendations, and application tracking.
9. **Milestone 8 — Testing and error handling:** Strengthen automated coverage and failure behavior across boundaries.
10. **Milestone 9 — Security review:** Review privacy, trust boundaries, side effects, prompt injection, secrets, and logging.

## Later scope

Resume tailoring and document generation, cover-letter creation, real job-source integrations, remote deployment, and a user interface are possible later enhancements. They are not V1 requirements.

## Current status

**Milestone 6 — Host, MCP Client, and LLM** is implemented and acceptance-tested against synthetic local data. The server exposes `search_jobs`, `score_job_match`, and `save_application`; Resources `candidate://profile` and `applications://all`; template `jobs://job/{job_id}`; and Prompt `prepare_application(job_id)`.

## Run locally

With Python 3.11+ and uv installed:

```powershell
uv sync
uv run python -m host.main
```

Before starting the Host, supply `NVIDIA_API_KEY` in the shell or a repository-root `.env` file (ignored by Git). Executable entry points load `.env` without overriding shell values. Host startup requires the key; automated tests do not. The default model is `nvidia/nemotron-3.5-lightning-30b-a3b` at `https://integrate.api.nvidia.com/v1`.

Example CLI commands:

```text
capabilities
prompts
ask Find AI Engineer jobs in Germany
read candidate://profile
prompt prepare_application {"job_id":"JOB-005"}
run-prompt prepare_application {"job_id":"JOB-005"}
quit
```

`prompt` retrieves and displays instructions; `run-prompt` retrieves them and runs the bounded LLM workflow. `help` lists all commands, including generic `call` and `read` operations.

Automatic policy permits `search_jobs`, `score_job_match`, and the read-only Resource bridge. LLM-requested `save_application` requires fresh approval of the displayed arguments: only `y`/`yes` approves. Declining performs no write. Manual `call save_application {...}` is direct user-requested execution and bypasses that LLM confirmation flow.

To isolate manual persistence in PowerShell before starting the Host:

```powershell
$applicationStore = Join-Path $env:TEMP "jobpilot-manual-applications.json"
if (-not (Test-Path -LiteralPath $applicationStore)) {
    Set-Content -LiteralPath $applicationStore -Value "[]"
}
$env:JOBPILOT_APPLICATIONS_PATH = $applicationStore
uv run python -m host.main
```

The child server inherits the Host environment; explicit client overrides take precedence. ApplicationService uses an explicitly injected path first, then `JOBPILOT_APPLICATIONS_PATH`, then `data/applications.json`.

## Verification

```text
uv run pytest
git diff --check
```

Automated persistence tests use temporary stores and require no live NVIDIA call. Manual smoke scripts require credentials and network access:

- `uv run python scripts/smoke_nvidia.py`: provider connectivity, dynamic Tool schemas, and a Prompt-text boundary check; never executes model-selected calls.
- `uv run python scripts/smoke_orchestration.py`: read-only Tool/Resource and user-selected Prompt orchestration; excludes `save_application`.
- `uv run python scripts/smoke_approval.py`: interactive decline/approve experiment using temporary application storage.

Orchestration defaults to three Tool rounds, retains history, and rejects exact repeated calls. Resource bridging supports text contents and simple URI placeholders. Search remains literal/deterministic; live model choices can vary. The Prompt's SUPPORTED / GAP / UNKNOWN-VERIFY policy guides grounding, but model output still requires human review. Resume and cover-letter files are not generated.

## Scope-control principle

Every important architectural component should have a clear MCP learning justification. Complexity that does not support the current learning objective should be postponed.
