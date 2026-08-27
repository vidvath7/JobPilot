# JobPilot

JobPilot is a learning-focused project that explores the Model Context Protocol (MCP) through an incremental job-application workflow.

## Problem statement

Job discovery, shortlisting, resume tailoring, cover-letter preparation, and application tracking are often fragmented across job platforms, documents, and spreadsheets. Repeating these manual activities makes the overall application process slow and difficult to manage consistently.

## Primary objective

The primary goal is to build a deep, practical understanding of MCP through an inspectable end-to-end project. Success means understanding and explaining how each MCP component works and why it is used—not maximizing feature breadth or automating the entire application process at once.

## Why MCP

JobPilot will use MCP to expose structured business capabilities and readable context to an AI host. MCP Tools will represent meaningful operations, Resources will expose identifiable context, and Prompts will demonstrate reusable model-facing workflows.

## Planned architecture

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

The Host, MCP Client, and LLM-based orchestration are planned for a later V1 milestone and are not currently implemented.

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

JobPilot is currently in **Milestone 0 — Foundation**. Project configuration, package scaffolding, controlled synthetic data, and a baseline smoke test exist. No MCP capability has been implemented yet.

## Scope-control principle

Every important architectural component should have a clear MCP learning justification. Complexity that does not support the current learning objective should be postponed.
