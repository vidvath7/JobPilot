"""MCP retrieval boundary for A4; extraction itself knows nothing about transports."""

from mcp import types

from host.external_job import NormalizedExternalJob, ExternalJobValidationError, external_job_id
from host.external_job_normalizer import ExternalJobNormalizer
from host.mcp_server_manager import MCPServerManager


class ExternalJobWorkflow:
    """Retrieve one explicitly selected external job, normalize it, and return it only."""

    def __init__(self, manager: MCPServerManager, normalizer: ExternalJobNormalizer):
        self._manager, self._normalizer = manager, normalizer

    async def get_job(self, company_slug: str, job_slug: str) -> NormalizedExternalJob:
        """Route only to Himalayas details; no local scoring, Prompt, or save operation."""
        external_job_id(company_slug, job_slug)
        result = await self._manager.call_tool("himalayas::get_job_details", {
            "company_slug": company_slug, "job_slug": job_slug,
        })
        if not isinstance(result, types.CallToolResult) or result.is_error:
            raise ExternalJobValidationError("Himalayas details did not return a successful Tool result.")
        if not result.content or any(not isinstance(block, types.TextContent) for block in result.content):
            raise ExternalJobValidationError("Himalayas details must contain authoritative text content.")
        # Preserve complete source text. An unsupported future result shape fails
        # explicitly rather than silently dropping binary/context evidence.
        text = "\n".join(block.text for block in result.content)
        return await self._normalizer.normalize(text, company_slug=company_slug, job_slug=job_slug)
