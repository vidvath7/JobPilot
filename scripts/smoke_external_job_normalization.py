"""Manual A4: current search -> exact returned slugs -> details -> in-memory job."""

import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from mcp import types
from host.external_job import ExternalJobValidationError
from host.external_job_normalizer import ExternalJobNormalizer
from host.external_job_workflow import ExternalJobWorkflow
from host.himalayas import create_himalayas_client
from host.mcp_client import JobPilotMCPClient
from host.mcp_server_manager import MCPServerManager
from host.nvidia_llm import NVIDIALLMClient, NVIDIAConfigurationError, safe_provider_error
from scripts.smoke_himalayas_search import slug_pair


def diagnostic(event):
    """Print only whitelisted adapter metadata, not provider messages or job text."""
    print("Diagnostic: " + json.dumps(event, ensure_ascii=True), flush=True)


class DiagnosticNormalizer(ExternalJobNormalizer):
    """Smoke-only stage markers separate MCP retrieval from extraction/validation."""

    async def normalize(self, raw_source_text, **context):
        diagnostic({"stage": "normalization.started", "source_characters": len(raw_source_text)})
        try:
            job = await super().normalize(raw_source_text, **context)
        except Exception:
            diagnostic({"stage": "normalization.failed"})
            raise
        diagnostic({"stage": "normalization.validated"})
        return job


async def run() -> None:
    """Use live metadata/data, never stale slugs or synthetic JOB-* IDs."""
    diagnostic({"stage": "nvidia.configuration"})
    llm = NVIDIALLMClient(diagnostic_handler=diagnostic)
    manager = MCPServerManager()
    manager.register(JobPilotMCPClient())
    manager.register(create_himalayas_client())
    diagnostic({"stage": "mcp.connect"})
    async with manager:
        diagnostic({"stage": "mcp.discovery"})
        await manager.discover_capabilities()
        diagnostic({"stage": "mcp.himalayas.search_jobs"})
        result = await manager.call_tool("himalayas::search_jobs", {"keyword": "machine learning"})
        if not isinstance(result, types.CallToolResult) or result.is_error:
            raise ExternalJobValidationError("Live search failed; no normalization attempted.")
        value = result.structured_content if result.structured_content is not None else "\n".join(
            block.text for block in result.content if isinstance(block, types.TextContent))
        pair = slug_pair(value)
        if pair is None:
            raise ExternalJobValidationError("Live search supplied no identifiable job slugs.")
        diagnostic({"stage": "mcp.himalayas.get_job_details"})
        job = await ExternalJobWorkflow(manager, DiagnosticNormalizer(llm)).get_job(*pair)
        for label, value in (
            ("External ID", job.external_id), ("Source", job.source), ("Title", job.title),
            ("Company", job.company), ("Location", job.location), ("Experience", job.experience_level),
            ("Required skills", job.required_skills), ("URL", job.url),
        ):
            print(f"{label}: {json.dumps(value, ensure_ascii=True)}")
        print("Description preview: " + json.dumps(job.description[:600], ensure_ascii=True))
        print("Returned in memory only. No matching, Prompt workflow, application save, or persistence.")


if __name__ == "__main__":
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    try:
        asyncio.run(run())
    except Exception as error:
        diagnostic(safe_provider_error(error))
        message = str(error) if isinstance(error, (ExternalJobValidationError, NVIDIAConfigurationError)) else "Provider/protocol payload omitted."
        print(f"Normalization smoke failed ({type(error).__name__}): {message}", file=sys.stderr)
        raise SystemExit(1) from None
