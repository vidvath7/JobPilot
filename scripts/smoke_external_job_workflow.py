"""Manual A5 demonstration with isolated application storage and explicit consent."""

import asyncio
import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

from host.external_application_workflow import ExternalApplicationWorkflow
from host.external_job_normalizer import ExternalJobNormalizer
from host.himalayas import create_himalayas_client
from host.mcp_client import JobPilotMCPClient
from host.mcp_server_manager import MCPServerManager
from host.nvidia_llm import NVIDIALLMClient, NVIDIAConfigurationError, safe_provider_error
from scripts.smoke_external_job_normalization import diagnostic
from server.services.application_service import ApplicationService


async def confirm_save(name: str, arguments: dict[str, object]) -> bool:
    """Ask fresh consent for exactly the pending JobPilot application record."""
    print("\nHost: approval requested for JobPilot application tracking")
    print("Tool/action: " + name)
    print("Arguments: " + json.dumps(arguments, indent=2, ensure_ascii=True))
    try:
        return input("Save this external application? [y/N]: ").strip().casefold() in {"y", "yes"}
    except (EOFError, KeyboardInterrupt):
        return False


async def run() -> None:
    """Search, prepare, and optionally save only to a temporary JSON store."""
    llm = NVIDIALLMClient(diagnostic_handler=diagnostic)
    with tempfile.TemporaryDirectory(prefix="jobpilot-a5-") as temporary:
        applications_path = Path(temporary) / "applications.json"
        applications_path.write_text("[]\n", encoding="utf-8")
        manager = MCPServerManager()
        manager.register(JobPilotMCPClient(server_environment={
            "JOBPILOT_APPLICATIONS_PATH": str(applications_path),
        }))
        manager.register(create_himalayas_client())
        async with manager:
            await manager.discover_capabilities()
            workflow = ExternalApplicationWorkflow(
                manager, ExternalJobNormalizer(llm), llm,
                ApplicationService(applications_path=applications_path),
            )
            print("External MCP: himalayas::search_jobs", flush=True)
            prepared = await workflow.prepare_first("machine learning", country="Germany")
            job, match = prepared.job, prepared.match
            print("External MCP: himalayas::get_job_details")
            print("Host/domain: normalize")
            print(f"Job: {job.title} at {job.company}")
            print(f"External ID: {job.external_id}")
            print(f"Source: {job.source} | URL: {job.url}")
            print("JobPilot: deterministic score")
            print(f"Match score: {match['score']}")
            print("Component scores: " + json.dumps(match["components"]))
            print("Key evidence: " + json.dumps(match["evidence"], ensure_ascii=True))
            print("JobPilot: application preparation")
            print("Preparation summary: " + prepared.guidance[:1800])
            record = await workflow.save(prepared, confirm_save)
            if record is None:
                print("Host: declined; no application saved.")
            else:
                print("JobPilot: save application " + json.dumps(record, ensure_ascii=True))
            print("Temporary application store: " + applications_path.read_text(encoding="utf-8"))
        print("Temporary application store will be removed on exit.")


if __name__ == "__main__":
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    try:
        asyncio.run(run())
    except Exception as error:
        print("Workflow smoke failed: " + json.dumps(safe_provider_error(error)), file=sys.stderr)
        if isinstance(error, NVIDIAConfigurationError):
            print("NVIDIA_API_KEY is required for this live smoke.", file=sys.stderr)
        raise SystemExit(1) from None
