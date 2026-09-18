"""Manual NVIDIA approval experiment with an isolated, temporary application store."""

import asyncio
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from host.main import confirm_action
from host.mcp_client import JobPilotMCPClient
from host.nvidia_llm import NVIDIALLMClient
from host.orchestrator import AUTO_EXECUTABLE_TOOLS, CONFIRMATION_REQUIRED_TOOLS, JobPilotOrchestrator


async def run() -> None:
    """Ask the human to decline, then approve; verify each persisted outcome."""
    llm = NVIDIALLMClient()
    production_path = PROJECT_ROOT / "data" / "applications.json"
    before = production_path.read_bytes()
    with TemporaryDirectory(prefix="jobpilot-approval-") as directory:
        store = Path(directory) / "applications.json"
        store.write_text("[]\n", encoding="utf-8")
        try:
            async with JobPilotMCPClient(server_environment={
                "JOBPILOT_APPLICATIONS_PATH": str(store),
            }) as client:
                catalog = await client.discover_capabilities()
                # Verify through the actual server that the injected store is used
                # by the application-history Resource before allowing a write.
                history = await client.read_resource("applications://all")
                if any(json.loads(item.text) != [] for item in history.contents):
                    raise RuntimeError("Application history is not empty; aborting smoke test.")
                for approve_expected in (False, True):
                    print("\n=== " + ("Approve with y/yes" if approve_expected else "Decline with Enter/n") + " ===")
                    decisions = []

                    async def approve(name, arguments):
                        """Every requested action still requires real interactive input."""
                        answer = await confirm_action(name, arguments)
                        decisions.append(answer)
                        return answer

                    engine = JobPilotOrchestrator(
                        llm, client, catalog, allowed_tools=AUTO_EXECUTABLE_TOOLS,
                        confirmation_required_tools=CONFIRMATION_REQUIRED_TOOLS,
                        approval_handler=approve,
                    )
                    result = await engine.ask("Save JOB-005 as applied.")
                    print(f"Answer: {result.answer}")
                    records = json.loads(store.read_text(encoding="utf-8"))
                    if decisions != [approve_expected]:
                        raise RuntimeError("Expected exactly one matching manual decision; experiment incomplete.")
                    if approve_expected:
                        if len(records) != 1 or records[0]["application_id"] != "APP-001" or records[0]["job_id"] != "JOB-005" or records[0]["status"] != "applied":
                            raise RuntimeError("Expected APP-001 for JOB-005 with status applied.")
                    elif records != []:
                        raise RuntimeError("Declined action modified temporary storage.")
                    print("Temporary persistence verified: " + json.dumps(records))
        finally:
            if production_path.read_bytes() != before:
                raise RuntimeError("Production applications changed during the experiment.")
            print("Production applications unchanged.")


if __name__ == "__main__":
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    try:
        asyncio.run(run())
    except Exception as error:
        print(f"Approval smoke failed ({type(error).__name__}).", file=sys.stderr)
        raise SystemExit(1) from None
