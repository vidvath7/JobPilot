"""Manual live observation of bounded, read-only LLM/MCP Tool rounds."""

import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

# Support direct script execution using the project interpreter.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from host.mcp_client import JobPilotMCPClient
from host.nvidia_llm import NVIDIALLMClient
from host.orchestrator import AUTO_EXECUTABLE_TOOLS, JobPilotOrchestrator
from host.prompt_workflow import PromptWorkflow


async def run() -> None:
    """Reuse one MCP connection; each question gets independent model history."""
    async with JobPilotMCPClient() as client:
        catalog = await client.discover_capabilities()
        orchestrator = JobPilotOrchestrator(
            NVIDIALLMClient(), client, catalog,
            allowed_tools=AUTO_EXECUTABLE_TOOLS,
        )
        for query in (
            "Find AI Engineer jobs in Germany.",
            "Find GenAI or LLM-related jobs in Germany.",
            "Show me the details for JOB-005.",
            "How well do I match JOB-005? Use the job details when explaining.",
        ):
            print(f"\n=== {query} ===", flush=True)
            result = await orchestrator.ask(query)
            for call in result.executed_tool_calls:
                print(f"Tool: {call.name} {json.dumps(call.arguments)}")
                print(f"is_error: {call.is_error}")
                print(f"result: {json.dumps(call.result)}")
            print(f"Answer: {result.answer}", flush=True)

        print("\n=== prepare_application Workflow ===", flush=True)
        workflow = PromptWorkflow(client, catalog, orchestrator)
        result = await workflow.run_prompt("prepare_application", {"job_id": "JOB-005"})
        for call in result.executed_tool_calls:
            print(f"Tool: {call.name} {json.dumps(call.arguments)}")
            print(f"is_error: {call.is_error}")
            print(f"result: {json.dumps(call.result)}")
        print(f"Answer: {result.answer}", flush=True)


if __name__ == "__main__":
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    try:
        asyncio.run(run())
    except Exception as error:
        # Print the safe exception type/message, without raw provider diagnostics or tracebacks.
        print(f"Orchestration smoke failed ({type(error).__name__}): {error}", file=sys.stderr)
        raise SystemExit(1) from None
