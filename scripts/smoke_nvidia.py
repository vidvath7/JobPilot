"""Manually exercise the NVIDIA adapter at MCP/Host/LLM boundaries.

This script is intentionally outside the automated test suite because it uses a
real hosted model and therefore requires a developer-provided credential and
network access. Model-selected Tools are printed for inspection but never
executed; automatic orchestration belongs to a later JobPilot milestone.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

from mcp import types
from dotenv import load_dotenv


# Running a file directly places ``scripts/`` on Python's import path. Add the
# repository root so the documented command works without installing JobPilot as
# a package or relying on the caller's current working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from host.llm import LLMResponse  # noqa: E402
from host.mcp_client import JobPilotMCPClient  # noqa: E402
from host.nvidia_llm import (  # noqa: E402
    NVIDIA_API_KEY_ENVIRONMENT_VARIABLE,
    NVIDIALLMClient,
)
from host.tool_conversion import catalog_tools_to_llm_tools  # noqa: E402


PLAIN_REQUEST = (
    "Reply briefly confirming that the JobPilot NVIDIA connection works."
)
EXACT_ROLE_REQUEST = "Find AI Engineer jobs in Germany."
GENAI_REQUEST = "Find GenAI or LLM-related jobs in Germany."


def _load_local_environment(repository_root: Path = PROJECT_ROOT) -> None:
    """Load the repository's local fallback without overriding shell values."""
    # Keep file discovery at this executable boundary. NVIDIALLMClient receives
    # configuration exclusively through the process environment.
    load_dotenv(repository_root / ".env", override=False)


def _print_response(response: LLMResponse) -> None:
    """Render only provider-neutral output, never raw SDK objects or secrets."""
    print(f"assistant content: {response.content!r}")
    if not response.tool_calls:
        print("tool_calls: []")
        return

    print("tool_calls:")
    for tool_call in response.tool_calls:
        print(f"  id: {tool_call.id}")
        print(f"  name: {tool_call.name}")
        print(
            "  arguments: "
            + json.dumps(
                tool_call.arguments,
                ensure_ascii=False,
                sort_keys=True,
            )
        )


def _extract_prompt_text(
    result: types.GetPromptResult | types.InputRequiredResult,
) -> str:
    """Extract text through the MCP v2 Prompt result model used by the Host."""
    if not isinstance(result, types.GetPromptResult):
        raise RuntimeError("MCP Prompt retrieval did not return a Prompt result.")

    text_parts = [
        message.content.text
        for message in result.messages
        if isinstance(message.content, types.TextContent)
    ]
    if not text_parts:
        raise RuntimeError("The retrieved MCP Prompt did not contain text content.")
    return "\n\n".join(text_parts)


async def _run_smoke_test() -> None:
    """Run independent live requests without interpreting or executing Tool calls."""
    llm_client = NVIDIALLMClient()

    print("=== Plain NVIDIA Test ===")
    plain_response = await llm_client.complete(
        [{"role": "user", "content": PLAIN_REQUEST}]
    )
    _print_response(plain_response)

    async with JobPilotMCPClient() as mcp_client:
        # Tool definitions come from real MCP discovery. The smoke script knows
        # neither their schemas nor which Tool the model should select.
        catalog = await mcp_client.discover_capabilities()
        llm_tools = catalog_tools_to_llm_tools(catalog)

        print("\n=== Exact Role Tool Selection ===")
        exact_role_response = await llm_client.complete(
            [{"role": "user", "content": EXACT_ROLE_REQUEST}],
            tools=llm_tools,
        )
        _print_response(exact_role_response)

        print("\n=== GenAI/LLM Tool Selection ===")
        genai_response = await llm_client.complete(
            [{"role": "user", "content": GENAI_REQUEST}],
            tools=llm_tools,
        )
        _print_response(genai_response)

        print("\n=== MCP Prompt to LLM Test ===")
        prompt_result = await mcp_client.get_prompt(
            "prepare_application",
            {"job_id": "JOB-005"},
        )
        prompt_text = _extract_prompt_text(prompt_result)
        prompt_response = await llm_client.complete(
            [{"role": "user", "content": prompt_text}]
        )
        _print_response(prompt_response)

        # Deliberately do not feed any LLMToolCall to mcp_client.call_tool().
        # This milestone observes model selection only; it performs no workflow.


def main() -> None:
    """Run the async smoke workflow while keeping failures credential-safe."""
    _load_local_environment()
    if not os.environ.get(NVIDIA_API_KEY_ENVIRONMENT_VARIABLE):
        print(
            "NVIDIA_API_KEY is required for this manual live smoke script.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    try:
        asyncio.run(_run_smoke_test())
    except Exception as error:
        # Provider exception messages can contain request diagnostics. Reporting
        # only the type keeps this manual utility from accidentally logging a
        # credential while still returning a failing process status.
        print(
            f"NVIDIA smoke test failed ({type(error).__name__}).",
            file=sys.stderr,
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
