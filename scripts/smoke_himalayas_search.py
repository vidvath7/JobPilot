"""Manual NVIDIA -> allow-listed Himalayas discovery; never persist/import jobs."""

import asyncio
import json
import re
import sys
from pathlib import Path
from urllib.parse import urlsplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from host.himalayas import create_himalayas_client
from host.mcp_client import JobPilotMCPClient
from host.mcp_server_manager import MCPServerManager
from host.multi_server_orchestrator import MultiServerOrchestrator
from host.nvidia_llm import NVIDIALLMClient, NVIDIAConfigurationError
from host.orchestrator import OrchestrationError
from host.owned_capabilities import QualifiedCapability


def report_error(error, output=print):
    """Show safe Host diagnostics, never provider exceptions or chained payloads."""
    message = str(error) if isinstance(error, (OrchestrationError, NVIDIAConfigurationError)) else "See configuration/connectivity; provider payload omitted."
    output(f"Experiment failed ({type(error).__name__}): {message}")


def slug_pair(value):
    """Find exact returned slug fields only; never invent an external job identity."""
    if isinstance(value, str):
        try:
            return slug_pair(json.loads(value))
        except (ValueError, TypeError):
            links = returned_job_links(value)
            return (links[0][1], links[0][2]) if links else None
    if isinstance(value, dict):
        company = value.get("company_slug") or value.get("companySlug")
        job = value.get("job_slug") or value.get("jobSlug")
        if isinstance(company, str) and isinstance(job, str):
            return company, job
        for child in value.values():
            if pair := slug_pair(child):
                return pair
    if isinstance(value, list):
        for child in value:
            if pair := slug_pair(child):
                return pair
    return None


def returned_job_links(text):
    """Read exact slug path segments from observed MCP text URLs, never from titles.

    The live server currently returns Markdown rather than structured_content.
    This smoke-only extraction supports its documented details arguments without
    normalizing/importing external jobs or modifying the LLM's original evidence.
    """
    links = []
    for url in re.findall(r"https://himalayas\.app/companies/[^\s<>\)]+", text):
        parsed = urlsplit(url)
        match = re.fullmatch(r"/companies/([^/]+)/jobs/([^/]+)", parsed.path)
        if parsed.hostname == "himalayas.app" and match:
            entry = (url, match[1], match[2])
            if entry not in links:
                links.append(entry)
    return links


def job_records(value):
    """Inspect JSON/text results for display only; preserve original fields in history."""
    if isinstance(value, str):
        try:
            return job_records(json.loads(value))
        except (ValueError, TypeError):
            return []
    if isinstance(value, dict):
        if "title" in value:
            return [value]
        return [job for child in value.values() for job in job_records(child)]
    if isinstance(value, list):
        return [job for child in value for job in job_records(child)]
    return []


def print_execution(call, registry, output=print):
    """Print every completed operation, including errors, without dumping SDK objects."""
    qualified = registry.resolve(call.name) if call.name != "read_mcp_resource" else None
    original = QualifiedCapability.parse(qualified).name if qualified else "resources/read"
    output(f"LLM Tool: {call.name}\nQualified: {qualified or 'Host Resource bridge'}\nMCP name: {original}")
    output(f"Arguments: {json.dumps(call.arguments)}\nis_error: {call.is_error}")
    jobs = job_records(call.result)
    output(f"Result summary: returned job objects={len(jobs)}")
    for job in jobs[:3]:
        fields = {key: job[key] for key in (
            "title", "companyName", "company_slug", "job_slug", "companySlug", "jobSlug", "applicationLink"
        ) if key in job}
        output(json.dumps(fields, ensure_ascii=True))
    if not jobs and isinstance(call.result, str) and original == "search_jobs":
        links = returned_job_links(call.result)
        output(f"Text result: {call.result.splitlines()[0] if call.result else '(empty)'}; returned job links={len(links)}")
        for url, company, job in links[:3]:
            output(json.dumps({"company_slug": company, "job_slug": job, "applicationLink": url}))
    elif not jobs or original == "get_job_details":
        output("Actual result: " + json.dumps(call.result, ensure_ascii=True))


async def run():
    """Observe live model choices and a details request grounded in a returned job."""
    # Check credentials before opening either transport, without printing secrets.
    llm = NVIDIALLMClient()
    manager = MCPServerManager()
    manager.register(JobPilotMCPClient())
    manager.register(create_himalayas_client())
    async with manager:
        catalog = await manager.discover_capabilities()
        # No approval handler: a model-requested local save fails closed.
        observed = []

        def observe_execution(call):
            observed.append(call)
            print_execution(call, orchestrator.registry)

        orchestrator = MultiServerOrchestrator(llm, manager, catalog, execution_observer=observe_execution)
        pair = None

        async def observe(query):
            print(f"\n=== {query} ===", flush=True)
            result = await orchestrator.ask(query)
            print(f"Answer: {result.answer}", flush=True)
            return result

        for query in (
            "Find mid-level machine learning or AI jobs in Germany.",
            "Find senior LLM jobs that are available worldwide.",
        ):
            try:
                result = await observe(query)
            except Exception as error:
                report_error(error)
        for call in observed:
            if call.name == "himalayas__search_jobs" and not call.is_error:
                pair = pair or slug_pair(call.result)
        if pair:
            # Explicit diagnostic read guarantees exact returned slugs are used,
            # independent of whether the model chooses a follow-up details call.
            print("\n=== Exact returned-slug details experiment ===")
            arguments = {"company_slug": pair[0], "job_slug": pair[1]}
            print("Qualified: himalayas::get_job_details\nArguments: " + json.dumps(arguments))
            details = await manager.call_tool("himalayas::get_job_details", arguments)
            print(f"is_error: {details.is_error}")
            print("Actual detail result: " + json.dumps(
                details.structured_content if details.structured_content is not None
                else [block.text for block in details.content if block.type == "text"], ensure_ascii=True))
        else:
            print("Details experiment skipped: search returned no identifiable slug pair.")


if __name__ == "__main__":
    # Windows legacy consoles must not interrupt successful live diagnostics
    # when returned job descriptions or model answers contain Unicode symbols.
    sys.stdout.reconfigure(errors="backslashreplace")
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    try:
        asyncio.run(run())
    except Exception as error:
        report_error(error)
        raise SystemExit(1) from None
