"""User-selected Prompt retrieval and orchestration owned entirely by the Host."""

from mcp import types

from host.capabilities import CapabilityCatalog
from host.mcp_client import JobPilotMCPClient
from host.orchestrator import JobPilotOrchestrator, OrchestrationResult, OrchestrationError
from host.prompt_conversion import prompt_messages_to_llm, PromptConversionError


class PromptWorkflowError(OrchestrationError):
    """Prompt selection, argument validation, or retrieval could not complete."""


class PromptWorkflow:
    """Retrieve only explicitly selected discovered Prompts; reuse one execution engine."""

    def __init__(self, client: JobPilotMCPClient, catalog: CapabilityCatalog,
                 orchestrator: JobPilotOrchestrator) -> None:
        self._client = client
        self._catalog = catalog
        self._orchestrator = orchestrator

    async def run_prompt(self, prompt_name: str,
                         arguments: dict[str, str] | None = None) -> OrchestrationResult:
        """Validate discovery metadata before MCP retrieval and preserve returned messages."""
        prompt = next((item for item in self._catalog.prompts if item.name == prompt_name), None)
        if prompt is None:
            raise PromptWorkflowError(f"Unknown discovered Prompt: {prompt_name}")
        supplied = {} if arguments is None else arguments
        if not isinstance(supplied, dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in supplied.items()
        ):
            raise PromptWorkflowError("Prompt arguments must be an object with string values.")
        expected = {argument.name for argument in prompt.arguments}
        if set(supplied) - expected:
            raise PromptWorkflowError("Unexpected Prompt arguments: " + ", ".join(sorted(set(supplied) - expected)))
        missing = {argument.name for argument in prompt.arguments if argument.required is True} - set(supplied)
        if missing:
            raise PromptWorkflowError("Missing required Prompt arguments: " + ", ".join(sorted(missing)))
        try:
            result = await self._client.get_prompt(prompt_name, None if arguments is None else dict(supplied))
        except Exception as error:
            raise PromptWorkflowError(f"MCP Prompt retrieval failed ({type(error).__name__}).") from error
        if not isinstance(result, types.GetPromptResult):
            raise PromptWorkflowError("MCP returned an unsupported Prompt result.")
        try:
            messages = prompt_messages_to_llm(result.messages)
        except PromptConversionError as error:
            raise PromptWorkflowError(str(error)) from error
        return await self._orchestrator.run(messages)
