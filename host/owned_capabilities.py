"""Host ownership metadata, separate from the existing single-server LLM catalog."""

from dataclasses import dataclass

from host.capabilities import (
    ToolCapability, ResourceCapability, ResourceTemplateCapability, PromptCapability,
)


def validate_server_id(server_id: str) -> None:
    """Reserve the qualification delimiter without silently rewriting identities."""
    if not server_id or server_id != server_id.strip() or "::" in server_id:
        raise ValueError("Server ID must be nonempty, trimmed, and contain no '::'.")


@dataclass(frozen=True)
class QualifiedCapability:
    """An application identifier; only its original name travels over MCP."""

    server_id: str
    name: str

    def __post_init__(self) -> None:
        validate_server_id(self.server_id)
        if not self.name or "::" in self.name:
            raise ValueError("Capability name must be nonempty and contain no '::'.")

    def __str__(self) -> str:
        return f"{self.server_id}::{self.name}"

    @classmethod
    def parse(cls, value: str) -> "QualifiedCapability":
        """Require ownership even when an unqualified name happens to be unique."""
        parts = value.split("::")
        if len(parts) != 2:
            raise ValueError("Use a qualified capability: server_id::name.")
        return cls(*parts)


@dataclass(frozen=True)
class OwnedToolCapability:
    """Keep colliding Tool names distinct without changing their schemas."""

    server_id: str
    capability: ToolCapability

    @property
    def qualified_name(self) -> str:
        return str(QualifiedCapability(self.server_id, self.capability.name))


@dataclass(frozen=True)
class OwnedResourceCapability:
    """URI identity does not by itself establish server ownership."""

    server_id: str
    capability: ResourceCapability


@dataclass(frozen=True)
class OwnedResourceTemplateCapability:
    """Retain the server owning a parameterized Resource URI."""

    server_id: str
    capability: ResourceTemplateCapability


@dataclass(frozen=True)
class OwnedPromptCapability:
    """Prompt names are scoped to their originating server just like Tool names."""

    server_id: str
    capability: PromptCapability

    @property
    def qualified_name(self) -> str:
        return str(QualifiedCapability(self.server_id, self.capability.name))


@dataclass(frozen=True)
class MultiServerCapabilityCatalog:
    """Ownership-preserving snapshot; never substituted into existing orchestration."""

    tools: tuple[OwnedToolCapability, ...]
    resources: tuple[OwnedResourceCapability, ...]
    resource_templates: tuple[OwnedResourceTemplateCapability, ...]
    prompts: tuple[OwnedPromptCapability, ...]

    def find_tools_by_name(self, name: str) -> tuple[OwnedToolCapability, ...]:
        """Return all owners rather than resolving a collision arbitrarily."""
        return tuple(tool for tool in self.tools if tool.capability.name == name)

    def get_tool(self, server_id: str, name: str) -> OwnedToolCapability:
        """Look up one explicitly owned Tool, failing on unknown identities."""
        for tool in self.find_tools_by_name(name):
            if tool.server_id == server_id:
                return tool
        raise KeyError(str(QualifiedCapability(server_id, name)))
