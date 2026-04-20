from __future__ import annotations

from writer_room.mcp.registry import MCPRegistry


class BaseAgent:
    def __init__(self, registry: MCPRegistry):
        self.registry = registry

    def resolve_tool_name(self, required_tags: list[str]) -> str:
        matches = self.registry.list_tools(required_tags=required_tags)
        if not matches:
            raise RuntimeError(f"No MCP tool found for tags: {required_tags}")
        return matches[0].name
