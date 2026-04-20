from __future__ import annotations

import importlib
import inspect
import pkgutil
from typing import Any

from pydantic import BaseModel

import writer_room.mcp.tools as tool_package
from writer_room.config import AppConfig
from writer_room.memory.vector_store import SimpleVectorMemory
from writer_room.mcp.base import MCPToolSpec


class MCPRegistry:
    def __init__(self, config: AppConfig):
        self.tools: dict[str, MCPToolSpec] = {}
        self.context: dict[str, Any] = {
            "memory": SimpleVectorMemory(config.output_dir / "memory_store.json"),
            "output_dir": config.output_dir,
            "images_dir": config.images_dir,
            "config": config,
        }
        self.discover_tools()

    def discover_tools(self) -> None:
        for module_info in pkgutil.iter_modules(tool_package.__path__, tool_package.__name__ + "."):
            module = importlib.import_module(module_info.name)
            for _, obj in inspect.getmembers(module):
                spec = getattr(obj, "_mcp_spec", None)
                if spec:
                    self.tools[spec.name] = spec

    def list_tools(self, required_tags: list[str] | None = None) -> list[MCPToolSpec]:
        results = list(self.tools.values())
        if required_tags:
            required = set(required_tags)
            results = [spec for spec in results if required.issubset(set(spec.tags))]
        return sorted(results, key=lambda spec: spec.name)

    def invoke(self, tool_name: str, payload: dict[str, Any] | BaseModel) -> Any:
        spec = self.tools[tool_name]
        data = payload if isinstance(payload, spec.input_model) else spec.input_model(**payload)
        result = spec.handler(data, self.context)
        if spec.output_model and not isinstance(result, spec.output_model):
            return spec.output_model(**result)
        return result
