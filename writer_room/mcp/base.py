from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field


class MCPToolSpec(BaseModel):
    name: str
    description: str
    tags: list[str] = Field(default_factory=list)
    input_model: type[BaseModel]
    output_model: type[BaseModel] | None = None
    handler: Callable[[BaseModel, dict[str, Any]], Any]


def mcp_tool(
    *,
    name: str,
    description: str,
    tags: list[str],
    input_model: type[BaseModel],
    output_model: type[BaseModel] | None = None,
):
    def decorator(func: Callable[[BaseModel, dict[str, Any]], Any]) -> Callable[[BaseModel, dict[str, Any]], Any]:
        setattr(
            func,
            "_mcp_spec",
            MCPToolSpec(
                name=name,
                description=description,
                tags=tags,
                input_model=input_model,
                output_model=output_model,
                handler=func,
            ),
        )
        return func

    return decorator
