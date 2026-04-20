from __future__ import annotations

from uuid import uuid4

from pydantic import BaseModel, Field

from writer_room.mcp.base import mcp_tool
from writer_room.memory.vector_store import MemoryRecord, SimpleVectorMemory


class CommitMemoryInput(BaseModel):
    category: str
    text: str
    metadata: dict = Field(default_factory=dict)


class CommitMemoryOutput(BaseModel):
    record_id: str
    category: str


class SearchMemoryInput(BaseModel):
    query: str
    top_k: int = 3


class SearchMemoryOutput(BaseModel):
    hits: list[str]


@mcp_tool(
    name="commit_memory",
    description="Persist a script, character, or image reference in shared vector memory.",
    tags=["memory", "persistence"],
    input_model=CommitMemoryInput,
    output_model=CommitMemoryOutput,
)
def commit_memory_tool(data: CommitMemoryInput, context: dict) -> CommitMemoryOutput:
    memory: SimpleVectorMemory = context["memory"]
    record = memory.add(
        MemoryRecord(
            record_id=str(uuid4()),
            category=data.category,
            text=data.text,
            metadata=data.metadata,
        )
    )
    return CommitMemoryOutput(record_id=record.record_id, category=record.category)


@mcp_tool(
    name="search_memory",
    description="Retrieve semantically related records from shared memory.",
    tags=["memory", "retrieval"],
    input_model=SearchMemoryInput,
    output_model=SearchMemoryOutput,
)
def search_memory_tool(data: SearchMemoryInput, context: dict) -> SearchMemoryOutput:
    memory: SimpleVectorMemory = context["memory"]
    hits = memory.search(data.query, top_k=data.top_k)
    return SearchMemoryOutput(hits=[f"{item.category}: {item.text}" for item in hits])
