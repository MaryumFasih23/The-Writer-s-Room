from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

from pydantic import BaseModel, Field


class MemoryRecord(BaseModel):
    record_id: str
    category: str
    text: str
    metadata: dict = Field(default_factory=dict)


class SimpleVectorMemory:
    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.records: list[MemoryRecord] = []
        self._load()

    def _load(self) -> None:
        if self.storage_path.exists():
            raw = json.loads(self.storage_path.read_text(encoding="utf-8"))
            self.records = [MemoryRecord(**item) for item in raw]
            self._dedupe_in_place()
            self._save()

    def _save(self) -> None:
        self.storage_path.write_text(
            json.dumps([item.model_dump() for item in self.records], indent=2),
            encoding="utf-8",
        )

    def add(self, record: MemoryRecord) -> MemoryRecord:
        existing = self._find_existing(record)
        if existing:
            existing.text = record.text
            existing.metadata = record.metadata
            self._save()
            return existing
        self.records.append(record)
        self._save()
        return record

    def _find_existing(self, record: MemoryRecord) -> MemoryRecord | None:
        key = self._dedupe_key(record)
        if key is None:
            return None
        for existing in self.records:
            if self._dedupe_key(existing) == key:
                return existing
        return None

    def _dedupe_in_place(self) -> None:
        deduped_reversed: list[MemoryRecord] = []
        seen_keys: set[tuple[str, str]] = set()

        for record in reversed(self.records):
            key = self._dedupe_key(record)
            if key is None:
                deduped_reversed.append(record)
                continue
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduped_reversed.append(record)

        self.records = list(reversed(deduped_reversed))

    @staticmethod
    def _dedupe_key(record: MemoryRecord) -> tuple[str, str] | None:
        if record.category == "character":
            name = str(record.metadata.get("name", "")).strip().lower()
            return ("character", name) if name else None
        if record.category == "script":
            title = str(record.metadata.get("title", "")).strip().lower()
            return ("script", title) if title else None
        if record.category in {"image", "image_asset"}:
            character = str(
                record.metadata.get("character")
                or record.metadata.get("character_name", "")
            ).strip().lower()
            return ("image", character) if character else None
        return None

    def search(self, query: str, top_k: int = 3) -> list[MemoryRecord]:
        if not self.records:
            return []

        query_vector = self._to_vector(query)
        scored = [
            (self._cosine_similarity(query_vector, self._to_vector(record.text)), record)
            for record in self.records
        ]
        ranked = sorted(scored, key=lambda item: item[0], reverse=True)[:top_k]
        return [record for score, record in ranked if score > 0]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\b[a-z0-9]+\b", text.lower())

    def _to_vector(self, text: str) -> Counter:
        return Counter(self._tokenize(text))

    @staticmethod
    def _cosine_similarity(left: Counter, right: Counter) -> float:
        if not left or not right:
            return 0.0

        terms = set(left) | set(right)
        numerator = sum(left.get(term, 0) * right.get(term, 0) for term in terms)
        left_norm = math.sqrt(sum(value * value for value in left.values()))
        right_norm = math.sqrt(sum(value * value for value in right.values()))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return numerator / (left_norm * right_norm)
