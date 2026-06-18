from __future__ import annotations

import math
from datetime import datetime
from typing import ClassVar
from uuid import uuid4

import tiktoken
from pydantic import BaseModel, ConfigDict, computed_field, Field, model_validator

from memory.clock import CLOCK, UTC

_enc = tiktoken.get_encoding("cl100k_base")


class CacheBlock(BaseModel):
    model_config = ConfigDict(frozen=False)

    DECAY_CONSTANT_S: ClassVar[float] = 3600.0  # 1-hour characteristic decay

    id: str = Field(default_factory=lambda: f"cb_{uuid4()}")
    content: str
    original_embedding: list[float] = Field(default_factory=list)
    current_embedding: list[float] = Field(default_factory=list)
    fidelity: float = 1.0
    compression_count: int = 0
    pointer_to_lt_id: str | None = None
    novelty_score: float
    access_count: int = 0
    last_accessed: datetime = Field(default_factory=lambda: CLOCK.now())
    dirty: bool = False  # content gained info not yet reconciled to LT

    @model_validator(mode="after")
    def _default_current_embedding(self) -> "CacheBlock":
        if not self.current_embedding:
            self.current_embedding = list(self.original_embedding)
        return self

    @computed_field
    @property
    def token_cost(self) -> int:
        return len(_enc.encode(self.content))

    @computed_field
    @property
    def decay_score(self) -> float:
        delta = (CLOCK.now() - self.last_accessed).total_seconds()
        return math.exp(-delta / self.DECAY_CONSTANT_S)

    def record_access(self) -> None:
        self.access_count += 1
        self.last_accessed = CLOCK.now()


class LongTermBlock(BaseModel):
    model_config = ConfigDict(frozen=False)

    DECAY_CONSTANT_S: ClassVar[float] = 86400.0 * 7  # 1-week characteristic decay

    id: str = Field(default_factory=lambda: f"lt_{uuid4()}")
    content: str
    original_embedding: list[float] = Field(default_factory=list)
    fidelity: float = 1.0
    compression_count: int = 0
    novelty_score: float
    access_count: int = 0
    last_accessed: datetime = Field(default_factory=lambda: CLOCK.now())
    is_reconstructed: bool = False

    @computed_field
    @property
    def decay_score(self) -> float:
        delta = (CLOCK.now() - self.last_accessed).total_seconds()
        return math.exp(-delta / self.DECAY_CONSTANT_S)

    def record_access(self) -> None:
        self.access_count += 1
        self.last_accessed = CLOCK.now()
