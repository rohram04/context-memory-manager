from __future__ import annotations

import struct
from datetime import datetime, timezone
from typing import Any

import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool

from memory.block import LongTermBlock
from memory.clock import CLOCK

UTC = timezone.utc


def _pack_vec(vec: list[float]) -> bytes | None:
    if not vec:
        return None
    return struct.pack(f"{len(vec)}f", *vec)


def _unpack_vec(data: bytes | None) -> list[float]:
    if not data:
        return []
    n = len(data) // 4
    return list(struct.unpack(f"{n}f", data))


def _cosine_sim(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    return float(np.dot(va, vb) / denom) if denom > 0 else 0.0


def _pg_vec_str(vec: list[float]) -> str | None:
    if not vec:
        return None
    return f"[{','.join(str(x) for x in vec)}]"


def _parse_pg_vec(v: Any) -> list[float]:
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return list(v)
    return [float(x) for x in str(v).strip("[]").split(",")]


class LongTermStore:
    def __init__(self, url: str, embedding_dim: int = 1536) -> None:
        kwargs: dict[str, Any] = {}
        if url.startswith("sqlite"):
            kwargs["connect_args"] = {"check_same_thread": False}
            # An in-memory sqlite DB lives inside a single connection; with the
            # default pool each connection gets its own empty DB. StaticPool keeps
            # one shared connection so the schema/data persist across threadpool
            # workers (the demo serves sync endpoints from a threadpool).
            if ":memory:" in url:
                kwargs["poolclass"] = StaticPool
        self._engine = create_engine(url, **kwargs)
        self._dialect = self._engine.dialect.name
        self._embedding_dim = embedding_dim
        self._create_tables()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _create_tables(self) -> None:
        with self._engine.connect() as conn:
            if self._dialect == "sqlite":
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS lt_blocks (
                        id                 TEXT PRIMARY KEY,
                        content            TEXT,
                        fidelity           REAL    NOT NULL DEFAULT 1.0,
                        compression_count  INTEGER NOT NULL DEFAULT 0,
                        novelty_score      REAL    NOT NULL,
                        access_count       INTEGER NOT NULL DEFAULT 0,
                        last_accessed      TEXT,
                        created_at         TEXT    NOT NULL DEFAULT (datetime('now')),
                        is_reconstructed   INTEGER NOT NULL DEFAULT 0,
                        original_embedding BLOB
                    )
                """))
            else:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                conn.execute(text(f"""
                    CREATE TABLE IF NOT EXISTS lt_blocks (
                        id                 TEXT PRIMARY KEY,
                        content            TEXT,
                        fidelity           FLOAT   NOT NULL DEFAULT 1.0,
                        compression_count  INT     NOT NULL DEFAULT 0,
                        novelty_score      FLOAT   NOT NULL,
                        access_count       INT     NOT NULL DEFAULT 0,
                        last_accessed      TIMESTAMPTZ,
                        created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
                        is_reconstructed   BOOLEAN NOT NULL DEFAULT FALSE,
                        original_embedding vector({self._embedding_dim})
                    )
                """))
                conn.execute(text(
                    "CREATE INDEX IF NOT EXISTS lt_blocks_emb_idx "
                    "ON lt_blocks USING hnsw (original_embedding vector_cosine_ops)"
                ))
            conn.commit()

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def _to_row(self, block: LongTermBlock) -> dict[str, Any]:
        base: dict[str, Any] = {
            "id": block.id,
            "content": block.content,
            "fidelity": block.fidelity,
            "compression_count": block.compression_count,
            "novelty_score": block.novelty_score,
            "access_count": block.access_count,
        }
        if self._dialect == "sqlite":
            return {
                **base,
                "last_accessed": block.last_accessed.isoformat(),
                "is_reconstructed": int(block.is_reconstructed),
                "original_embedding": _pack_vec(block.original_embedding),
            }
        return {
            **base,
            "last_accessed": block.last_accessed,
            "is_reconstructed": block.is_reconstructed,
            "original_embedding": _pg_vec_str(block.original_embedding),
        }

    def _from_row(self, row: dict[str, Any]) -> LongTermBlock:
        last_accessed = row["last_accessed"]
        if isinstance(last_accessed, str):
            last_accessed = datetime.fromisoformat(last_accessed)
        if last_accessed is None:
            last_accessed = CLOCK.now()
        if last_accessed.tzinfo is None:
            last_accessed = last_accessed.replace(tzinfo=UTC)

        if self._dialect == "sqlite":
            original_embedding = _unpack_vec(row.get("original_embedding"))
        else:
            original_embedding = _parse_pg_vec(row.get("original_embedding"))
        is_reconstructed = bool(row["is_reconstructed"])

        return LongTermBlock(
            id=row["id"],
            content=row["content"],
            fidelity=row["fidelity"],
            compression_count=row["compression_count"],
            novelty_score=row["novelty_score"],
            access_count=row["access_count"],
            last_accessed=last_accessed,
            is_reconstructed=is_reconstructed,
            original_embedding=original_embedding,
        )

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add(self, block: LongTermBlock) -> None:
        row = self._to_row(block)
        with self._engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO lt_blocks
                    (id, content, fidelity, compression_count, novelty_score,
                     access_count, last_accessed, is_reconstructed,
                     original_embedding)
                VALUES
                    (:id, :content, :fidelity, :compression_count, :novelty_score,
                     :access_count, :last_accessed, :is_reconstructed,
                     :original_embedding)
            """), row)
            conn.commit()

    def get(self, block_id: str) -> LongTermBlock | None:
        with self._engine.connect() as conn:
            row = conn.execute(
                text("SELECT * FROM lt_blocks WHERE id = :id"),
                {"id": block_id},
            ).mappings().first()
        return self._from_row(dict(row)) if row else None

    def update(self, block: LongTermBlock) -> None:
        row = self._to_row(block)
        with self._engine.connect() as conn:
            conn.execute(text("""
                UPDATE lt_blocks SET
                    content=:content, fidelity=:fidelity,
                    compression_count=:compression_count, novelty_score=:novelty_score,
                    access_count=:access_count, last_accessed=:last_accessed,
                    is_reconstructed=:is_reconstructed,
                    original_embedding=:original_embedding
                WHERE id=:id
            """), row)
            conn.commit()

    def remove(self, block_id: str) -> LongTermBlock | None:
        block = self.get(block_id)
        if block is not None:
            with self._engine.connect() as conn:
                conn.execute(text("DELETE FROM lt_blocks WHERE id = :id"), {"id": block_id})
                conn.commit()
        return block

    # ------------------------------------------------------------------
    # Access tracking
    # ------------------------------------------------------------------

    def access(self, block_id: str) -> LongTermBlock | None:
        block = self.get(block_id)
        if block is None:
            return None
        block.record_access()
        last_accessed: Any = block.last_accessed.isoformat() if self._dialect == "sqlite" else block.last_accessed
        with self._engine.connect() as conn:
            conn.execute(text("""
                UPDATE lt_blocks SET access_count=:ac, last_accessed=:la WHERE id=:id
            """), {"ac": block.access_count, "la": last_accessed, "id": block_id})
            conn.commit()
        return block

    # ------------------------------------------------------------------
    # Vector retrieval
    # ------------------------------------------------------------------

    def similarity_search(
        self, query_vec: list[float], top_k: int
    ) -> list[tuple[LongTermBlock, float]]:
        """Return (block, similarity_score) pairs, highest similarity first."""
        if self._dialect == "sqlite":
            return self._similarity_sqlite(query_vec, top_k)
        return self._similarity_pg(query_vec, top_k)

    def _similarity_sqlite(
        self, query_vec: list[float], top_k: int
    ) -> list[tuple[LongTermBlock, float]]:
        with self._engine.connect() as conn:
            rows = conn.execute(
                text("SELECT * FROM lt_blocks WHERE original_embedding IS NOT NULL")
            ).mappings().all()
        scored = sorted(
            rows,
            key=lambda r: _cosine_sim(query_vec, _unpack_vec(r["original_embedding"])),
            reverse=True,
        )
        return [
            (self._from_row(dict(r)), _cosine_sim(query_vec, _unpack_vec(r["original_embedding"])))
            for r in scored[:top_k]
        ]

    def _similarity_pg(
        self, query_vec: list[float], top_k: int
    ) -> list[tuple[LongTermBlock, float]]:
        with self._engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT *, 1 - (original_embedding <=> CAST(:vec AS vector)) AS similarity
                FROM lt_blocks
                WHERE original_embedding IS NOT NULL
                ORDER BY original_embedding <=> CAST(:vec AS vector)
                LIMIT :k
            """), {"vec": _pg_vec_str(query_vec), "k": top_k}).mappings().all()
        return [(self._from_row(dict(r)), float(r["similarity"])) for r in rows]

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def __contains__(self, block_id: str) -> bool:
        with self._engine.connect() as conn:
            result = conn.execute(
                text("SELECT 1 FROM lt_blocks WHERE id = :id"), {"id": block_id}
            ).first()
        return result is not None

    def __len__(self) -> int:
        with self._engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM lt_blocks")).scalar()
        return result or 0

    def all_blocks(self) -> list[LongTermBlock]:
        with self._engine.connect() as conn:
            rows = conn.execute(text("SELECT * FROM lt_blocks")).mappings().all()
        return [self._from_row(dict(r)) for r in rows]
