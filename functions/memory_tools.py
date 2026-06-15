from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from memory.block import CacheBlock

if TYPE_CHECKING:
    from ContextManager import ContextManager


class MemoryTools:
    """Anthropic tool schemas and dispatch for the LLM memory tool surface.

    Pass SCHEMAS to the `tools=` parameter of every API call.
    On each tool_use block in the response, call dispatch(name, input).
    """

    SCHEMAS: ClassVar[list[dict]] = [
        {
            "name": "store",
            "description": (
                "Create a new memory block with the given content. "
                "Use this to persist information you want to remember across turns. "
                "Set novelty high (close to 1.0) for unique or surprising information, "
                "low (close to 0.0) for routine or redundant information."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The content to store as a new memory block.",
                    },
                    "novelty": {
                        "type": "number",
                        "description": "Novelty score in [0.0, 1.0]. Controls compression resistance.",
                    },
                },
                "required": ["content", "novelty"],
            },
        },
        {
            "name": "compress",
            "description": (
                "Compress a context block to reduce its token cost. "
                "Write a summary of the block as compressed_content — "
                "the full block content is shown in MEMORY STATUS. "
                "If fidelity drops below threshold after compression the block is removed from "
                "context (long-term copy preserved, recoverable via promote or query_lt)."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "block_id": {
                        "type": "string",
                        "description": "ID of the context block to compress (cb_...).",
                    },
                    "compressed_content": {
                        "type": "string",
                        "description": "Your summary of the block content.",
                    },
                },
                "required": ["block_id", "compressed_content"],
            },
        },
        {
            "name": "evict",
            "description": (
                "Immediately move a context block to long-term memory without compressing it. "
                "Use when a block is unlikely to be needed soon and you want to free token budget."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "block_id": {
                        "type": "string",
                        "description": "ID of the context block to evict (cb_...).",
                    },
                },
                "required": ["block_id"],
            },
        },
        {
            "name": "promote",
            "description": (
                "Promote a long-term memory block back into the active context window. "
                "If the block already has a stub in context, the stub is expanded in place."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "lt_block_id": {
                        "type": "string",
                        "description": "ID of the long-term block to promote (lt_...).",
                    },
                },
                "required": ["lt_block_id"],
            },
        },
        {
            "name": "augment",
            "description": (
                "Replace an existing context block's content with a new merged version. "
                "Write the complete merged result in new_content — combine what the block "
                "currently says with the new information into a single coherent block. "
                "The block's embedding is reset and fidelity restored to 1.0."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "block_id": {
                        "type": "string",
                        "description": "ID of the context block to augment (cb_...).",
                    },
                    "new_content": {
                        "type": "string",
                        "description": "The complete merged content for the block.",
                    },
                },
                "required": ["block_id", "new_content"],
            },
        },
        {
            "name": "query_lt",
            "description": (
                "Search long-term memory by semantic similarity to a natural language query. "
                "Returns the top matching blocks with their metadata. "
                "Use this to check whether relevant information exists in LT before promoting."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query to search long-term memory.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (default 5).",
                    },
                },
                "required": ["query"],
            },
        },
        {
            "name": "update_novelty",
            "description": (
                "Set the novelty score for a context block. Higher novelty (closer to 1.0) "
                "makes the block resist compression; lower novelty (closer to 0.0) makes it "
                "compress sooner. Use when new information changes how significant a block is."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "block_id": {
                        "type": "string",
                        "description": "ID of the context block to update (cb_...).",
                    },
                    "novelty": {
                        "type": "number",
                        "description": "New novelty score in [0.0, 1.0].",
                    },
                },
                "required": ["block_id", "novelty"],
            },
        },
    ]

    def __init__(self, context_manager: ContextManager) -> None:
        self._cm = context_manager

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def dispatch(self, tool_name: str, tool_input: dict) -> str:
        """Route a tool_use block from the LLM to the appropriate handler."""
        handlers = {
            "store": self._store,
            "compress": self._compress,
            "evict": self._evict,
            "promote": self._promote,
            "augment": self._augment,
            "query_lt": self._query_lt,
            "update_novelty": self._update_novelty,
        }
        handler = handlers.get(tool_name)
        if handler is None:
            return f"Error: unknown tool '{tool_name}'."
        try:
            return handler(**tool_input)
        except TypeError as e:
            return f"Error: bad arguments for '{tool_name}': {e}"

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def _store(self, content: str, novelty: float) -> str:
        embedding = self._cm.embed(content)
        block = CacheBlock(
            content=content,
            original_embedding=embedding,
            novelty_score=max(0.0, min(1.0, novelty)),
        )
        self._cm.insert(block)
        return f"Stored new block {block.id}. Token cost: {block.token_cost}."

    def _compress(self, block_id: str, compressed_content: str) -> str:
        block = self._cm.compress(block_id, compressed_content)
        if block is None:
            return f"Error: block {block_id} not found."
        if block_id not in self._cm._store:
            return (
                f"Compressed {block_id} — removed "
                f"(fidelity {block.fidelity:.2f} below threshold)."
            )
        return (
            f"Compressed {block_id}. "
            f"Fidelity: {block.fidelity:.2f}. Token cost: {block.token_cost}."
        )

    def _evict(self, block_id: str) -> str:
        block = self._cm.evict(block_id)
        if block is None:
            return f"Error: block {block_id} not found."
        lt_id = block.pointer_to_lt_id or "unknown"
        return f"Evicted {block_id} → LT {lt_id}."

    def _promote(self, lt_block_id: str) -> str:
        block = self._cm.promote(lt_block_id)
        if block is None:
            return f"Error: LT block {lt_block_id} not found."
        return f"Promoted {lt_block_id} → context block {block.id}."

    def _augment(self, block_id: str, new_content: str) -> str:
        embedding = self._cm.embed(new_content)
        block = self._cm.augment(block_id, new_content, embedding)
        if block is None:
            return f"Error: block {block_id} not found."
        return f"Augmented {block_id}. New token cost: {block.token_cost}."

    def _query_lt(self, query: str, top_k: int = 5) -> str:
        vec = self._cm.embed(query)
        results = self._cm._lt.similarity_search(vec, top_k)
        if not results:
            return "No matching long-term blocks found."
        lines = [f"Long-term search results for '{query}':"]
        for lt_block, score in results:
            preview = lt_block.content[:80] + ("..." if len(lt_block.content) > 80 else "")
            lines.append(
                f"  {lt_block.id} | sim: {score:.2f} | novelty: {lt_block.novelty_score:.2f}"
                f" | decay: {lt_block.decay_score:.2f} | fidelity: {lt_block.fidelity:.2f}"
            )
            lines.append(f'    "{preview}"')
        return "\n".join(lines)

    def _update_novelty(self, block_id: str, novelty: float) -> str:
        block = self._cm.update_novelty(block_id, novelty)
        if block is None:
            return f"Error: block {block_id} not found."
        return f"Updated novelty for {block_id}: {block.novelty_score:.2f}."
