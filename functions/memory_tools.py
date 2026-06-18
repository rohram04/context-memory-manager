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
                "Call this only when no existing context block covers the same topic — "
                "check MEMORY BLOCKS first and prefer augment() if a related block exists. "
                "Set novelty high (close to 1.0) for unique, surprising, or critical information; "
                "low (close to 0.0) for routine or redundant information. "
                "You may call this multiple times in one turn to persist several distinct facts."
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
                "Call this when budget pressure is medium or higher, targeting low-novelty "
                "or low-decay blocks first (both shown in MEMORY BLOCKS). "
                "Read the block's current content from MEMORY BLOCKS, write your own concise "
                "summary as compressed_content (it must be SHORTER than the current content), "
                "and omit details that are no longer relevant. 'Fidelity' (shown in MEMORY "
                "BLOCKS) measures how much of the original meaning survives; if it drops below "
                "~0.5 after a pass the block is removed from context (long-term copy preserved, "
                "recoverable via promote or query_lt). You may compress multiple blocks in one turn."
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
                        "description": (
                            "Your concise summary of the block's current content — must be "
                            "shorter than it currently is."
                        ),
                    },
                },
                "required": ["block_id", "compressed_content"],
            },
        },
        {
            "name": "evict",
            "description": (
                "Immediately move a context block to long-term memory without compressing it. "
                "Call this when budget pressure is high or critical and a block has low decay "
                "(not accessed recently) and is unlikely to be needed in the next few turns. "
                "Prefer evict over compress when you want to remove the block entirely rather "
                "than keep a stub; evicted blocks remain recoverable via query_lt + promote. "
                "You may evict multiple blocks in one turn."
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
                "Call this when the user's question or the current task relates to information "
                "that is currently in long-term memory (use query_lt first to find candidate IDs). "
                "If the block already has a stub in context, the stub is expanded in place. "
                "You may promote multiple blocks in one turn."
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
                "Call this whenever new information relates to a topic already covered by a "
                "context block — prefer this over store() to avoid duplicate blocks. "
                "Read the block's current content from MEMORY BLOCKS, then write the complete "
                "merged result in new_content: integrate the existing facts and the new "
                "information into one coherent block, dropping anything superseded. "
                "Do not concatenate — synthesize. The block's embedding and fidelity are reset. "
                "You may augment multiple blocks in one turn."
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
                        "description": "The complete merged content for the block (not an append — a full rewrite).",
                    },
                },
                "required": ["block_id", "new_content"],
            },
        },
        {
            "name": "query_lt",
            "description": (
                "Search long-term memory by semantic similarity to a natural language query. "
                "Call this at the start of each turn to check whether the user's question or "
                "the current topic has relevant information in long-term storage. "
                "If results look useful, follow up with promote() to bring them into context. "
                "Returns the top matching blocks with IDs, similarity scores, and content previews."
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
                "Set the novelty score for a context block. "
                "Call this when new information reveals that an existing block is more or less "
                "significant than originally scored — for example, a fact that seemed routine "
                "turns out to be a key constraint, or a previously surprising fact becomes stale. "
                "Higher novelty (closer to 1.0) makes the block resist compression; "
                "lower novelty (closer to 0.0) lets it compress sooner."
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
