# Known Bugs / Open Issues

Tracked medium-severity issues in the core memory manager (non-demo). Each entry
notes the location, the problem, and the suggested fix. Fixed issues:

- Stale compression-heap priority in `memory/store.py:next_candidate`
- OpenRouter tool calls dropped on `finish_reason="stop"` in
  `llm/openrouter_tools.py:_map_finish_reason`
- `promote()` not refreshing access/recency on the promoted in-context block
  (`ContextManager.py:promote()` — now calls `store.access()` on both branches)
- `compress()` early-return removing blocks without LT reconciliation
  (`ContextManager.py:compress()` — now routes through `evict()`)

---

## 1. Long-term store omits the current-content `embedding` column

**Location:** `memory/longterm.py` — `LongTermBlock` table + `similarity_search()`.

**Problem:** The documented schema (CLAUDE.md) gives `LongTermBlock` two vectors:
`original_embedding` (full-content vector, never updated) and `embedding` (vector
of the *current* possibly-compressed content, used for similarity retrieval). The
implementation only stores `original_embedding` and ranks `similarity_search()`
against it. So LT retrieval always matches on the original vector even after an
LT-side compression pass changes the content. This is functionally consistent
today only because LT-side compression is not yet wired up; it diverges from the
documented design and will produce stale retrieval matches once LT compression
lands.

**Fix:** Add the `embedding` column (current-content vector), update it whenever LT
content changes, and rank `similarity_search()` against `embedding` while keeping
`original_embedding` for fidelity computation.
