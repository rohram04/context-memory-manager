# Known Bugs / Open Issues

Tracked medium-severity issues in the core memory manager (non-demo). Each entry
notes the location, the problem, and the suggested fix. Two related high-severity
bugs (stale compression-heap priority in `memory/store.py:next_candidate`, and
OpenRouter tool calls dropped on `finish_reason="stop"` in
`llm/openrouter_tools.py:_map_finish_reason`) were fixed in the same commit that
introduced this file.

---

## 1. `promote()` does not refresh access/recency on the promoted block

**Location:** `ContextManager.py` — `promote()` (existing-stub branch).

**Problem:** When promoting a long-term block back into context over an existing
stub, the method updates content/fidelity/compression_count and re-sorts the heap,
but never calls `record_access()` on the in-context `CacheBlock`. Only the LT
record's access count is bumped. As a result a just-promoted block does not gain
the recency/decay weight that the rest of the system assumes ("just-promoted
blocks are freshly accessed → low compression priority"). The new-block promotion
path is similarly affected: the created `CacheBlock` defaults to `access_count = 0`.
The augmentation path, by contrast, *does* call `record_access()`.

**Fix:** Call `record_access()` on the promoted/updated `CacheBlock` (both the
existing-stub branch and the new-block branch) so promotion honors the
"freshly accessed" invariant, then re-sort the heap.

---

## 2. Long-term store omits the current-content `embedding` column

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

---

## 3. `compress()` early-return removes a block without LT reconciliation

**Location:** `ContextManager.py` — `compress()`, top-of-function fidelity guard.

**Problem:** `compress()` opens with a guard that removes the block from context if
its fidelity is already below the removal threshold *before* any new compression
pass. That path calls `self._store.remove(...)` directly and returns, skipping the
reconcile-to-LT step. If the block was augmented since its last LT sync (i.e. it is
"dirty"), the information gained since then is lost because the LT copy is never
updated before removal. The branch is normally dead — a block that drops below
threshold is already removed and exits the heap on the pass that dropped it — so it
is only reachable via a stale heap entry, but the silent data loss is a real risk.

**Fix:** Either drop the redundant guard entirely, or route it through the same
reconcile-then-remove path that eviction uses, so a dirty block's LT copy is
brought up to date before the in-context block is removed.
