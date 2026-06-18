// Net-per-message diff for the Context tab.
// Compares two consecutive context snapshots and classifies each block by the
// NET effect of the message that produced `curr`. Purely client-side; no server
// diffing, no embeddings.

import type { Block, ContextSnapshot } from "./api";

export type DiffCategory =
  | "unchanged"
  | "compressed"
  | "augmented"
  | "promoted-in-place"
  | "inserted"
  | "promoted"
  | "removed";

export interface FieldDeltas {
  novelty: number;
  decay: number;
  access: number;
  token_cost: number;
}

export interface DiffEntry {
  category: DiffCategory;
  fieldDeltas: FieldDeltas;
}

export type DiffMap = Map<string, DiffEntry>;

const EPS = 1e-6;

function deltas(prev: Block | undefined, curr: Block): FieldDeltas {
  return {
    novelty: curr.novelty - (prev?.novelty ?? curr.novelty),
    decay: curr.decay - (prev?.decay ?? curr.decay),
    access: curr.access_count - (prev?.access_count ?? curr.access_count),
    token_cost: curr.token_cost - (prev?.token_cost ?? curr.token_cost),
  };
}

/**
 * computeDiff(prev, curr) → Map<blockId, {category, fieldDeltas}>.
 * `prev` undefined (first message) → every block is "inserted"/"promoted".
 */
export function computeDiff(
  prev: ContextSnapshot | undefined,
  curr: ContextSnapshot
): DiffMap {
  const out: DiffMap = new Map();
  const prevById = new Map<string, Block>();
  for (const b of prev?.blocks ?? []) prevById.set(b.id, b);
  const currIds = new Set(curr.blocks.map((b) => b.id));

  for (const b of curr.blocks) {
    const p = prevById.get(b.id);
    if (!p) {
      // Unmatched in curr. A stub that wasn't in prev = re-promotion after a
      // full eviction.
      const category: DiffCategory = b.tier === "stub" ? "promoted" : "inserted";
      out.set(b.id, { category, fieldDeltas: deltas(undefined, b) });
      continue;
    }

    const fd = deltas(p, b);
    let category: DiffCategory;
    const contentSame = p.content === b.content;

    if (b.compression_count > p.compression_count) {
      category = "compressed";
    } else if (contentSame) {
      category = "unchanged";
    } else if (b.fidelity > p.fidelity + EPS) {
      // content changed and fidelity rose → a stub expanded back to fuller form
      category = "promoted-in-place";
    } else {
      // content changed, fidelity ~ unchanged/high → merged new content in
      category = "augmented";
    }
    out.set(b.id, { category, fieldDeltas: fd });
  }

  // Blocks present in prev but gone from curr → removed (evicted to LT).
  for (const p of prev?.blocks ?? []) {
    if (!currIds.has(p.id)) {
      out.set(p.id, {
        category: "removed",
        fieldDeltas: { novelty: 0, decay: 0, access: 0, token_cost: 0 },
      });
    }
  }

  return out;
}

export const CATEGORY_LABEL: Record<DiffCategory, string> = {
  unchanged: "unchanged",
  compressed: "compressed",
  augmented: "augmented",
  "promoted-in-place": "promoted in place",
  inserted: "inserted",
  promoted: "promoted",
  removed: "removed → LT",
};

// glow color per category (matches tailwind diff palette)
export const CATEGORY_COLOR: Record<DiffCategory, string | null> = {
  unchanged: null,
  compressed: "#d29922",
  augmented: "#58a6ff",
  "promoted-in-place": "#58a6ff",
  inserted: "#3fb950",
  promoted: "#3fb950",
  removed: "#f85149",
};
