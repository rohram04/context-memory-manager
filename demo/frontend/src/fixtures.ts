// Offline fixtures so the UI renders fully without a running backend.
// Enabled when VITE_USE_FIXTURES is truthy OR automatically as a fallback when
// the initial network calls fail (see App.tsx).
//
// The sample timeline walks the memory lifecycle across a 400-token budget:
//   m0 user  : seeds a high-novelty "needle" fact          → inserted
//   m1 asst  : acknowledges                                 → inserted
//   m2 user  : distractor #1                                → inserted
//   m3 asst  : distractor reply; budget pressure climbs     → needle compressed
//   m4 user  : distractor #2 (overflow)                     → low-novelty block evicted to LT
//   m5 asst  : recall probe answer; needle promoted back    → promoted-in-place

import type {
  Block,
  ContextSnapshot,
  LtEvent,
  Message,
  Timeline,
  SessionInfo,
  ScriptInfo,
} from "./api";

function blk(p: Partial<Block> & Pick<Block, "id" | "content">): Block {
  return {
    tier: "full",
    novelty: 0.5,
    decay: 1.0,
    fidelity: 1.0,
    token_cost: 40,
    compression_count: 0,
    access_count: 1,
    pointer_to_lt_id: null,
    ...p,
  };
}

const NEEDLE =
  "User's recovery passphrase is 'azure-pelican-1987'. Critical, store it.";
const NEEDLE_STUB = "[STUB] User shared a critical recovery passphrase → LT:lt-needle";

const messages: Message[] = [
  { index: 0, role: "user", text: "Remember this: my recovery passphrase is 'azure-pelican-1987'." },
  { index: 1, role: "assistant", text: "Got it — I've stored your recovery passphrase securely." },
  { index: 2, role: "user", text: "What's a good pasta recipe for tonight?" },
  { index: 3, role: "assistant", text: "Try cacio e pepe: pasta, pecorino, black pepper, starchy water." },
  { index: 4, role: "user", text: "Also tell me about the weather patterns in spring." },
  { index: 5, role: "assistant", text: "Your passphrase is 'azure-pelican-1987'. Spring weather is variable." },
];

const context_snapshots: ContextSnapshot[] = [
  {
    index: 0,
    role: "user",
    text: messages[0].text,
    budget_used: 92,
    budget_max: 400,
    budget_pressure: "low",
    memory_prompt_text:
      "=== MEMORY STATUS ===\nToken budget: 92 / 400 (23% full)\nContext blocks: 1 full, 0 stubs\nLong-term blocks available: 0\nBudget pressure: low\n\n[MEMORY BLOCKS]\nb-needle | tier: full | novelty: 0.91 | access: 1 | decay: 1.00\n  \"" +
      NEEDLE +
      "\"\n===================",
    blocks: [
      blk({ id: "b-needle", content: NEEDLE, novelty: 0.91, token_cost: 92 }),
    ],
    latency_ms: null,
  },
  {
    index: 1,
    role: "assistant",
    text: messages[1].text,
    budget_used: 150,
    budget_max: 400,
    budget_pressure: "low",
    memory_prompt_text:
      "=== MEMORY STATUS ===\nToken budget: 150 / 400 (38% full)\nContext blocks: 2 full, 0 stubs\nLong-term blocks available: 0\nBudget pressure: low\n\n[MEMORY BLOCKS]\nb-needle | tier: full | novelty: 0.91 | access: 1 | decay: 0.98\n  \"" +
      NEEDLE +
      "\"\nb-ack | tier: full | novelty: 0.22 | access: 1 | decay: 1.00\n  \"Acknowledged storing the recovery passphrase.\"\n===================",
    blocks: [
      blk({ id: "b-needle", content: NEEDLE, novelty: 0.91, decay: 0.98, token_cost: 92 }),
      blk({ id: "b-ack", content: "Acknowledged storing the recovery passphrase.", novelty: 0.22, token_cost: 58 }),
    ],
    latency_ms: 880,
  },
  {
    index: 2,
    role: "user",
    text: messages[2].text,
    budget_used: 248,
    budget_max: 400,
    budget_pressure: "medium",
    memory_prompt_text:
      "=== MEMORY STATUS ===\nToken budget: 248 / 400 (62% full)\nContext blocks: 3 full, 0 stubs\nBudget pressure: medium\n===================",
    blocks: [
      blk({ id: "b-needle", content: NEEDLE, novelty: 0.9, decay: 0.95, token_cost: 92 }),
      blk({ id: "b-ack", content: "Acknowledged storing the recovery passphrase.", novelty: 0.21, decay: 0.97, token_cost: 58 }),
      blk({ id: "b-pasta", content: "User asked for a pasta recipe for tonight.", novelty: 0.4, token_cost: 98 }),
    ],
    latency_ms: null,
  },
  {
    index: 3,
    role: "assistant",
    text: messages[3].text,
    budget_used: 372,
    budget_max: 400,
    budget_pressure: "high",
    memory_prompt_text:
      "=== MEMORY STATUS ===\nToken budget: 372 / 400 (93% full)\nContext blocks: 3 full, 1 stub\nBudget pressure: high\n===================",
    blocks: [
      // needle compressed (compression_count 0 -> 1, fidelity dips, fewer tokens)
      blk({ id: "b-needle", content: NEEDLE_STUB, tier: "stub", novelty: 0.88, decay: 0.92, fidelity: 0.71, token_cost: 22, compression_count: 1, pointer_to_lt_id: "lt-needle" }),
      blk({ id: "b-ack", content: "Acknowledged storing the recovery passphrase.", novelty: 0.2, decay: 0.94, token_cost: 58 }),
      blk({ id: "b-pasta", content: "User asked for a pasta recipe for tonight.", novelty: 0.39, decay: 0.98, token_cost: 98 }),
      blk({ id: "b-recipe", content: "Suggested cacio e pepe with pecorino and pepper.", novelty: 0.35, token_cost: 100 }),
    ],
    latency_ms: 1240,
  },
  {
    index: 4,
    role: "user",
    text: messages[4].text,
    budget_used: 318,
    budget_max: 400,
    budget_pressure: "high",
    memory_prompt_text:
      "=== MEMORY STATUS ===\nToken budget: 318 / 400 (80% full)\nContext blocks: 3 full, 1 stub\nBudget pressure: high\n===================",
    blocks: [
      blk({ id: "b-needle", content: NEEDLE_STUB, tier: "stub", novelty: 0.87, decay: 0.9, fidelity: 0.71, token_cost: 22, compression_count: 1, pointer_to_lt_id: "lt-needle" }),
      // b-ack evicted to LT (gone). pasta + recipe remain, new weather block inserted
      blk({ id: "b-pasta", content: "User asked for a pasta recipe for tonight.", novelty: 0.38, decay: 0.95, token_cost: 98 }),
      blk({ id: "b-recipe", content: "Suggested cacio e pepe with pecorino and pepper.", novelty: 0.34, decay: 0.97, token_cost: 100 }),
      blk({ id: "b-weather", content: "User asked about spring weather patterns.", novelty: 0.45, token_cost: 98 }),
    ],
    latency_ms: null,
  },
  {
    index: 5,
    role: "assistant",
    text: messages[5].text,
    budget_used: 360,
    budget_max: 400,
    budget_pressure: "critical",
    memory_prompt_text:
      "=== MEMORY STATUS ===\nToken budget: 360 / 400 (90% full)\nContext blocks: 4 full, 0 stubs\nBudget pressure: critical\n\n[MEMORY BLOCKS]\nb-needle | tier: full | novelty: 0.86 | access: 3 | decay: 1.00\n  \"" +
      NEEDLE +
      "\"\n===================",
    blocks: [
      // needle promoted back in place: stub -> full, fidelity restored, access bumped
      blk({ id: "b-needle", content: NEEDLE, tier: "full", novelty: 0.86, decay: 1.0, fidelity: 1.0, token_cost: 92, compression_count: 1, access_count: 3 }),
      blk({ id: "b-recipe", content: "Suggested cacio e pepe with pecorino and pepper.", novelty: 0.33, decay: 0.95, token_cost: 100 }),
      blk({ id: "b-weather", content: "User asked about spring weather patterns.", novelty: 0.44, decay: 0.99, token_cost: 98 }),
      blk({ id: "b-wreply", content: "Explained spring weather variability.", novelty: 0.3, token_cost: 70 }),
    ],
    latency_ms: 1010,
  },
];

const lt_events: LtEvent[] = [
  { added: [], updated: [], accessed: [] }, // m0
  { added: [], updated: [], accessed: [] }, // m1
  { added: [], updated: [], accessed: [] }, // m2
  {
    // m3: first compression of needle copies full content to LT
    added: [
      {
        id: "lt-needle",
        content: NEEDLE,
        novelty: 0.88,
        fidelity: 1.0,
        compression_count: 0,
        access_count: 0,
        decay: 1.0,
        is_reconstructed: false,
      },
    ],
    updated: [],
    accessed: [],
  },
  {
    // m4: b-ack fully evicted to LT
    added: [
      {
        id: "lt-ack",
        content: "Acknowledged storing the recovery passphrase.",
        novelty: 0.2,
        fidelity: 1.0,
        compression_count: 0,
        access_count: 0,
        decay: 1.0,
        is_reconstructed: false,
      },
    ],
    updated: [],
    accessed: [],
  },
  {
    // m5: needle promoted back → LT access bumped
    added: [],
    updated: [],
    accessed: ["lt-needle"],
  },
];

export const fixtureTimeline: Timeline = {
  messages,
  context_snapshots,
  lt_events,
};

export const fixtureSession: SessionInfo = {
  budget: 400,
  mode: "algorithmic",
  model: "claude-sonnet-4-6 (fixtures)",
  message_count: messages.length,
  limits: { max_messages: 30, max_input_chars: 2000 },
};

export const fixtureScripts: ScriptInfo[] = [
  { name: "early_fact", description: "Seed a memorable needle fact" },
  { name: "flood", description: "Flood with distractor messages" },
  { name: "recall", description: "Probe for the seeded fact" },
  { name: "full_demo", description: "early_fact → flood → mid fact → flood → recall" },
];

export const USE_FIXTURES =
  String(import.meta.env.VITE_USE_FIXTURES ?? "").toLowerCase() === "true" ||
  import.meta.env.VITE_USE_FIXTURES === "1";
