import { useMemo, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import type { Block, ContextSnapshot } from "./api";
import {
  computeDiff,
  CATEGORY_COLOR,
  CATEGORY_LABEL,
  type DiffEntry,
} from "./diff";
import { BudgetMeter } from "./components/BudgetMeter";
import { MetricBar, InfoTip } from "./components/Primitives";

const TIPS = {
  novelty:
    "Novelty (0–1): how semantically distinct/surprising this block is. High novelty resists compression.",
  decay:
    "Decay (0–1): time-weighted retention. Drops with time since last access; low decay → compressed sooner.",
  fidelity:
    "Fidelity (0–1): cosine similarity of current content vs the original full block. Falls below threshold → removed from context.",
  token_cost: "Tokens this block consumes in the context window.",
  compression: "Number of compression passes applied to this block.",
  access: "How many times this block has been accessed.",
  tier: "full = complete content in context. stub = compressed summary pointing to a long-term copy.",
};

function CategoryBadge({ entry }: { entry: DiffEntry }) {
  if (entry.category === "unchanged") return null;
  const color = CATEGORY_COLOR[entry.category] ?? "#8b949e";
  return (
    <span
      className="rounded px-1.5 py-0.5 text-[9px] font-semibold uppercase tracking-wide"
      style={{ color, backgroundColor: color + "1f" }}
    >
      {CATEGORY_LABEL[entry.category]}
    </span>
  );
}

function BlockCard({
  block,
  entry,
  onGotoLt,
}: {
  block: Block;
  entry: DiffEntry;
  onGotoLt: (ltId: string) => void;
}) {
  const glow = CATEGORY_COLOR[entry.category];
  const isStub = block.tier === "stub";

  return (
    <motion.div
      layout
      key={block.id}
      initial={{ opacity: 0, scale: 0.92, y: 8 }}
      animate={{
        opacity: 1,
        scale: 1,
        y: 0,
        boxShadow: glow
          ? `0 0 0 1px ${glow}, 0 0 18px -2px ${glow}99`
          : "0 1px 2px rgba(0,0,0,0.4)",
      }}
      exit={{
        opacity: 0,
        scale: 0.9,
        boxShadow: `0 0 0 1px #f85149, 0 0 18px -2px #f8514999`,
        transition: { duration: 0.28 },
      }}
      transition={{ type: "spring", stiffness: 260, damping: 30 }}
      className="group rounded-lg border border-ide-border bg-ide-card p-3 hover:border-ide-borderBright hover:bg-ide-cardHover"
    >
      <div className="mb-2 flex items-center justify-between gap-2">
        <div className="flex items-center gap-1.5">
          <InfoTip label={TIPS.tier}>
            <span
              className={`rounded px-1.5 py-0.5 text-[9px] font-bold uppercase tracking-wider ${
                isStub
                  ? "bg-amber-500/15 text-amber-300"
                  : "bg-emerald-500/15 text-emerald-300"
              }`}
            >
              {block.tier}
            </span>
          </InfoTip>
          <span className="font-mono text-[10px] text-ide-textFaint">
            {block.id}
          </span>
        </div>
        <CategoryBadge entry={entry} />
      </div>

      <motion.p
        layout="position"
        className="mb-2.5 whitespace-pre-wrap break-words font-mono text-[12px] leading-relaxed text-ide-text"
      >
        {block.content}
      </motion.p>

      <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5">
        <MetricBar
          label="nov"
          value={block.novelty}
          color="#a371f7"
          tip={TIPS.novelty}
          delta={entry.fieldDeltas.novelty}
        />
        <MetricBar
          label="dec"
          value={block.decay}
          color="#58a6ff"
          tip={TIPS.decay}
          delta={entry.fieldDeltas.decay}
        />
        <MetricBar
          label="fid"
          value={block.fidelity}
          color="#3fb950"
          tip={TIPS.fidelity}
        />
        <InfoTip label={TIPS.token_cost}>
          <span className="font-mono text-[10px] text-ide-textDim">
            {block.token_cost}t
          </span>
        </InfoTip>
        <InfoTip label={TIPS.compression}>
          <span className="font-mono text-[10px] text-ide-textDim">
            ×{block.compression_count}
          </span>
        </InfoTip>
        <InfoTip label={TIPS.access}>
          <span className="font-mono text-[10px] text-ide-textDim">
            acc {block.access_count}
          </span>
        </InfoTip>
        {isStub && (
          <button
            onClick={() => block.pointer_to_lt_id && onGotoLt(block.pointer_to_lt_id)}
            className="ml-auto rounded border border-accent/40 px-1.5 py-0.5 text-[10px] text-accent transition hover:bg-accent/15"
          >
            → LT
          </button>
        )}
      </div>
    </motion.div>
  );
}

export function ContextTab({
  snapshot,
  prevSnapshot,
  onGotoLt,
}: {
  snapshot: ContextSnapshot | undefined;
  prevSnapshot: ContextSnapshot | undefined;
  onGotoLt: (ltId: string) => void;
}) {
  const [raw, setRaw] = useState(false);

  const diff = useMemo(
    () => (snapshot ? computeDiff(prevSnapshot, snapshot) : new Map()),
    [snapshot, prevSnapshot]
  );

  if (!snapshot) {
    return (
      <div className="flex h-full items-center justify-center text-sm text-ide-textFaint">
        No snapshot for this message.
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col">
      <div className="border-b border-ide-border bg-ide-panel/60 px-3 py-2.5">
        <BudgetMeter
          used={snapshot.budget_used}
          max={snapshot.budget_max}
          pressure={snapshot.budget_pressure}
        />
        <div className="mt-2 flex items-center justify-between">
          <span className="text-[11px] text-ide-textFaint">
            {snapshot.blocks.length} block
            {snapshot.blocks.length === 1 ? "" : "s"} in context
          </span>
          <button
            onClick={() => setRaw((r) => !r)}
            className={`rounded border px-2 py-0.5 text-[11px] transition ${
              raw
                ? "border-accent bg-accent/15 text-accent"
                : "border-ide-border text-ide-textDim hover:border-ide-borderBright"
            }`}
          >
            raw text
          </button>
        </div>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto p-3">
        {raw ? (
          <pre className="whitespace-pre-wrap break-words rounded-lg border border-ide-border bg-black/40 p-3 font-mono text-[11px] leading-relaxed text-ide-text">
            {snapshot.memory_prompt_text}
          </pre>
        ) : (
          <div className="flex flex-col gap-2.5">
            <AnimatePresence mode="popLayout" initial={false}>
              {snapshot.blocks.map((b) => (
                <BlockCard
                  key={b.id}
                  block={b}
                  entry={
                    diff.get(b.id) ?? {
                      category: "unchanged",
                      fieldDeltas: {
                        novelty: 0,
                        decay: 0,
                        access: 0,
                        token_cost: 0,
                      },
                    }
                  }
                  onGotoLt={onGotoLt}
                />
              ))}
            </AnimatePresence>
            {snapshot.blocks.length === 0 && (
              <div className="py-8 text-center text-sm text-ide-textFaint">
                Context window is empty at this message.
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
