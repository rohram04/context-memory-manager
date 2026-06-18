import { useMemo } from "react";
import { AnimatePresence, motion } from "framer-motion";
import type { LtEvent, LtView } from "./api";
import { MetricBar, InfoTip } from "./components/Primitives";

type Hl = "added" | "updated" | "accessed" | null;

const HL_COLOR: Record<Exclude<Hl, null>, string> = {
  added: "#3fb950",
  updated: "#d29922",
  accessed: "#58a6ff",
};

function foldLt(events: LtEvent[], upto: number) {
  const map = new Map<string, LtView>();
  for (let i = 0; i <= upto && i < events.length; i++) {
    const ev = events[i];
    for (const a of ev.added) map.set(a.id, a);
    for (const u of ev.updated) map.set(u.id, u);
    // accessed bumps access_count on the folded view (server snapshots also
    // reflect this, but fold defensively)
    for (const id of ev.accessed) {
      const cur = map.get(id);
      if (cur) map.set(id, { ...cur, access_count: cur.access_count });
    }
  }
  return map;
}

function LtCard({ blk, hl }: { blk: LtView; hl: Hl }) {
  const glow = hl ? HL_COLOR[hl] : null;
  return (
    <motion.div
      layout
      key={blk.id}
      initial={{ opacity: 0, scale: 0.92, y: 8 }}
      animate={{
        opacity: 1,
        scale: 1,
        y: 0,
        boxShadow: glow
          ? `0 0 0 1px ${glow}, 0 0 18px -2px ${glow}99`
          : "0 1px 2px rgba(0,0,0,0.4)",
      }}
      exit={{ opacity: 0, scale: 0.9 }}
      transition={{ type: "spring", stiffness: 260, damping: 30 }}
      data-lt-id={blk.id}
      className="rounded-lg border border-ide-border bg-ide-card p-3 hover:border-ide-borderBright hover:bg-ide-cardHover"
    >
      <div className="mb-2 flex items-center justify-between gap-2">
        <span className="font-mono text-[10px] text-ide-textFaint">{blk.id}</span>
        <div className="flex items-center gap-1.5">
          {blk.is_reconstructed && (
            <InfoTip label="Rebuilt via LLM reconstruction after heavy LT compression (lossy).">
              <span className="rounded bg-fuchsia-500/15 px-1.5 py-0.5 text-[9px] font-semibold uppercase tracking-wide text-fuchsia-300">
                reconstructed
              </span>
            </InfoTip>
          )}
          {hl && (
            <span
              className="rounded px-1.5 py-0.5 text-[9px] font-semibold uppercase tracking-wide"
              style={{ color: HL_COLOR[hl], backgroundColor: HL_COLOR[hl] + "1f" }}
            >
              {hl}
            </span>
          )}
        </div>
      </div>
      <p className="mb-2.5 whitespace-pre-wrap break-words font-mono text-[12px] leading-relaxed text-ide-text">
        {blk.content}
      </p>
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5">
        <MetricBar
          label="nov"
          value={blk.novelty}
          color="#a371f7"
          tip="Novelty (0–1) carried over from the context block."
        />
        <MetricBar
          label="dec"
          value={blk.decay}
          color="#58a6ff"
          tip="LT decay (0–1) — decays slower than context blocks."
        />
        <MetricBar
          label="fid"
          value={blk.fidelity}
          color="#3fb950"
          tip="Fidelity (0–1) vs original full content."
        />
        <span className="font-mono text-[10px] text-ide-textDim">
          ×{blk.compression_count}
        </span>
        <span className="font-mono text-[10px] text-ide-textDim">
          acc {blk.access_count}
        </span>
      </div>
    </motion.div>
  );
}

export function LongTermTab({
  events,
  selectedIndex,
  highlightLtId,
}: {
  events: LtEvent[];
  selectedIndex: number;
  highlightLtId: string | null;
}) {
  const { blocks, hlById } = useMemo(() => {
    const folded = foldLt(events, selectedIndex);
    const ev = events[selectedIndex];
    const hl = new Map<string, Hl>();
    if (ev) {
      for (const a of ev.added) hl.set(a.id, "added");
      for (const u of ev.updated) if (!hl.has(u.id)) hl.set(u.id, "updated");
      for (const id of ev.accessed) if (!hl.has(id)) hl.set(id, "accessed");
    }
    return { blocks: Array.from(folded.values()), hlById: hl };
  }, [events, selectedIndex]);

  return (
    <div className="flex h-full flex-col">
      <div className="border-b border-ide-border bg-ide-panel/60 px-3 py-2.5 text-[11px] text-ide-textFaint">
        {blocks.length} long-term block{blocks.length === 1 ? "" : "s"} at this
        message
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto p-3">
        {blocks.length === 0 ? (
          <div className="py-8 text-center text-sm text-ide-textFaint">
            Long-term store is empty — nothing has been compressed or evicted yet.
          </div>
        ) : (
          <div className="flex flex-col gap-2.5">
            <AnimatePresence mode="popLayout" initial={false}>
              {blocks.map((b) => (
                <LtCard
                  key={b.id}
                  blk={b}
                  hl={
                    highlightLtId === b.id
                      ? "accessed"
                      : hlById.get(b.id) ?? null
                  }
                />
              ))}
            </AnimatePresence>
          </div>
        )}
      </div>
    </div>
  );
}
