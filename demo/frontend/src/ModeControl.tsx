import * as Tooltip from "@radix-ui/react-tooltip";
import { AnimatePresence, motion } from "framer-motion";
import type { Mode } from "./api";

const OPTIONS: { value: Mode; label: string }[] = [
  { value: "algorithmic", label: "Algorithmic" },
  { value: "llm", label: "LLM" },
];

export function ModeControl({
  mode,
  onChange,
  disabled,
}: {
  mode: Mode;
  onChange: (m: Mode) => void;
  disabled: boolean;
}) {
  return (
    <div className="flex items-center gap-1.5">
      <span className="text-[10px] uppercase tracking-wider text-ide-textFaint">
        mode
      </span>
      <div
        className="flex overflow-hidden rounded-md border border-ide-border"
        role="radiogroup"
        aria-label="Memory mode"
      >
        {OPTIONS.map((o) => {
          const active = o.value === mode;
          return (
            <button
              key={o.value}
              role="radio"
              aria-checked={active}
              disabled={disabled}
              onClick={() => !active && onChange(o.value)}
              className={`px-2.5 py-1 text-xs font-medium transition ${
                active
                  ? "bg-accent text-white"
                  : "bg-ide-card text-ide-textDim hover:text-ide-text disabled:opacity-40"
              }`}
            >
              {o.label}
            </button>
          );
        })}
      </div>
      <AnimatePresence>
        {mode === "llm" && (
          <Tooltip.Root>
            <Tooltip.Trigger asChild>
              <motion.span
                initial={{ opacity: 0, x: -4 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -4 }}
                transition={{ duration: 0.15 }}
                className="cursor-help rounded border border-fuchsia-500/40 bg-fuchsia-500/10 px-1.5 py-0.5 text-[10px] font-medium text-fuchsia-300"
              >
                model-managed
              </motion.span>
            </Tooltip.Trigger>
            <Tooltip.Portal>
              <Tooltip.Content
                side="bottom"
                sideOffset={6}
                className="z-50 max-w-[260px] rounded-md border border-ide-border bg-ide-panel px-2.5 py-1.5 text-[11px] leading-snug text-ide-textDim shadow-lg"
              >
                The model self-manages its own memory via tools
                (store/augment/compress/evict/promote). Slower, more LLM tool
                calls per turn, and may cost more.
                <Tooltip.Arrow className="fill-ide-border" />
              </Tooltip.Content>
            </Tooltip.Portal>
          </Tooltip.Root>
        )}
      </AnimatePresence>
    </div>
  );
}
