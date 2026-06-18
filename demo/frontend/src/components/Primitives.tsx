import { useEffect, useRef, useState } from "react";
import * as Tooltip from "@radix-ui/react-tooltip";
import { motion, useMotionValue, useTransform, animate } from "framer-motion";

export function InfoTip({
  children,
  label,
}: {
  children: React.ReactNode;
  label: string;
}) {
  return (
    <Tooltip.Root delayDuration={150}>
      <Tooltip.Trigger asChild>
        <span className="cursor-help">{children}</span>
      </Tooltip.Trigger>
      <Tooltip.Portal>
        <Tooltip.Content
          sideOffset={6}
          className="z-50 max-w-[240px] rounded-md border border-ide-borderBright bg-ide-card px-2.5 py-1.5 text-xs leading-snug text-ide-text shadow-elevated"
        >
          {label}
          <Tooltip.Arrow className="fill-ide-borderBright" />
        </Tooltip.Content>
      </Tooltip.Portal>
    </Tooltip.Root>
  );
}

/** A small labeled metric bar (0..1). */
export function MetricBar({
  label,
  value,
  color,
  tip,
  delta,
}: {
  label: string;
  value: number;
  color: string;
  tip: string;
  delta?: number;
}) {
  const pct = Math.max(0, Math.min(1, value)) * 100;
  const arrow =
    delta !== undefined && Math.abs(delta) > 1e-4
      ? delta > 0
        ? "▲"
        : "▼"
      : null;
  return (
    <InfoTip label={tip}>
      <div className="flex items-center gap-1.5">
        <span className="w-[34px] shrink-0 text-[10px] uppercase tracking-wide text-ide-textFaint">
          {label}
        </span>
        <div className="relative h-1.5 w-12 overflow-hidden rounded-full bg-ide-border">
          <motion.div
            className="absolute inset-y-0 left-0 rounded-full"
            style={{ backgroundColor: color }}
            initial={false}
            animate={{ width: `${pct}%` }}
            transition={{ type: "spring", stiffness: 220, damping: 28 }}
          />
        </div>
        <span className="w-[30px] shrink-0 font-mono text-[10px] tabular-nums text-ide-textDim">
          {value.toFixed(2)}
        </span>
        {arrow && (
          <span
            className="text-[9px]"
            style={{ color: delta! > 0 ? "#3fb950" : "#f85149" }}
          >
            {arrow}
          </span>
        )}
      </div>
    </InfoTip>
  );
}

/** Count-up animated integer. */
export function CountUp({ value }: { value: number }) {
  const mv = useMotionValue(value);
  const rounded = useTransform(mv, (v) => Math.round(v).toString());
  const [text, setText] = useState(value.toString());
  useEffect(() => {
    const controls = animate(mv, value, { duration: 0.5, ease: "easeOut" });
    const unsub = rounded.on("change", (v) => setText(v));
    return () => {
      controls.stop();
      unsub();
    };
  }, [value, mv, rounded]);
  return <span className="tabular-nums">{text}</span>;
}

/** Hook: previous value of x. */
export function usePrev<T>(x: T): T | undefined {
  const ref = useRef<T | undefined>(undefined);
  useEffect(() => {
    ref.current = x;
  }, [x]);
  return ref.current;
}
