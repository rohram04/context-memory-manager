import { motion } from "framer-motion";
import { CountUp } from "./Primitives";

const PRESSURE_COLOR: Record<string, string> = {
  low: "#3fb950",
  medium: "#d29922",
  high: "#f0883e",
  critical: "#f85149",
};

export function BudgetMeter({
  used,
  max,
  pressure,
}: {
  used: number;
  max: number;
  pressure: string;
}) {
  const pct = max > 0 ? Math.min(1, used / max) : 0;
  const color = PRESSURE_COLOR[pressure] ?? "#58a6ff";
  const critical = pressure === "critical";

  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-baseline justify-between text-xs">
        <span className="text-ide-textDim">
          Budget{" "}
          <span className="font-mono text-ide-text">
            <CountUp value={used} /> / {max}
          </span>{" "}
          <span className="text-ide-textFaint">tokens</span>
        </span>
        <span
          className={`rounded px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide ${
            critical ? "animate-pulseCritical" : ""
          }`}
          style={{ color, backgroundColor: color + "22" }}
        >
          {pressure}
        </span>
      </div>
      <div className="relative h-2 w-full overflow-hidden rounded-full bg-ide-border">
        <motion.div
          className="absolute inset-y-0 left-0 rounded-full"
          style={{ background: `linear-gradient(90deg, ${color}cc, ${color})` }}
          initial={false}
          animate={{ width: `${pct * 100}%` }}
          transition={{ type: "spring", stiffness: 180, damping: 26 }}
        />
      </div>
    </div>
  );
}
