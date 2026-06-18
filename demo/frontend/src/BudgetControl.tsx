import type { Budget } from "./api";

const OPTIONS: Budget[] = [400, 800, 1200];

export function BudgetControl({
  budget,
  onChange,
  disabled,
}: {
  budget: Budget;
  onChange: (b: Budget) => void;
  disabled: boolean;
}) {
  return (
    <div className="flex items-center gap-1.5">
      <span className="text-[10px] uppercase tracking-wider text-ide-textFaint">
        budget
      </span>
      <div
        className="flex overflow-hidden rounded-md border border-ide-border"
        role="radiogroup"
        aria-label="Token budget"
      >
        {OPTIONS.map((o) => {
          const active = o === budget;
          return (
            <button
              key={o}
              role="radio"
              aria-checked={active}
              disabled={disabled}
              onClick={() => !active && onChange(o)}
              className={`px-2.5 py-1 text-xs font-mono transition ${
                active
                  ? "bg-accent text-white"
                  : "bg-ide-card text-ide-textDim hover:text-ide-text disabled:opacity-40"
              }`}
            >
              {o}
            </button>
          );
        })}
      </div>
    </div>
  );
}
