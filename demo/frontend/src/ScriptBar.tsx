import { InfoTip } from "./components/Primitives";
import type { ScriptInfo } from "./api";

export function ScriptBar({
  scripts,
  onRun,
  running,
  disabled,
  streaming,
  onStop,
}: {
  scripts: ScriptInfo[];
  onRun: (name: string) => void;
  running: string | null;
  disabled: boolean;
  streaming: boolean;
  onStop: () => void;
}) {
  if (scripts.length === 0) return null;
  return (
    <div className="flex flex-wrap items-center gap-2">
      <span className="text-[10px] uppercase tracking-wider text-ide-textFaint">
        scripts
      </span>
      {scripts.map((s) => (
        <InfoTip key={s.name} label={s.description}>
          <button
            onClick={() => onRun(s.name)}
            disabled={disabled || running !== null}
            className="rounded-md border border-ide-border bg-ide-card px-2.5 py-1 text-xs text-ide-textDim transition hover:border-accent hover:text-accent disabled:opacity-40"
          >
            {running === s.name ? "running…" : s.name}
          </button>
        </InfoTip>
      ))}
      {streaming && (
        <button
          onClick={onStop}
          className="rounded-md border border-diff-removed/50 bg-diff-removed/10 px-2.5 py-1 text-xs font-medium text-diff-removed transition hover:border-diff-removed hover:bg-diff-removed/20"
        >
          ◼ Stop
        </button>
      )}
    </div>
  );
}
