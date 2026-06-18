import { useEffect, useState } from "react";
import * as Tabs from "@radix-ui/react-tabs";
import { Panel, PanelGroup, PanelResizeHandle } from "react-resizable-panels";
import type { ContextSnapshot, LtEvent } from "./api";
import { ContextTab } from "./ContextTab";
import { LongTermTab } from "./LongTermTab";

const TAB_TRIGGER =
  "px-3 py-1.5 text-xs font-medium text-ide-textDim data-[state=active]:text-ide-text data-[state=active]:border-b-2 data-[state=active]:border-accent transition-colors";

export function Inspector({
  snapshot,
  prevSnapshot,
  ltEvents,
  selectedIndex,
}: {
  snapshot: ContextSnapshot | undefined;
  prevSnapshot: ContextSnapshot | undefined;
  ltEvents: LtEvent[];
  selectedIndex: number;
}) {
  const [tab, setTab] = useState<"context" | "lt">("context");
  const [split, setSplit] = useState(false);
  const [highlightLtId, setHighlightLtId] = useState<string | null>(null);

  // "go to definition": stub → LT. The caller passes the stub's real
  // pointer_to_lt_id (the actual LT block id), so we highlight it directly —
  // no id-prefix heuristic. Switch to the LT tab unless we're already split.
  function gotoLt(ltId: string) {
    setHighlightLtId(ltId);
    if (!split) setTab("lt");
  }

  // clear transient highlight when navigating messages
  useEffect(() => {
    setHighlightLtId(null);
  }, [selectedIndex]);

  const contextPane = (
    <ContextTab
      snapshot={snapshot}
      prevSnapshot={prevSnapshot}
      onGotoLt={gotoLt}
    />
  );
  const ltPane = (
    <LongTermTab
      events={ltEvents}
      selectedIndex={selectedIndex}
      highlightLtId={highlightLtId}
    />
  );

  return (
    <div className="flex h-full min-h-0 flex-col bg-ide-panel">
      <div className="flex items-center justify-between border-b border-ide-border px-2">
        {split ? (
          <div className="flex items-center gap-4 px-1 py-1.5 text-xs font-medium">
            <span className="text-ide-text">Context</span>
            <span className="text-ide-textFaint">·</span>
            <span className="text-ide-text">Long-term</span>
          </div>
        ) : (
          <Tabs.Root
            value={tab}
            onValueChange={(v) => setTab(v as "context" | "lt")}
          >
            <Tabs.List className="flex">
              <Tabs.Trigger value="context" className={TAB_TRIGGER}>
                Context
              </Tabs.Trigger>
              <Tabs.Trigger value="lt" className={TAB_TRIGGER}>
                Long-term
              </Tabs.Trigger>
            </Tabs.List>
          </Tabs.Root>
        )}
        <button
          onClick={() => setSplit((s) => !s)}
          className={`rounded border px-2 py-0.5 text-[11px] transition ${
            split
              ? "border-accent bg-accent/15 text-accent"
              : "border-ide-border text-ide-textDim hover:border-ide-borderBright"
          }`}
          title="Show Context and Long-term side by side"
        >
          {split ? "unsplit" : "split"}
        </button>
      </div>

      <div className="min-h-0 flex-1">
        {split ? (
          <PanelGroup direction="vertical">
            <Panel defaultSize={55} minSize={20}>
              {contextPane}
            </Panel>
            <PanelResizeHandle className="h-1.5 bg-ide-border transition-colors hover:bg-accent/50" />
            <Panel defaultSize={45} minSize={20}>
              {ltPane}
            </Panel>
          </PanelGroup>
        ) : tab === "context" ? (
          contextPane
        ) : (
          ltPane
        )}
      </div>
    </div>
  );
}
