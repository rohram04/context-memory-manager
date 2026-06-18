import { useCallback, useEffect, useRef, useState } from "react";
import * as Tooltip from "@radix-ui/react-tooltip";
import { Panel, PanelGroup, PanelResizeHandle } from "react-resizable-panels";
import {
  api,
  streamChat,
  streamScript,
  type Budget,
  type ContextSnapshot,
  type LtEvent,
  type Message,
  type Mode,
  type ScriptInfo,
  type SessionInfo,
  type StreamHandle,
  type StreamHandlers,
  type Timeline,
} from "./api";
import {
  fixtureScripts,
  fixtureSession,
  fixtureTimeline,
  USE_FIXTURES,
} from "./fixtures";
import { ChatPanel } from "./ChatPanel";
import { Inspector } from "./Inspector";
import { ScriptBar } from "./ScriptBar";
import { BudgetControl } from "./BudgetControl";
import { ModeControl } from "./ModeControl";

function emptyTimeline(): Timeline {
  return { messages: [], context_snapshots: [], lt_events: [] };
}

export default function App() {
  const [session, setSession] = useState<SessionInfo | null>(null);
  const [budget, setBudget] = useState<Budget>(400);
  const [mode, setMode] = useState<Mode>("algorithmic");
  const [timeline, setTimeline] = useState<Timeline>(emptyTimeline());
  const [scripts, setScripts] = useState<ScriptInfo[]>([]);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [sending, setSending] = useState(false);
  const [runningScript, setRunningScript] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [usingFixtures, setUsingFixtures] = useState(false);
  const [booting, setBooting] = useState(true);

  // Live-streaming state. `pendingUser` is the optimistic user bubble shown the
  // instant a message is sent, before the server commits the `user` snapshot.
  // `streamingText` is the in-progress assistant bubble that fills token by
  // token. `aborter` holds the active stream handle for the Stop control.
  const [pendingUser, setPendingUser] = useState<string | null>(null);
  const [streamingText, setStreamingText] = useState<string>("");
  const [streaming, setStreaming] = useState(false);
  const aborter = useRef<StreamHandle | null>(null);

  const loadFixtures = useCallback(() => {
    setSession(fixtureSession);
    setTimeline(fixtureTimeline);
    setScripts(fixtureScripts);
    setSelectedIndex(fixtureTimeline.messages.length - 1);
    setUsingFixtures(true);
  }, []);

  // boot: create session, load timeline + scripts. Fall back to fixtures on
  // failure (or immediately when VITE_USE_FIXTURES is set).
  const boot = useCallback(
    async (b: Budget, m: Mode) => {
      setBooting(true);
      setError(null);
      if (USE_FIXTURES) {
        loadFixtures();
        setBooting(false);
        return;
      }
      try {
        const s = await api.createSession(b, m);
        setSession(s);
        const [tl, sc] = await Promise.all([api.timeline(), api.scripts()]);
        setTimeline(tl);
        setScripts(sc);
        setSelectedIndex(Math.max(0, tl.messages.length - 1));
        setUsingFixtures(false);
      } catch (e) {
        // backend not running → offline fixtures so the UI is still usable
        console.warn("Backend unavailable, falling back to fixtures:", e);
        loadFixtures();
      } finally {
        setBooting(false);
      }
    },
    [loadFixtures]
  );

  useEffect(() => {
    void boot(budget, mode);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Abort any in-flight stream on unmount.
  useEffect(() => () => aborter.current?.abort(), []);

  // Recreate the session with the given budget+mode, resetting the timeline and
  // selection. Both the budget and mode selectors funnel through here so they
  // stay consistent. Updates the lifted budget/mode state first, then reboots.
  const recreateSession = useCallback(
    async (b: Budget, m: Mode) => {
      // Cancel any active stream before tearing down the session.
      aborter.current?.abort();
      aborter.current = null;
      setSending(false);
      setRunningScript(null);
      setStreaming(false);
      setStreamingText("");
      setPendingUser(null);
      setBudget(b);
      setMode(m);
      if (usingFixtures) {
        setSession((s) => (s ? { ...s, budget: b, mode: m } : s));
        return;
      }
      setError(null);
      setTimeline(emptyTimeline());
      setSelectedIndex(0);
      try {
        await api.deleteSession();
      } catch {
        /* ignore — may not exist */
      }
      await boot(b, m);
    },
    [boot, usingFixtures]
  );

  // Commit one snapshot + its LT delta to the timeline, append the message,
  // and point the inspector at the freshly committed turn so it updates live.
  const commitSnapshot = useCallback(
    (snapshot: ContextSnapshot, ltEvent: LtEvent) => {
      setTimeline((prev) => {
        const msg: Message = {
          index: prev.messages.length,
          role: snapshot.role,
          text: snapshot.text,
        };
        const next: Timeline = {
          messages: [...prev.messages, msg],
          context_snapshots: [...prev.context_snapshots, snapshot],
          lt_events: [...prev.lt_events, ltEvent],
        };
        setSelectedIndex(next.messages.length - 1);
        return next;
      });
    },
    []
  );

  // Shared event handlers for both typed messages and scripts. For scripts the
  // server streams every turn, firing onUser/onToken/onAssistant per turn —
  // each turn commits live through the same path.
  const makeHandlers = useCallback(
    (onFinish: () => void): StreamHandlers => ({
      onUser: (snapshot, ltEvent) => {
        setPendingUser(null);
        commitSnapshot(snapshot, ltEvent);
      },
      onToken: (delta) => {
        setStreamingText((prev) => prev + delta);
      },
      onAssistant: (snapshot, ltEvent) => {
        setStreamingText("");
        commitSnapshot(snapshot, ltEvent);
      },
      onDone: () => {
        onFinish();
      },
      onError: (detail) => {
        setError(detail);
        onFinish();
      },
    }),
    [commitSnapshot]
  );

  const finishSend = useCallback(() => {
    setSending(false);
    setStreaming(false);
    setStreamingText("");
    setPendingUser(null);
    aborter.current = null;
  }, []);

  const finishScript = useCallback(() => {
    setRunningScript(null);
    setStreaming(false);
    setStreamingText("");
    setPendingUser(null);
    aborter.current = null;
  }, []);

  function handleSend(text: string) {
    if (usingFixtures) {
      setError("Fixtures mode — start the backend to chat for real.");
      return;
    }
    if (streaming) return;
    setSending(true);
    setStreaming(true);
    setError(null);
    setPendingUser(text); // optimistic user bubble, instant
    setStreamingText("");
    aborter.current = streamChat(text, makeHandlers(finishSend));
  }

  function handleRunScript(name: string) {
    if (usingFixtures) {
      setError("Fixtures mode — start the backend to run scripts for real.");
      return;
    }
    if (streaming) return;
    setRunningScript(name);
    setStreaming(true);
    setError(null);
    setStreamingText("");
    aborter.current = streamScript(name, makeHandlers(finishScript));
  }

  // Stop: abort the in-flight stream and finalize gracefully — commit nothing
  // further, drop the partial assistant text and optimistic user bubble.
  const handleStop = useCallback(() => {
    aborter.current?.abort();
    aborter.current = null;
    setSending(false);
    setRunningScript(null);
    setStreaming(false);
    setStreamingText("");
    setPendingUser(null);
  }, []);

  const handleBudgetChange = (b: Budget) => recreateSession(b, mode);
  const handleModeChange = (m: Mode) => recreateSession(budget, m);

  const snapshots = timeline.context_snapshots;
  const idx = Math.min(selectedIndex, Math.max(0, snapshots.length - 1));
  const snapshot = snapshots[idx];
  const prevSnapshot = idx > 0 ? snapshots[idx - 1] : undefined;
  const atCap =
    !!session?.limits &&
    timeline.messages.length >= session.limits.max_messages;

  return (
    <Tooltip.Provider>
      <div className="flex h-screen flex-col overflow-hidden bg-ide-bg text-ide-text">
        {/* top bar */}
        <header className="flex shrink-0 items-center justify-between gap-4 border-b border-ide-border bg-ide-panel px-4 py-2">
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2">
              <span className="text-sm font-semibold tracking-tight">
                Memory Manager
              </span>
              <span className="rounded bg-gradient-to-r from-accent/30 to-fuchsia-500/30 px-1.5 py-0.5 text-[10px] font-medium text-accent">
                Lifecycle Inspector
              </span>
            </div>
            {usingFixtures && (
              <span className="rounded border border-amber-500/40 bg-amber-500/10 px-1.5 py-0.5 text-[10px] font-medium text-amber-300">
                fixtures mode (no backend)
              </span>
            )}
          </div>
          <div className="flex items-center gap-4">
            {session && (
              <span className="hidden font-mono text-[11px] text-ide-textFaint sm:inline">
                {session.model} · {session.mode}
              </span>
            )}
            <ModeControl
              mode={mode}
              onChange={handleModeChange}
              disabled={booting}
            />
            <span className="h-4 w-px bg-ide-border" />
            <BudgetControl
              budget={budget}
              onChange={handleBudgetChange}
              disabled={booting}
            />
          </div>
        </header>

        {/* script bar / status */}
        <div className="flex shrink-0 flex-wrap items-center justify-between gap-2 border-b border-ide-border bg-ide-panel/60 px-4 py-1.5">
          <ScriptBar
            scripts={scripts}
            onRun={handleRunScript}
            running={runningScript}
            disabled={atCap || usingFixtures || streaming}
            streaming={streaming}
            onStop={handleStop}
          />
          <div className="flex items-center gap-3 text-[11px]">
            {atCap && (
              <span className="text-amber-300">message cap reached</span>
            )}
            {error && (
              <span className="max-w-[420px] truncate text-diff-removed" title={error}>
                {error}
              </span>
            )}
            <span className="text-ide-textFaint">
              {timeline.messages.length}
              {session?.limits ? `/${session.limits.max_messages}` : ""} msgs
            </span>
          </div>
        </div>

        {/* main split: left = inspector, right = chat */}
        <div className="min-h-0 flex-1">
          {booting ? (
            <div className="flex h-full items-center justify-center text-sm text-ide-textFaint">
              Booting session…
            </div>
          ) : (
            <PanelGroup direction="horizontal">
              <Panel defaultSize={58} minSize={30}>
                <Inspector
                  snapshot={snapshot}
                  prevSnapshot={prevSnapshot}
                  ltEvents={timeline.lt_events}
                  selectedIndex={idx}
                />
              </Panel>
              <PanelResizeHandle className="w-1.5 bg-ide-border transition-colors hover:bg-accent/50" />
              <Panel defaultSize={42} minSize={26}>
                <ChatPanel
                  messages={timeline.messages}
                  selectedIndex={idx}
                  onSelect={setSelectedIndex}
                  onSend={handleSend}
                  onStop={handleStop}
                  sending={sending}
                  streaming={streaming}
                  pendingUser={pendingUser}
                  streamingText={streamingText}
                  disabled={atCap || usingFixtures}
                  maxInputChars={session?.limits?.max_input_chars ?? 2000}
                />
              </Panel>
            </PanelGroup>
          )}
        </div>
      </div>
    </Tooltip.Provider>
  );
}
