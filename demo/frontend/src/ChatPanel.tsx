import { useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import type { Message } from "./api";
import { TypingIndicator } from "./TypingIndicator";

export function ChatPanel({
  messages,
  selectedIndex,
  onSelect,
  onSend,
  onStop,
  sending,
  streaming,
  pendingUser,
  streamingText,
  disabled,
  maxInputChars,
}: {
  messages: Message[];
  selectedIndex: number;
  onSelect: (i: number) => void;
  onSend: (text: string) => void;
  onStop: () => void;
  sending: boolean;
  streaming: boolean;
  pendingUser: string | null;
  streamingText: string;
  disabled: boolean;
  maxInputChars: number;
}) {
  const [text, setText] = useState("");
  const listRef = useRef<HTMLDivElement>(null);

  // Live assistant bubble is shown whenever a stream is in flight; before the
  // first token arrives it falls back to the typing indicator.
  const showTyping = streaming && streamingText.length === 0;
  const showStreamingBubble = streaming && streamingText.length > 0;

  // autoscroll to bottom on new messages and while streaming
  useEffect(() => {
    listRef.current?.scrollTo({
      top: listRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages.length, pendingUser, streamingText, showTyping]);

  // Arrow Up/Down move selection (when not typing in the textarea)
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === "TEXTAREA" || tag === "INPUT") return;
      if (e.key === "ArrowUp") {
        e.preventDefault();
        onSelect(Math.max(0, selectedIndex - 1));
      } else if (e.key === "ArrowDown") {
        e.preventDefault();
        onSelect(Math.min(messages.length - 1, selectedIndex + 1));
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [selectedIndex, messages.length, onSelect]);

  const inputDisabled = disabled || sending || streaming;

  function submit() {
    const t = text.trim();
    if (!t || inputDisabled) return;
    onSend(t);
    setText("");
  }

  return (
    <div className="flex h-full min-h-0 flex-col bg-ide-bg">
      <div ref={listRef} className="min-h-0 flex-1 overflow-y-auto px-3 py-4">
        <div className="flex flex-col gap-2">
          {messages.length === 0 && !pendingUser && !streaming && (
            <div className="py-10 text-center text-sm text-ide-textFaint">
              Start chatting, or run a script below. Click any message to inspect
              the memory state at that point.
            </div>
          )}
          {messages.map((m) => {
            const selected = m.index === selectedIndex;
            const isUser = m.role === "user";
            return (
              <motion.button
                key={m.index}
                layout
                onClick={() => onSelect(m.index)}
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                className={`flex w-full ${isUser ? "justify-end" : "justify-start"}`}
              >
                <div
                  className={`max-w-[82%] rounded-2xl px-3.5 py-2 text-left text-[13px] leading-relaxed transition ${
                    isUser
                      ? "bg-accent/90 text-white"
                      : "border border-ide-border bg-ide-card text-ide-text"
                  } ${
                    selected
                      ? "ring-2 ring-accent ring-offset-2 ring-offset-ide-bg"
                      : "opacity-90 hover:opacity-100"
                  }`}
                >
                  <div className="mb-0.5 flex items-center gap-2 text-[9px] uppercase tracking-wider opacity-60">
                    <span>{m.role}</span>
                    <span>#{m.index}</span>
                  </div>
                  <span className="whitespace-pre-wrap break-words">{m.text}</span>
                </div>
              </motion.button>
            );
          })}

          {/* Optimistic user bubble: shown instantly on send, before the
              server commits the user snapshot. */}
          {pendingUser && (
            <motion.div
              layout
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              className="flex w-full justify-end"
            >
              <div className="max-w-[82%] rounded-2xl bg-accent/90 px-3.5 py-2 text-left text-[13px] leading-relaxed text-white">
                <div className="mb-0.5 flex items-center gap-2 text-[9px] uppercase tracking-wider opacity-60">
                  <span>user</span>
                  <span>·</span>
                </div>
                <span className="whitespace-pre-wrap break-words">
                  {pendingUser}
                </span>
              </div>
            </motion.div>
          )}

          {/* Live assistant bubble: typing indicator until the first token,
              then the accumulating text with a blinking cursor. */}
          {showTyping && <TypingIndicator />}
          {showStreamingBubble && (
            <motion.div
              layout
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              className="flex w-full justify-start"
            >
              <div className="max-w-[82%] rounded-2xl border border-ide-border bg-ide-card px-3.5 py-2 text-left text-[13px] leading-relaxed text-ide-text">
                <div className="mb-0.5 flex items-center gap-2 text-[9px] uppercase tracking-wider opacity-60">
                  <span>assistant</span>
                  <span>·</span>
                </div>
                <span className="whitespace-pre-wrap break-words">
                  {streamingText}
                  <span className="ml-0.5 inline-block w-[2px] animate-pulse bg-ide-textDim align-middle text-transparent">
                    |
                  </span>
                </span>
              </div>
            </motion.div>
          )}
        </div>
      </div>

      <div className="border-t border-ide-border bg-ide-panel/70 p-3">
        <div className="flex items-end gap-2">
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value.slice(0, maxInputChars))}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                submit();
              }
            }}
            placeholder={
              disabled
                ? "Message cap reached"
                : streaming
                  ? "Streaming…"
                  : "Message the agent… (Enter to send)"
            }
            disabled={inputDisabled}
            rows={1}
            className="max-h-32 min-h-[40px] flex-1 resize-none rounded-lg border border-ide-border bg-ide-bg px-3 py-2 text-[13px] text-ide-text placeholder:text-ide-textFaint focus:border-accent focus:outline-none disabled:opacity-50"
          />
          {streaming ? (
            <button
              onClick={onStop}
              className="h-[40px] rounded-lg border border-diff-removed/50 bg-diff-removed/10 px-4 text-sm font-medium text-diff-removed transition hover:bg-diff-removed/20"
            >
              Stop
            </button>
          ) : (
            <button
              onClick={submit}
              disabled={inputDisabled || !text.trim()}
              className="h-[40px] rounded-lg bg-accent px-4 text-sm font-medium text-white transition hover:bg-accent-glow disabled:opacity-40"
            >
              Send
            </button>
          )}
        </div>
        <div className="mt-1 text-right text-[10px] text-ide-textFaint">
          {text.length}/{maxInputChars}
        </div>
      </div>
    </div>
  );
}
