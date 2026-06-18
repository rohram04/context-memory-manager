// Typed fetch wrappers + shared types.
// All requests are relative ("/api/...") and send credentials so the signed
// session cookie set by the backend is included. In dev, vite proxies these to
// http://localhost:8000 (see vite.config.ts) keeping us same-origin.

export type Role = "user" | "assistant";

export interface Block {
  id: string;
  content: string;
  tier: "full" | "stub";
  novelty: number;
  decay: number;
  fidelity: number;
  token_cost: number;
  compression_count: number;
  access_count: number;
  pointer_to_lt_id: string | null;
}

export interface ContextSnapshot {
  index: number;
  role: Role;
  text: string;
  budget_used: number;
  budget_max: number;
  budget_pressure: string;
  memory_prompt_text: string;
  blocks: Block[];
  latency_ms: number | null;
}

export interface LtView {
  id: string;
  content: string;
  novelty: number;
  fidelity: number;
  compression_count: number;
  access_count: number;
  decay: number;
  is_reconstructed: boolean;
}

export interface LtEvent {
  added: LtView[];
  updated: LtView[];
  accessed: string[];
}

export interface Message {
  index: number;
  role: Role;
  text: string;
}

export interface Limits {
  max_messages: number;
  max_input_chars: number;
}

export interface SessionInfo {
  budget: number;
  mode: string;
  model: string;
  message_count?: number;
  limits: Limits;
}

export interface Timeline {
  messages: Message[];
  context_snapshots: ContextSnapshot[];
  lt_events: LtEvent[];
}

export interface ChatResponse {
  messages: Message[];
  context_snapshots: ContextSnapshot[];
  lt_events: LtEvent[];
  reply: string;
  latency_ms: number;
}

export interface ScriptInfo {
  name: string;
  description: string;
}

export type Budget = 400 | 800 | 1200;
export type Mode = "algorithmic" | "llm";

const JSON_HEADERS = { "Content-Type": "application/json" };

async function req<T>(
  path: string,
  init?: Omit<RequestInit, "body"> & { body?: unknown }
): Promise<T> {
  const { body, ...rest } = init ?? {};
  const res = await fetch(path, {
    credentials: "include",
    headers: body !== undefined ? JSON_HEADERS : undefined,
    body: body !== undefined ? JSON.stringify(body) : undefined,
    ...rest,
  });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const j = await res.json();
      if (j && typeof j === "object" && "detail" in j) detail = String(j.detail);
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }
  // DELETE may return 204
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

// --- Streaming (NDJSON over a POST body) ---------------------------------
// The backend emits one JSON object per line:
//   {"type":"user",      snapshot, lt_event}
//   {"type":"token",     delta}
//   {"type":"assistant", snapshot, lt_event, text}
//   {"type":"done"}
//   {"type":"error",     detail}
// We read response.body as a stream, decode incrementally, buffer + split on
// "\n", JSON.parse each complete line, and dispatch to handlers. The returned
// abort() cancels the in-flight request via AbortController.

export interface StreamHandlers {
  onUser?: (snapshot: ContextSnapshot, ltEvent: LtEvent) => void;
  onToken?: (delta: string) => void;
  onAssistant?: (
    snapshot: ContextSnapshot,
    ltEvent: LtEvent,
    text: string
  ) => void;
  onDone?: () => void;
  onError?: (detail: string) => void;
}

interface StreamLine {
  type: "user" | "token" | "assistant" | "done" | "error";
  snapshot?: ContextSnapshot;
  lt_event?: LtEvent;
  delta?: string;
  text?: string;
  detail?: string;
}

function dispatchLine(line: string, h: StreamHandlers) {
  const trimmed = line.trim();
  if (!trimmed) return;
  let evt: StreamLine;
  try {
    evt = JSON.parse(trimmed) as StreamLine;
  } catch {
    // Ignore malformed lines rather than tearing down the whole stream.
    return;
  }
  switch (evt.type) {
    case "user":
      if (evt.snapshot) h.onUser?.(evt.snapshot, evt.lt_event ?? EMPTY_LT);
      break;
    case "token":
      if (typeof evt.delta === "string") h.onToken?.(evt.delta);
      break;
    case "assistant":
      if (evt.snapshot)
        h.onAssistant?.(evt.snapshot, evt.lt_event ?? EMPTY_LT, evt.text ?? "");
      break;
    case "done":
      h.onDone?.();
      break;
    case "error":
      h.onError?.(evt.detail ?? "stream error");
      break;
  }
}

const EMPTY_LT: LtEvent = { added: [], updated: [], accessed: [] };

async function streamNdjson(
  path: string,
  body: unknown,
  handlers: StreamHandlers,
  signal: AbortSignal
): Promise<void> {
  const res = await fetch(path, {
    method: "POST",
    credentials: "include",
    headers: JSON_HEADERS,
    body: JSON.stringify(body),
    signal,
  });

  if (!res.ok || !res.body) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const j = await res.json();
      if (j && typeof j === "object" && "detail" in j) detail = String(j.detail);
    } catch {
      /* ignore */
    }
    handlers.onError?.(detail);
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  // eslint-disable-next-line no-constant-condition
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let nl: number;
    while ((nl = buffer.indexOf("\n")) >= 0) {
      const line = buffer.slice(0, nl);
      buffer = buffer.slice(nl + 1);
      dispatchLine(line, handlers);
    }
  }
  // flush any trailing decoder state + a final partial line without a newline
  buffer += decoder.decode();
  if (buffer.trim()) dispatchLine(buffer, handlers);
}

export interface StreamHandle {
  abort: () => void;
}

function runStream(
  path: string,
  body: unknown,
  handlers: StreamHandlers
): StreamHandle {
  const controller = new AbortController();
  streamNdjson(path, body, handlers, controller.signal).catch((e: unknown) => {
    // An AbortError is an intentional Stop — finalize quietly.
    if (e instanceof DOMException && e.name === "AbortError") return;
    if (controller.signal.aborted) return;
    handlers.onError?.(e instanceof Error ? e.message : String(e));
  });
  return { abort: () => controller.abort() };
}

export function streamChat(
  message: string,
  handlers: StreamHandlers
): StreamHandle {
  return runStream("/api/chat/stream", { message }, handlers);
}

export function streamScript(
  name: string,
  handlers: StreamHandlers
): StreamHandle {
  return runStream(
    `/api/script/${encodeURIComponent(name)}/stream`,
    {},
    handlers
  );
}

export const api = {
  createSession: (budget?: Budget, mode?: Mode) =>
    req<SessionInfo>("/api/session", {
      method: "POST",
      body: { budget, mode },
    }),
  getSession: () => req<SessionInfo>("/api/session"),
  deleteSession: () => req<void>("/api/session", { method: "DELETE" }),
  chat: (message: string) =>
    req<ChatResponse>("/api/chat", { method: "POST", body: { message } }),
  timeline: () => req<Timeline>("/api/timeline"),
  scripts: () => req<ScriptInfo[]>("/api/scripts"),
  runScript: (name: string) =>
    req<ChatResponse>(`/api/script/${encodeURIComponent(name)}`, {
      method: "POST",
    }),
};
