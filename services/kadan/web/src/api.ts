import type { ChatEvent, Resource } from "./types.ts";

export class ApiError extends Error {
  readonly status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

export async function api<T>(
  path: string,
  body?: unknown,
  method = "POST",
): Promise<T> {
  const response = await fetch(
    `/api${path}`,
    body === undefined
      ? {}
      : body instanceof FormData
        ? { method, body }
        : {
            method,
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
          },
  );
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new ApiError(
      typeof error.detail === "string"
        ? error.detail
        : `Request failed (${response.status}).`,
      response.status,
    );
  }
  return response.json();
}

export const agentPath = (key: string) => `/agents/${encodeURIComponent(key)}`;

export async function allResources(
  key: string,
  kind?: string,
  q = "",
): Promise<Resource[]> {
  const rows: Resource[] = [];
  for (let offset = 0; ; offset += 200) {
    const params = new URLSearchParams({
      limit: "200",
      offset: String(offset),
      q,
    });
    if (kind) params.set("kind", kind);
    const page = await api<Resource[]>(`${agentPath(key)}/resources?${params}`);
    rows.push(...page);
    if (page.length < 200) return rows;
  }
}

export async function readEvents(
  stream: ReadableStream<Uint8Array>,
  onEvent: (event: ChatEvent) => void,
) {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = "",
    finished = false;
  try {
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value, { stream: !done });
      let match: RegExpExecArray | null;
      while ((match = /\r?\n\r?\n/.exec(buffer))) {
        const frame = buffer.slice(0, match.index);
        buffer = buffer.slice(match.index + match[0].length);
        const lines = frame.split(/\r?\n/);
        const event = lines
          .find((line) => line.startsWith("event:"))
          ?.slice(6)
          .trim();
        const payload = lines
          .filter((line) => line.startsWith("data:"))
          .map((line) => line.slice(5).trimStart())
          .join("\n");
        if (!event || !payload) continue;
        if (
          ![
            "user_saved",
            "status",
            "delta",
            "done",
            "error",
          ].includes(event)
        )
          continue;
        const item = { event, data: JSON.parse(payload) } as ChatEvent;
        onEvent(item);
        if (item.event === "error") throw new Error(item.data.message);
        if (item.event === "done") finished = true;
      }
      if (done) break;
    }
    if (!finished)
      throw new Error(
        "The response was interrupted. Reload to check what was saved.",
      );
  } finally {
    reader.releaseLock();
  }
}

export async function streamChat(
  path: string,
  body: unknown,
  signal: AbortSignal,
  onEvent: (event: ChatEvent) => void,
) {
  const response = await fetch(`/api${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(
      typeof error.detail === "string"
        ? error.detail
        : `Chat unavailable (${response.status}).`,
    );
  }
  if (!response.body)
    throw new Error("The browser could not open the response stream.");
  await readEvents(response.body, onEvent);
}
