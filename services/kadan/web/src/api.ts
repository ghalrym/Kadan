import type { Resource } from "./types.ts";

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
