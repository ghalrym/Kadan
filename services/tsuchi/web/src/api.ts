import { useEffect, useState } from "react";
import type { Resource } from "./types";

export function agentPath(key: string) {
  return `/api/agents/${encodeURIComponent(key)}`;
}
export function assetUrl(key: string, id: string, download = false) {
  return `${agentPath(key)}/assets/${encodeURIComponent(id)}/content${download ? "?download=true" : ""}`;
}
export async function request<T>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  const response = await fetch(path, {
    ...options,
    headers:
      options.body instanceof FormData
        ? options.headers
        : { "Content-Type": "application/json", ...options.headers },
  });
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    const detail = body?.detail;
    const message =
      typeof detail === "string"
        ? detail
        : Array.isArray(detail)
          ? detail
              .map(
                (item: { loc?: string[]; msg?: string }) =>
                  `${item.loc?.slice(1).join(".") || "Request"}: ${item.msg}`,
              )
              .join("; ")
          : `Request failed (${response.status}).`;
    throw new Error(
      response.status === 409
        ? `${message} Close and reopen this item to load the current version.`
        : message,
    );
  }
  return response.json();
}
export function json(method: string, body?: unknown): RequestInit {
  return {
    method,
    ...(body === undefined ? {} : { body: JSON.stringify(body) }),
  };
}
export async function allResources(
  key: string,
  kind?: string,
): Promise<Resource[]> {
  const result: Resource[] = [];
  for (let offset = 0; ; offset += 200) {
    const page = await request<Resource[]>(
      `${agentPath(key)}/resources?limit=200&offset=${offset}${kind ? `&kind=${kind}` : ""}`,
    );
    result.push(...page);
    if (page.length < 200) return result;
  }
}
export function useLoad<T>(load: () => Promise<T>, dependencies: unknown[]) {
  const [state, setState] = useState<{
    data: T | null;
    loading: boolean;
    error: string;
  }>({ data: null, loading: true, error: "" });
  useEffect(() => {
    let active = true;
    setState({ data: null, loading: true, error: "" });
    load()
      .then((data) => {
        if (active) setState({ data, loading: false, error: "" });
      })
      .catch((error) => {
        if (active)
          setState({
            data: null,
            loading: false,
            error:
              error instanceof Error ? error.message : "Unable to load data.",
          });
      });
    return () => {
      active = false;
    };
    // The caller supplies the precise data identity and refresh dependencies.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, dependencies);
  return state;
}
export function formatBytes(value: number) {
  if (value < 1024) return `${value} B`;
  const exponent = Math.min(Math.floor(Math.log(value) / Math.log(1024)), 4);
  return `${(value / 1024 ** exponent).toFixed(exponent > 1 ? 1 : 0)} ${["B", "KB", "MB", "GB", "TB"][exponent]}`;
}
export function formatDate(value: string) {
  return new Date(value).toLocaleString(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  });
}
