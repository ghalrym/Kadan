import { useCallback, useEffect, useRef, useState } from "react";
import type { Agent, Status } from "../kadan/types";

/** Turn a failed response into a sentence a person can act on. */
async function failure(response: Response): Promise<Error> {
  const body = await response.json().catch(() => null);
  const detail = body?.detail;
  if (typeof detail === "string") return new Error(detail);
  if (response.status === 409) return new Error("This changed somewhere else. Reload it, then try again.");
  return new Error(`Kadan Server couldn’t finish that (error ${response.status}). Try again in a moment.`);
}

export async function getJson<T>(path: string, signal?: AbortSignal): Promise<T> {
  let response: Response;
  try {
    response = await fetch(path, { signal, cache: "no-store" });
  } catch (error) {
    if ((error as Error).name === "AbortError") throw error;
    throw new Error("Can’t reach Kadan Server. Check that it’s running, then try again.");
  }
  if (!response.ok) throw await failure(response);
  return response.json() as Promise<T>;
}

export async function sendJson<T>(path: string, method: string, body?: unknown): Promise<T> {
  let response: Response;
  try {
    response = await fetch(path, {
      method,
      headers: body === undefined ? undefined : { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch {
    throw new Error("Can’t reach Kadan Server. Check that it’s running, then try again.");
  }
  if (!response.ok) throw await failure(response);
  return response.json() as Promise<T>;
}

export const kadanPath = (key: string) => `/kadan/api/agents/${encodeURIComponent(key)}`;
export const storagePath = (key: string) => `/tsuchi/api/agents/${encodeURIComponent(key)}`;

/** Read a URL now and then every `interval` ms; `reload` reads again at once. */
export function usePoll<T>(path: string | null, interval = 5000) {
  const [state, setState] = useState<{ path: string | null; data: T | null; error: string }>({
    path: null, data: null, error: "",
  });
  const [tick, setTick] = useState(0);
  useEffect(() => {
    if (!path) return;
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    const read = async () => {
      try {
        const data = await getJson<T>(path, controller.signal);
        if (!controller.signal.aborted) setState({ path, data, error: "" });
      } catch (error) {
        if (!controller.signal.aborted) {
          setState((current) => ({ path, data: current.path === path ? current.data : null, error: (error as Error).message }));
        }
      }
      if (!controller.signal.aborted && interval > 0) timer = setTimeout(read, interval);
    };
    void read();
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, [path, interval, tick]);
  const reload = useCallback(() => setTick((value) => value + 1), []);
  const current = state.path === path;
  return { data: current ? state.data : null, error: current ? state.error : "", reload };
}

/** Run one change at a time and keep its error for the page to show. */
export function useAction() {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const running = useRef(false);
  const run = useCallback(async <T,>(work: () => Promise<T>): Promise<T | undefined> => {
    if (running.current) return undefined;
    running.current = true;
    setBusy(true);
    setError("");
    try {
      return await work();
    } catch (failed) {
      setError((failed as Error).message);
      return undefined;
    } finally {
      running.current = false;
      setBusy(false);
    }
  }, []);
  return { busy, error, setError, run };
}

export type { Agent, Status };

export function formatBytes(bytes: number): string {
  if (bytes < 1000) return `${bytes} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let value = bytes / 1000;
  let unit = 0;
  while (value >= 1000 && unit < units.length - 1) {
    value /= 1000;
    unit += 1;
  }
  return `${value.toFixed(value < 10 ? 1 : 0)} ${units[unit]}`;
}

export function formatWhen(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

/** The address other devices use to reach this server. */
export const serverAddress = () => (typeof window === "undefined" ? "" : window.location.origin);
