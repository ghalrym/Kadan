import { useCallback, useEffect, useRef, useState } from "react";
import { fetchModels, modelsDownloading, type CapabilityModel } from "../hikari/api/settings";

/** AI model rows, polled quickly while a download runs and slowly otherwise. */
export function useModels() {
  const [rows, setRows] = useState<CapabilityModel[] | null>(null);
  const [error, setError] = useState("");
  const [actionError, setActionError] = useState("");
  const [needsRefresh, setNeedsRefresh] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [refresh, setRefresh] = useState(0);
  const action = useRef<AbortController | null>(null);
  const version = useRef(0);
  const downloading = !!rows && modelsDownloading(rows);

  useEffect(() => () => action.current?.abort(), []);

  useEffect(() => {
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      const reading = version.current;
      try {
        if (!action.current) {
          const next = await fetchModels(controller.signal);
          if (!controller.signal.aborted && !action.current && reading === version.current) {
            setRows(next);
            setError("");
          }
        }
      } catch (failure) {
        if (!controller.signal.aborted && !action.current && reading === version.current) {
          setError(failure instanceof Error ? failure.message : "Couldn’t load AI models.");
        }
      }
      if (!controller.signal.aborted) timer = setTimeout(poll, downloading ? 1500 : 10000);
    };
    void poll();
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, [downloading, refresh]);

  /** Run one change; a lost response may still have started work, so ask for a refresh. */
  const run = useCallback(async (work: (signal: AbortSignal) => Promise<CapabilityModel[]>, failed: string) => {
    if (action.current) return false;
    const controller = new AbortController();
    action.current = controller;
    version.current += 1;
    setSubmitting(true);
    setActionError("");
    try {
      const next = await work(AbortSignal.any([controller.signal, AbortSignal.timeout(60000)]));
      if (controller.signal.aborted) return false;
      setRows(next);
      setNeedsRefresh(false);
      setRefresh((value) => value + 1);
      return true;
    } catch (failure) {
      if (!controller.signal.aborted) {
        setActionError(failure instanceof Error ? failure.message : failed);
        setNeedsRefresh(true);
      }
      return false;
    } finally {
      action.current = null;
      if (!controller.signal.aborted) setSubmitting(false);
    }
  }, []);

  const reload = useCallback(() => {
    void run(fetchModels, "Couldn’t load AI models.");
  }, [run]);

  return { rows, error, actionError, needsRefresh, submitting, run, reload };
}
