import { useEffect, useRef, useState } from "react";
import { ArrowUp, MousePointer2, RotateCw, Square, X } from "lucide-react";
import { agentPath, api } from "./api";
import { ComputerControl } from "./computerControl";
import { stopComputerTask, workLabel, type ComputerWork } from "./computerWork";
import { vncUrl } from "./presence";
import { randomUuid } from "./uuid";

type Connection = "connecting" | "connected" | "disconnected";
const LABELS: Record<Connection, string> = {
  connecting: "Connecting…",
  connected: "Live",
  disconnected: "Disconnected",
};
const WIDTH_KEY = "kadan.computer.width";
/** The chat keeps at least this much room beside the computer. */
const MIN_CHAT = 360;
const MIN_PANEL = 320;
const WORK_POLL_MS = 3000;

function savedWidth() {
  const value = Number(localStorage.getItem(WIDTH_KEY));
  return Number.isFinite(value) && value >= MIN_PANEL ? value : null;
}

interface Props {
  agentKey: string;
  agentName: string;
  onClose: () => void;
}

/** A live VNC view of the agent's computer beside the chat, relayed through Kadan's API. */
export default function ComputerViewer({ agentKey, agentName, onClose }: Props) {
  const panel = useRef<HTMLElement>(null);
  const screen = useRef<HTMLDivElement>(null);
  const viewer = useRef<ComputerControl | null>(null);
  const [connection, setConnection] = useState<Connection>("connecting");
  const [control, setControl] = useState(false);
  const [attempt, setAttempt] = useState(0);
  const [width, setWidth] = useState<number | null>(savedWidth);
  const [work, setWork] = useState<ComputerWork | null>(null);
  const [draft, setDraft] = useState("");
  const [sending, setSending] = useState(false);
  const [error, setError] = useState("");
  const workPath = `${agentPath(agentKey)}/computer`;
  const status = workLabel(work);

  async function refreshWork() {
    try {
      setWork(await api<ComputerWork>(workPath));
    } catch {
      setWork(null);
    }
  }
  useEffect(() => {
    void refreshWork();
    const timer = setInterval(() => void refreshWork(), WORK_POLL_MS);
    return () => clearInterval(timer);
  }, [workPath]);

  async function sendRequest(event: React.FormEvent) {
    event.preventDefault();
    const content = draft.trim();
    if (!content || sending) return;
    setSending(true);
    try {
      await api(`${workPath}/requests`, { id: randomUuid(), content });
      setDraft("");
      void refreshWork();
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "The request could not be sent.");
    } finally {
      setSending(false);
    }
  }
  async function stopTask() {
    if (!work?.task) return;
    try {
      await stopComputerTask(workPath, work.task.id);
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "The task could not be stopped.");
    }
    void refreshWork();
  }

  useEffect(() => {
    const target = screen.current;
    if (!target) return;
    let closed = false;
    setConnection("connecting");
    // Loaded on demand: noVNC is large and only this view needs it.
    import("@novnc/novnc").then(({ default: Client }) => {
      if (closed) return;
      const client = new Client(target, vncUrl(location));
      const computer = new ComputerControl(client);
      client.scaleViewport = true;
      client.background = "transparent";
      client.addEventListener("connect", () => setConnection("connected"));
      client.addEventListener("disconnect", () => {
        if (viewer.current === computer) setConnection("disconnected");
      });
      viewer.current = computer;
    }).catch(() => {
      if (!closed) setConnection("disconnected");
    });
    return () => {
      closed = true;
      const computer = viewer.current;
      viewer.current = null;
      computer?.disconnect();
    };
  }, [attempt]);
  useEffect(() => {
    viewer.current?.setControl(control);
  }, [control, connection]);

  function resize(event: React.PointerEvent<HTMLDivElement>) {
    const container = panel.current?.parentElement;
    if (!container) return;
    event.preventDefault();
    const handle = event.currentTarget;
    handle.setPointerCapture(event.pointerId);
    const right = container.getBoundingClientRect().right;
    const max = container.clientWidth - MIN_CHAT;
    let next = width;
    const move = (moved: PointerEvent) => {
      next = Math.round(Math.min(max, Math.max(MIN_PANEL, right - moved.clientX)));
      setWidth(next);
    };
    const end = () => {
      handle.removeEventListener("pointermove", move);
      handle.removeEventListener("pointerup", end);
      handle.removeEventListener("pointercancel", end);
      if (next) localStorage.setItem(WIDTH_KEY, String(next));
    };
    handle.addEventListener("pointermove", move);
    handle.addEventListener("pointerup", end);
    handle.addEventListener("pointercancel", end);
  }
  function nudge(event: React.KeyboardEvent<HTMLDivElement>) {
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
    event.preventDefault();
    const container = panel.current?.parentElement;
    const current = panel.current?.getBoundingClientRect().width ?? MIN_PANEL;
    const max = (container?.clientWidth ?? current) - MIN_CHAT;
    const next = Math.min(max, Math.max(MIN_PANEL, current + (event.key === "ArrowLeft" ? 40 : -40)));
    setWidth(next);
    localStorage.setItem(WIDTH_KEY, String(next));
  }

  return (
    <aside ref={panel} className="computer-panel" aria-labelledby="computer-panel-title"
      style={width ? { width: `min(${width}px, calc(100% - ${MIN_CHAT}px))` } : undefined}
      onKeyDown={(event) => {
        if (event.key === "Escape" && !control) onClose();
      }}>
      <div className="computer-resize" role="separator" aria-orientation="vertical"
        aria-label="Resize computer view" tabIndex={0} onPointerDown={resize} onKeyDown={nudge} />
      <header className="computer-panel-head">
        <div>
          <h2 id="computer-panel-title">{agentName}’s computer</h2>
          <p className={`computer-connection ${connection}`} role="status">
            <i aria-hidden="true" />
            {LABELS[connection]}
            {connection === "connected" && (control ? " · You have control" : " · Watching")}
          </p>
        </div>
        <div className="computer-panel-actions">
          <span className={`computer-work${work?.task ? " busy" : status.startsWith("Failed") ? " failed" : ""}`}
            role="status" title={status}>{status}</span>
          {work?.task && (
            <button className="icon" aria-label="Stop the current task" title="Stop the current task"
              onClick={() => void stopTask()}><Square size={14} /></button>
          )}
          {connection === "disconnected" ? (
            <button onClick={() => setAttempt((count) => count + 1)}>
              <RotateCw size={16} />
              Reconnect
            </button>
          ) : (
            <button className={control ? "active" : ""} aria-pressed={control}
              disabled={connection !== "connected"} onClick={() => setControl(!control)}
              title={control ? "Return to watching" : "Use the mouse and keyboard on this computer"}>
              <MousePointer2 size={16} />
              {control ? "Stop controlling" : "Take control"}
            </button>
          )}
          <button className="icon" aria-label="Close computer" onClick={onClose}><X size={18} /></button>
        </div>
      </header>
      <div className="computer-screen" ref={screen} />
      <form className="computer-request" onSubmit={sendRequest}>
        <input value={draft} onChange={(event) => { setDraft(event.target.value); setError(""); }}
          placeholder={work?.task ? "Steer the current task" : `Ask ${agentName}’s computer to do something`}
          aria-label="Request for the computer" maxLength={10000}
          onKeyDown={(event) => event.stopPropagation()} />
        <button className="icon" type="submit" aria-label="Send request" disabled={!draft.trim() || sending}>
          <ArrowUp size={16} />
        </button>
        {error && <p className="computer-request-error" role="alert">{error}</p>}
      </form>
    </aside>
  );
}
