import { useEffect, useRef, useState } from "react";
import { MousePointer2, RotateCw, X } from "lucide-react";
import type RFB from "@novnc/novnc";
import { vncUrl } from "./presence";

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

function savedWidth() {
  const value = Number(localStorage.getItem(WIDTH_KEY));
  return Number.isFinite(value) && value >= MIN_PANEL ? value : null;
}

interface Props {
  agentName: string;
  onClose: () => void;
}

/** A live VNC view of the agent's computer beside the chat, relayed through Kadan's API. */
export default function ComputerViewer({ agentName, onClose }: Props) {
  const panel = useRef<HTMLElement>(null);
  const screen = useRef<HTMLDivElement>(null);
  const rfb = useRef<RFB | null>(null);
  const [connection, setConnection] = useState<Connection>("connecting");
  const [control, setControl] = useState(false);
  const [attempt, setAttempt] = useState(0);
  const [width, setWidth] = useState<number | null>(savedWidth);

  useEffect(() => {
    const target = screen.current;
    if (!target) return;
    let closed = false;
    setConnection("connecting");
    // Loaded on demand: noVNC is large and only this view needs it.
    import("@novnc/novnc").then(({ default: Client }) => {
      if (closed) return;
      const client = new Client(target, vncUrl(location));
      client.scaleViewport = true;
      client.viewOnly = true;
      client.background = "transparent";
      client.addEventListener("connect", () => setConnection("connected"));
      client.addEventListener("disconnect", () => {
        if (rfb.current === client) setConnection("disconnected");
      });
      rfb.current = client;
    }).catch(() => {
      if (!closed) setConnection("disconnected");
    });
    return () => {
      closed = true;
      const client = rfb.current;
      rfb.current = null;
      client?.disconnect();
    };
  }, [attempt]);
  useEffect(() => {
    if (rfb.current) rfb.current.viewOnly = !control;
    if (control) rfb.current?.focus();
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
    </aside>
  );
}
