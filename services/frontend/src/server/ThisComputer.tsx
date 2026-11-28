import { useCallback, useEffect, useState } from "react";
import { FolderOpen, Play, RefreshCw, Square } from "lucide-react";
import {
  getHostLogs,
  getHostStatus,
  pickDesktopFolder,
  restartHostService,
  saveHostSettings,
  startHost,
  stopHost,
  type HostLogEntry,
  type HostService,
  type HostSettings,
  type HostStatus,
} from "../kadan/desktop";
import { StatusPill } from "./common";

const PHASE: Record<HostService["phase"], { tone: "ok" | "busy" | "muted" | "error"; label: string }> = {
  running: { tone: "ok", label: "Running" },
  starting: { tone: "busy", label: "Starting" },
  stopped: { tone: "muted", label: "Stopped" },
  error: { tone: "error", label: "Problem" },
};

/** One line for the whole computer: running, starting, stopped or in trouble. */
export function hostSummary(status: HostStatus["status"]): { tone: "ok" | "busy" | "muted" | "error"; label: string } {
  const phases = status.services.map((service) => service.phase);
  if (phases.includes("error")) return { tone: "error", label: "Something needs attention" };
  if (phases.includes("starting")) return { tone: "busy", label: "Starting…" };
  if (status.running && phases.every((phase) => phase === "running")) return { tone: "ok", label: "Running on this computer" };
  if (!status.running) return { tone: "muted", label: "Stopped" };
  return { tone: "busy", label: "Starting…" };
}

function Logs() {
  const [service, setService] = useState("");
  const [logs, setLogs] = useState<HostLogEntry[] | null>(null);
  const [error, setError] = useState("");
  const load = useCallback(() => {
    getHostLogs(service || undefined).then((rows) => { setLogs(rows); setError(""); }, (failure) => setError((failure as Error).message));
  }, [service]);
  useEffect(() => {
    load();
    const timer = setInterval(load, 4000);
    return () => clearInterval(timer);
  }, [load]);
  return (
    <details className="srv-card srv-advanced">
      <summary>
        <span>
          <strong>Logs</strong>
          <small>What Kadan’s services have printed recently. Useful when something goes wrong.</small>
        </span>
      </summary>
      <div className="srv-advanced-body">
        <div className="srv-filters" role="group" aria-label="Show logs from">
          {[["", "Everything"], ["api", "Main service"], ["runner", "Agent"], ["worker", "AI models"]].map(([value, label]) => (
            <button key={value} className={service === value ? "selected" : ""} aria-pressed={service === value}
              onClick={() => setService(value)}>{label}</button>
          ))}
        </div>
        {error && <p className="error" role="alert">{error}</p>}
        <pre className="srv-logs" aria-live="off">
          {logs?.length ? logs.slice(-300).map((entry) => `${entry.service}: ${entry.line}`).join("\n") : "Nothing yet."}
        </pre>
      </div>
    </details>
  );
}

function SettingsForm({ settings, running, onSaved }: { settings: HostSettings; running: boolean; onSaved: () => void }) {
  const [draft, setDraft] = useState(settings);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const changed = JSON.stringify(draft) !== JSON.stringify(settings);
  const needsRestart = draft.server_port !== settings.server_port
    || draft.bind_all_interfaces !== settings.bind_all_interfaces
    || draft.data_dir !== settings.data_dir;

  async function save() {
    setBusy(true);
    setError("");
    setNotice("");
    try {
      await saveHostSettings(draft, running && needsRestart);
      setNotice(running && needsRestart ? "Saved. Kadan restarted with the new settings." : "Saved.");
      onSaved();
    } catch (failure) {
      setError((failure as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="srv-card srv-editor">
      <header className="srv-card-head">
        <h2>Server settings</h2>
        <p>How Kadan runs on this computer.</p>
      </header>
      <label className="srv-check">
        <input type="checkbox" checked={draft.bind_all_interfaces} disabled={busy}
          onChange={(event) => setDraft({ ...draft, bind_all_interfaces: event.target.checked })} />
        Let other devices on my network use Kadan on this computer
      </label>
      <label className="srv-check">
        <input type="checkbox" checked={draft.auto_start_server} disabled={busy}
          onChange={(event) => setDraft({ ...draft, auto_start_server: event.target.checked })} />
        Start Kadan’s services when the app opens
      </label>
      <label className="srv-check">
        <input type="checkbox" checked={draft.launch_at_login} disabled={busy}
          onChange={(event) => setDraft({ ...draft, launch_at_login: event.target.checked })} />
        Open Kadan when I sign in to this computer
      </label>
      <label>
        Where Kadan keeps its data and AI models
        <div className="srv-folder">
          <input value={draft.data_dir} disabled={busy} onChange={(event) => setDraft({ ...draft, data_dir: event.target.value })} />
          <button type="button" className="button" disabled={busy} onClick={() => {
            void pickDesktopFolder().then((picked) => picked.path && setDraft({ ...draft, data_dir: picked.path }));
          }}>
            <FolderOpen size={14} />
            Choose…
          </button>
        </div>
      </label>
      <label>
        Port
        <input type="number" min={1} max={65535} value={draft.server_port} disabled={busy}
          onChange={(event) => setDraft({ ...draft, server_port: Number(event.target.value) || 8000 })} />
      </label>
      <p className="field-hint">
        Changing the port, the folder or network access restarts Kadan’s services. Moving the folder doesn’t move what’s
        already saved.
      </p>
      {error && <p className="error" role="alert">{error}</p>}
      {notice && <p className="field-hint" role="status">{notice}</p>}
      <div className="form-actions srv-editor-actions">
        <span className="srv-spacer" />
        <button disabled={busy || !changed} onClick={() => setDraft(settings)}>Undo changes</button>
        <button className="primary" disabled={busy || !changed || !draft.data_dir.trim()} onClick={() => void save()}>
          {busy ? "Saving…" : running && needsRestart ? "Save and restart" : "Save"}
        </button>
      </div>
    </section>
  );
}

/** Kadan's services on this computer, when the Kadan app hosts them. */
export default function ThisComputer() {
  const [host, setHost] = useState<HostStatus | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [refresh, setRefresh] = useState(0);

  useEffect(() => {
    let alive = true;
    const load = () => getHostStatus().then(
      (next) => { if (alive) { setHost(next); setError(""); } },
      (failure) => { if (alive) setError((failure as Error).message); },
    );
    void load();
    const timer = setInterval(load, 3000);
    return () => { alive = false; clearInterval(timer); };
  }, [refresh]);

  async function act(work: () => Promise<unknown>) {
    setBusy(true);
    setError("");
    try {
      await work();
    } catch (failure) {
      setError((failure as Error).message);
    } finally {
      setBusy(false);
      setRefresh((value) => value + 1);
    }
  }

  if (!host) return error ? <p className="error" role="alert">{error}</p> : <p className="srv-loading">Checking this computer…</p>;

  if (host.mode === "connect") {
    return (
      <section className="srv-card">
        <header className="srv-card-head">
          <h2>This computer</h2>
          <p>
            Kadan on this computer uses another computer’s services, so nothing heavy runs here. To run Kadan here
            instead, open <strong>Settings</strong> in Kadan and choose <strong>This computer</strong> under Kadan Server.
          </p>
        </header>
      </section>
    );
  }

  const summary = hostSummary(host.status);
  const gpu = host.hardware.gpus.map((card) => card.name).join(", ");
  return (
    <>
      <section className="srv-card">
        <header className="srv-card-head srv-card-head-row">
          <div>
            <h2>This computer</h2>
            <p>{host.hardware.hostname}{gpu ? ` · ${gpu}` : " · no graphics card found, so AI models run slowly or not at all"}</p>
          </div>
          <StatusPill tone={summary.tone}>{summary.label}</StatusPill>
        </header>
        {!host.status.python_runtime_found && (
          <p className="error" role="alert">Part of Kadan is missing from this installation, so it can’t run here. Reinstall Kadan.</p>
        )}
        {error && <p className="error" role="alert">{error}</p>}
        <ul className="srv-model-list">
          {host.status.services.map((service) => (
            <li key={service.id} className="srv-model">
              <div className="srv-model-text">
                <strong>{service.label}</strong>
                <span>{service.message}</span>
              </div>
              <div className="srv-model-side srv-row-actions">
                <StatusPill tone={PHASE[service.phase].tone}>{PHASE[service.phase].label}</StatusPill>
                <button className="button" aria-label={`Restart ${service.label}`} title="Restart"
                  disabled={busy || !host.status.running} onClick={() => void act(() => restartHostService(service.id))}>
                  <RefreshCw size={14} />
                </button>
              </div>
            </li>
          ))}
        </ul>
        <div className="form-actions srv-editor-actions">
          {host.status.running ? (
            <button className="danger-button" disabled={busy} onClick={() => void act(stopHost)}>
              <Square size={14} />
              Stop Kadan’s services
            </button>
          ) : (
            <button className="primary" disabled={busy} onClick={() => void act(startHost)}>
              <Play size={14} />
              Start Kadan’s services
            </button>
          )}
          {host.status.running && (
            <button disabled={busy} onClick={() => void act(startHost)}>
              <RefreshCw size={14} />
              Restart everything
            </button>
          )}
        </div>
      </section>
      <SettingsForm key={JSON.stringify(host.settings)} settings={host.settings} running={host.status.running}
        onSaved={() => setRefresh((value) => value + 1)} />
      <Logs />
    </>
  );
}
