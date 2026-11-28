import { useEffect, useRef, useState } from "react";
import { CheckCircle2, Laptop, Plug, Server, Trash2, XCircle } from "lucide-react";
import {
  getDesktopServerConfig,
  saveDesktopServerConfig,
  testDesktopServerConnection,
  type DesktopServerConfig,
  type RunMode,
  type SavedServer,
  type ServerTestResult,
} from "./desktop";

/** What the sidebar shows for where Kadan runs. */
export function runLocationLabel(config: Pick<DesktopServerConfig, "mode" | "active_host">): string {
  return config.mode === "connect" ? config.active_host : "This computer";
}

interface Props {
  onConnected: (host: string) => void;
  onClose: () => void;
}

export default function ServerSettings({ onConnected, onClose }: Props) {
  const dialog = useRef<HTMLDialogElement>(null);
  const [config, setConfig] = useState<DesktopServerConfig | null>(null);
  const [mode, setMode] = useState<RunMode>("host");
  const [host, setHost] = useState("");
  const [label, setLabel] = useState("");
  const [testing, setTesting] = useState(false);
  const [saving, setSaving] = useState(false);
  const [testResult, setTestResult] = useState<ServerTestResult | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    dialog.current?.showModal();
    void (async () => {
      try {
        const loaded = await getDesktopServerConfig();
        setConfig(loaded);
        setMode(loaded.mode ?? "host");
        setHost(loaded.mode === "connect" ? loaded.active_host : "");
      } catch (err) {
        setError((err as Error).message);
      }
    })();
  }, []);

  async function runTest(targetHost = host) {
    const trimmed = targetHost.trim();
    if (!trimmed) return;
    setTesting(true);
    setError("");
    setTestResult(null);
    try {
      const result = await testDesktopServerConnection(trimmed);
      setTestResult(result);
    } catch (err) {
      setTestResult({ ok: false, message: (err as Error).message });
    } finally {
      setTesting(false);
    }
  }

  async function save(event: React.FormEvent) {
    event.preventDefault();
    if (mode === "host") {
      setSaving(true);
      setError("");
      try {
        const saved = await saveDesktopServerConfig({
          mode: "host",
          active_host: config?.active_host ?? "http://127.0.0.1:8000",
          saved_servers: config?.saved_servers ?? [],
        });
        onConnected(runLocationLabel(saved));
        onClose();
      } catch (err) {
        setError((err as Error).message);
      } finally {
        setSaving(false);
      }
      return;
    }
    const trimmed = host.trim().replace(/\/+$/, "");
    if (!trimmed) {
      setError("Enter the address shown in Kadan on the other computer, like http://192.168.1.50:8000.");
      return;
    }
    setSaving(true);
    setError("");
    try {
      const existing = config?.saved_servers ?? [];
      const alreadySaved = existing.some((s) => s.host === trimmed);
      const nextSaved: SavedServer[] = alreadySaved
        ? existing
        : [
            ...existing,
            {
              id: `srv-${Date.now()}`,
              label: label.trim() || trimmed.replace(/^https?:\/\//, ""),
              host: trimmed,
            },
          ];
      const saved = await saveDesktopServerConfig({
        mode: "connect",
        active_host: trimmed,
        saved_servers: nextSaved,
      });
      setConfig(saved);
      setLabel("");
      onConnected(runLocationLabel(saved));
      onClose();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setSaving(false);
    }
  }

  async function removeSaved(id: string) {
    if (!config) return;
    const nextSaved = config.saved_servers.filter((s) => s.id !== id);
    try {
      const saved = await saveDesktopServerConfig({
        mode: config.mode,
        active_host: config.active_host,
        saved_servers: nextSaved,
      });
      setConfig(saved);
    } catch (err) {
      setError((err as Error).message);
    }
  }

  return (
    <dialog
      ref={dialog}
      className="rename-agent server-settings"
      aria-labelledby="server-settings-title"
      onCancel={(event) => {
        event.preventDefault();
        onClose();
      }}
    >
      <form onSubmit={save}>
        <h2 id="server-settings-title">Where Kadan runs</h2>
        <fieldset className="run-mode" disabled={saving}>
          <legend className="sr-only">Where Kadan runs</legend>
          <label className={mode === "host" ? "selected" : ""}>
            <input type="radio" name="run-mode" checked={mode === "host"} onChange={() => setMode("host")} />
            <Laptop size={18} aria-hidden="true" />
            <span>
              <strong>This computer</strong>
              <small>Kadan runs here, including its AI models. Best if this computer has a good graphics card.</small>
            </span>
          </label>
          <label className={mode === "connect" ? "selected" : ""}>
            <input type="radio" name="run-mode" checked={mode === "connect"} onChange={() => setMode("connect")} />
            <Server size={18} aria-hidden="true" />
            <span>
              <strong>Another computer</strong>
              <small>Use Kadan running on another computer, like a home server. This computer does no heavy work.</small>
            </span>
          </label>
        </fieldset>
        {mode === "connect" && (<>
        <p className="field-hint">
          On the other computer, open Kadan Server → Settings and copy its address (for example{" "}
          <code>http://192.168.1.50:8000</code>).
        </p>

        <label>
          Server address
          <input
            autoFocus
            required
            placeholder="http://192.168.1.50:8000"
            value={host}
            disabled={saving}
            onChange={(event) => {
              setHost(event.target.value);
              setTestResult(null);
            }}
          />
        </label>

        <label>
          Friendly name (optional)
          <input
            placeholder="Home GPU PC"
            value={label}
            disabled={saving}
            onChange={(event) => setLabel(event.target.value)}
          />
        </label>

        {testResult && (
          <p className={testResult.ok ? "server-test-ok" : "error"} role="status">
            {testResult.ok ? <CheckCircle2 size={15} /> : <XCircle size={15} />}
            <span>
              {testResult.message}
              {testResult.ok && testResult.version ? ` (v${testResult.version})` : ""}
              {testResult.ok && typeof testResult.latency_ms === "number"
                ? ` · ${testResult.latency_ms} ms`
                : ""}
            </span>
          </p>
        )}

        {error && (
          <p className="error" role="alert">
            {error}
          </p>
        )}

        {config && config.saved_servers.length > 0 && (
          <div className="saved-servers">
            <span className="panel-label">Saved servers</span>
            <ul>
              {config.saved_servers.map((server) => (
                <li
                  key={server.id}
                  className={server.host === host.trim() ? "active" : ""}
                >
                  <button
                    type="button"
                    className="saved-server-pick"
                    onClick={() => {
                      setHost(server.host);
                      setTestResult(null);
                    }}
                  >
                    <strong>{server.label}</strong>
                    <code>{server.host}</code>
                  </button>
                  <button
                    type="button"
                    className="icon"
                    aria-label={`Remove ${server.label}`}
                    onClick={() => void removeSaved(server.id)}
                  >
                    <Trash2 size={14} />
                  </button>
                </li>
              ))}
            </ul>
          </div>
        )}

        </>)}
        {mode === "host" && error && (
          <p className="error" role="alert">
            {error}
          </p>
        )}

        <div className="form-actions">
          {mode === "connect" && (
            <button
              type="button"
              disabled={testing || saving || !host.trim()}
              onClick={() => void runTest()}
            >
              <Plug size={15} />
              {testing ? "Testing…" : "Test connection"}
            </button>
          )}
          <span style={{ flex: 1 }} />
          <button type="button" disabled={saving} onClick={onClose}>
            Cancel
          </button>
          <button className="primary" disabled={saving || (mode === "connect" && !host.trim())}>
            {saving ? "Saving…" : mode === "host" ? "Run Kadan here" : "Connect"}
          </button>
        </div>
      </form>
    </dialog>
  );
}
