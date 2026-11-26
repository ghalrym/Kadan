import { useEffect, useRef, useState } from "react";
import { CheckCircle2, Plug, Plus, Trash2, XCircle } from "lucide-react";
import {
  getDesktopServerConfig,
  saveDesktopServerConfig,
  testDesktopServerConnection,
  type DesktopServerConfig,
  type SavedServer,
  type ServerTestResult,
} from "./desktop";

interface Props {
  onConnected: (host: string) => void;
  onClose: () => void;
}

export default function ServerSettings({ onConnected, onClose }: Props) {
  const dialog = useRef<HTMLDialogElement>(null);
  const [config, setConfig] = useState<DesktopServerConfig | null>(null);
  const [host, setHost] = useState("http://127.0.0.1:8000");
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
        setHost(loaded.active_host);
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
    const trimmed = host.trim().replace(/\/+$/, "");
    if (!trimmed) {
      setError("Enter a server address like http://127.0.0.1:8000.");
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
        active_host: trimmed,
        saved_servers: nextSaved,
      });
      setConfig(saved);
      setLabel("");
      onConnected(saved.active_host);
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
        <h2 id="server-settings-title">Kadan Server Connection</h2>
        <p className="field-hint">
          Enter the host address shown in Kadan Server (for example{" "}
          <code>http://127.0.0.1:8000</code> on this computer or{" "}
          <code>http://192.168.1.50:8000</code> on your network).
        </p>

        <label>
          Server address
          <input
            autoFocus
            required
            placeholder="http://127.0.0.1:8000"
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

        <div className="form-actions">
          <button
            type="button"
            disabled={testing || saving || !host.trim()}
            onClick={() => void runTest()}
          >
            <Plug size={15} />
            {testing ? "Testing…" : "Test connection"}
          </button>
          <span style={{ flex: 1 }} />
          <button type="button" disabled={saving} onClick={onClose}>
            Cancel
          </button>
          <button className="primary" disabled={saving || !host.trim()}>
            <Plus size={15} />
            {saving ? "Connecting…" : "Connect"}
          </button>
        </div>
      </form>
    </dialog>
  );
}
