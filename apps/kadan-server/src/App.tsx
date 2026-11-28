import { useCallback, useEffect, useState } from "react";
import {
  Activity,
  Check,
  Copy,
  Cpu,
  Download,
  ExternalLink,
  FolderOpen,
  Play,
  RefreshCw,
  ScrollText,
  Server,
  Settings,
  Sparkles,
  Square,
} from "lucide-react";
import {
  checkUpdates,
  getConfig,
  getHardware,
  getLogs,
  getStatus,
  getUpdateStatus,
  installUpdate,
  openExternalUrl,
  pickFolder,
  restartService,
  startServer,
  stopServer,
  updateConfig,
} from "./api";
import type {
  HardwareSummary,
  LogEntry,
  ServerConfig,
  ServerStatusSnapshot,
  UpdateCheckResult,
  UpdateStatus,
} from "./types";

type Tab = "server" | "computer" | "logs";

function formatGb(mb: number): string {
  return `${(mb / 1024).toFixed(1)} GB`;
}

export default function App() {
  const [tab, setTab] = useState<Tab>("server");
  const [config, setConfig] = useState<ServerConfig | null>(null);
  const [draft, setDraft] = useState<ServerConfig | null>(null);
  const [status, setStatus] = useState<ServerStatusSnapshot | null>(null);
  const [hardware, setHardware] = useState<HardwareSummary | null>(null);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [logFilter, setLogFilter] = useState<string>("all");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [copiedUrl, setCopiedUrl] = useState<string | null>(null);
  const [updateStatus, setUpdateStatus] = useState<UpdateStatus | null>(null);
  const [updateCheck, setUpdateCheck] = useState<UpdateCheckResult | null>(null);
  const [updating, setUpdating] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const [st, hw] = await Promise.all([getStatus(), getHardware()]);
      setStatus(st);
      setHardware(hw);
      if (tab === "logs") {
        setLogs(await getLogs(logFilter));
      }
    } catch (err) {
      setError(String(err));
    }
  }, [tab, logFilter]);

  useEffect(() => {
    void (async () => {
      try {
        const [cfg, st, hw, upd] = await Promise.all([
          getConfig(),
          getStatus(),
          getHardware(),
          getUpdateStatus(),
        ]);
        setConfig(cfg);
        setDraft(cfg);
        setStatus(st);
        setHardware(hw);
        setUpdateStatus(upd);
      } catch (err) {
        setError(String(err));
      }
    })();
  }, []);

  useEffect(() => {
    const timer = setInterval(() => void refresh(), 2500);
    return () => clearInterval(timer);
  }, [refresh]);

  useEffect(() => {
    if (tab === "logs") {
      void getLogs(logFilter).then(setLogs).catch(() => {});
    }
  }, [tab, logFilter]);

  const apiSvc = status?.services.find((s) => s.id === "api");
  const apiReady = apiSvc?.phase === "running";

  const localHostUrl = config ? `http://127.0.0.1:${config.server_port}` : "http://127.0.0.1:8000";
  const lanHostUrl =
    config && config.bind_all_interfaces && hardware?.lan_ip
      ? `http://${hardware.lan_ip}:${config.server_port}`
      : null;

  async function copyUrl(url: string) {
    try {
      await navigator.clipboard.writeText(url);
      setCopiedUrl(url);
      setTimeout(() => setCopiedUrl((cur) => (cur === url ? null : cur)), 2000);
    } catch {
      // ignore clipboard errors
    }
  }

  async function handleStart() {
    setBusy(true);
    setError("");
    setNotice("");
    try {
      setStatus(await startServer());
    } catch (err) {
      setError(String(err));
    } finally {
      setBusy(false);
    }
  }

  async function handleStop() {
    setBusy(true);
    setError("");
    try {
      setStatus(await stopServer());
    } catch (err) {
      setError(String(err));
    } finally {
      setBusy(false);
    }
  }

  async function handleRestartService(serviceId: string) {
    setBusy(true);
    setError("");
    try {
      setStatus(await restartService(serviceId));
    } catch (err) {
      setError(String(err));
    } finally {
      setBusy(false);
    }
  }

  async function handleCheckUpdates() {
    setUpdating(true);
    setError("");
    setNotice("");
    try {
      const result = await checkUpdates();
      setUpdateCheck(result);
      setUpdateStatus(await getUpdateStatus());
      if (result.update_available && result.latest_version) {
        setNotice(`Version ${result.latest_version} is available.`);
      } else {
        setNotice(`Kadan Server is up to date (${result.current_version}).`);
      }
    } catch (err) {
      setError(String(err));
    } finally {
      setUpdating(false);
    }
  }

  async function handleInstallUpdate() {
    setUpdating(true);
    setError("");
    setNotice("");
    try {
      const result = await installUpdate();
      setNotice(result.message);
    } catch (err) {
      setError(String(err));
    } finally {
      setUpdating(false);
    }
  }

  async function handleSaveSettings(restartNow: boolean) {
    if (!draft) return;
    setBusy(true);
    setError("");
    setNotice("");
    try {
      const saved = await updateConfig(draft, restartNow);
      setConfig(saved);
      setDraft(saved);
      setStatus(await getStatus());
      setNotice(restartNow ? "Settings saved and server restarted." : "Settings saved.");
    } catch (err) {
      setError(String(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="shell">
      <header className="topbar">
        <div className="brand">
          <span className={`status-orb ${apiReady ? "ok" : status?.running ? "warn" : "off"}`} />
          <div>
            <h1>Kadan Server</h1>
            <span className="subtitle">
              {apiReady
                ? lanHostUrl
                  ? `Serving on ${lanHostUrl}`
                  : `Serving on ${localHostUrl}`
                : status?.running
                  ? "Starting services…"
                  : "Stopped"}
            </span>
          </div>
        </div>

        <nav className="tabs" aria-label="Server views">
          <button className={tab === "server" ? "active" : ""} onClick={() => setTab("server")}>
            <Sparkles size={15} />
            Kadan Server
          </button>
          <button className={tab === "computer" ? "active" : ""} onClick={() => setTab("computer")}>
            <Server size={15} />
            This computer
          </button>
          <button className={tab === "logs" ? "active" : ""} onClick={() => setTab("logs")}>
            <ScrollText size={15} />
            Logs
          </button>
        </nav>

        <div className="top-actions">
          {status?.running ? (
            <button className="btn danger" disabled={busy} onClick={() => void handleStop()}>
              <Square size={14} />
              Stop Server
            </button>
          ) : (
            <button className="btn primary" disabled={busy} onClick={() => void handleStart()}>
              <Play size={14} />
              Start Server
            </button>
          )}
        </div>
      </header>

      {error && (
        <div className="banner error" role="alert">
          <span>{error}</span>
          <button onClick={() => setError("")}>Dismiss</button>
        </div>
      )}
      {notice && (
        <div className="banner ok" role="status">
          <span>{notice}</span>
          <button onClick={() => setNotice("")}>Dismiss</button>
        </div>
      )}

      {tab === "server" && (
        <main className="embedded-pane">
          {apiReady && status ? (
            <iframe title="Kadan Server" src={`${localHostUrl}/server`} className="embedded-frame" />
          ) : (
            <div className="empty-state">
              <Sparkles size={32} />
              <h2>{status?.running ? "Kadan Server is starting…" : "Kadan Server is stopped"}</h2>
              <p>
                {status?.running
                  ? "This takes a few seconds. Your AI models, agent and activity will show up here."
                  : "Start it to manage AI models, see what your agent remembers and follow its activity."}
              </p>
              {!status?.running && (
                <button className="btn primary" disabled={busy} onClick={() => void handleStart()}>
                  <Play size={14} />
                  Start Server
                </button>
              )}
            </div>
          )}
        </main>
      )}
      {tab === "computer" && (
        <main className="content">
          <section className="card">
            <div className="card-head">
              <h2>Connect Kadan Desktop</h2>
              <p>
                Paste one of these addresses into <strong>Kadan → Server settings</strong>.
              </p>
            </div>
            <div className="host-list">
              <div className="host-row">
                <div>
                  <span className="host-label">This computer</span>
                  <code>{localHostUrl}</code>
                </div>
                <div className="row-actions">
                  <button className="btn" onClick={() => void copyUrl(localHostUrl)}>
                    {copiedUrl === localHostUrl ? <Check size={14} /> : <Copy size={14} />}
                    {copiedUrl === localHostUrl ? "Copied" : "Copy"}
                  </button>
                  <button
                    className="btn"
                    disabled={!apiReady}
                    onClick={() => void openExternalUrl(localHostUrl)}
                  >
                    <ExternalLink size={14} />
                    Open in Browser
                  </button>
                </div>
              </div>
              {lanHostUrl && (
                <div className="host-row">
                  <div>
                    <span className="host-label">Other devices on your network</span>
                    <code>{lanHostUrl}</code>
                  </div>
                  <div className="row-actions">
                    <button className="btn" onClick={() => void copyUrl(lanHostUrl)}>
                      {copiedUrl === lanHostUrl ? <Check size={14} /> : <Copy size={14} />}
                      {copiedUrl === lanHostUrl ? "Copied" : "Copy"}
                    </button>
                  </div>
                </div>
              )}
            </div>
          </section>

          <div className="grid-two">
            <section className="card">
              <div className="card-head">
                <h2>
                  <Activity size={16} /> Services
                </h2>
              </div>
              <div className="service-list">
                {(status?.services ?? []).map((svc) => (
                  <div key={svc.id} className="service-row">
                    <span className={`pill ${svc.phase}`}>{svc.phase}</span>
                    <div className="service-info">
                      <strong>{svc.label}</strong>
                      <span>{svc.message}</span>
                    </div>
                    <button
                      className="btn small"
                      disabled={busy || !status?.running}
                      onClick={() => void handleRestartService(svc.id)}
                      title={`Restart ${svc.label}`}
                    >
                      <RefreshCw size={13} />
                    </button>
                  </div>
                ))}
              </div>
            </section>

            <section className="card">
              <div className="card-head">
                <h2>
                  <Cpu size={16} /> Hardware
                </h2>
              </div>
              {hardware ? (
                <div className="hw-list">
                  <div className="hw-row">
                    <span>Computer</span>
                    <strong>{hardware.hostname}</strong>
                  </div>
                  <div className="hw-row">
                    <span>CPU</span>
                    <strong>{hardware.cpu_cores} cores</strong>
                  </div>
                  <div className="hw-row">
                    <span>Memory</span>
                    <strong>
                      {formatGb(hardware.available_ram_mb)} free / {formatGb(hardware.total_ram_mb)}
                    </strong>
                  </div>
                  {hardware.gpus.length > 0 ? (
                    hardware.gpus.map((gpu, idx) => (
                      <div key={idx} className="hw-row">
                        <span>GPU {hardware.gpus.length > 1 ? idx + 1 : ""}</span>
                        <strong>
                          {gpu.name} ({formatGb(gpu.free_vram_mb)} free / {formatGb(gpu.total_vram_mb)})
                        </strong>
                      </div>
                    ))
                  ) : (
                    <div className="hw-row">
                      <span>GPU</span>
                      <strong>CPU mode (no NVIDIA GPU detected)</strong>
                    </div>
                  )}
                  <div className="hw-row">
                    <span>Data folder</span>
                    <code className="truncate">{config?.data_dir}</code>
                  </div>
                  <div className="hw-row">
                    <span>Version</span>
                    <div className="row-actions">
                      <strong>{updateStatus?.current_version ?? "0.1.0"}</strong>
                      {updateCheck?.update_available && updateCheck.can_self_update ? (
                        <button
                          className="btn primary small"
                          disabled={updating || busy}
                          onClick={() => void handleInstallUpdate()}
                        >
                          <Download size={13} />
                          Update to {updateCheck.latest_version}
                        </button>
                      ) : (
                        <button
                          className="btn small"
                          disabled={updating || busy}
                          onClick={() => void handleCheckUpdates()}
                        >
                          <RefreshCw size={13} />
                          Check for updates
                        </button>
                      )}
                    </div>
                  </div>
                </div>
              ) : (
                <p className="muted">Reading hardware details…</p>
              )}
            </section>
          </div>

          {draft && (
            <section className="card">
              <div className="card-head">
                <h2>
                  <Settings size={16} /> Server Settings
                </h2>
                <p>
                  AI models are managed in the <strong>Kadan Server</strong> tab, under{" "}
                  <strong>AI models</strong>.
                </p>
              </div>
              <div className="form-grid">
                <label className="checkbox-row">
                  <input
                    type="checkbox"
                    checked={draft.bind_all_interfaces}
                    onChange={(e) =>
                      setDraft({ ...draft, bind_all_interfaces: e.target.checked })
                    }
                  />
                  <span>Allow connections from other computers on my network</span>
                </label>

                <label>
                  Server Port
                  <input
                    type="number"
                    min={1}
                    max={65535}
                    value={draft.server_port}
                    onChange={(e) =>
                      setDraft({ ...draft, server_port: Number(e.target.value) || 8000 })
                    }
                  />
                </label>

                <label>
                  Models &amp; Data Folder
                  <div className="folder-row">
                    <input
                      value={draft.data_dir}
                      onChange={(e) => setDraft({ ...draft, data_dir: e.target.value })}
                    />
                    <button
                      type="button"
                      className="btn"
                      onClick={async () => {
                        const picked = await pickFolder(draft.data_dir);
                        if (picked) setDraft({ ...draft, data_dir: picked });
                      }}
                    >
                      <FolderOpen size={15} />
                      Browse
                    </button>
                  </div>
                </label>

                <label className="checkbox-row">
                  <input
                    type="checkbox"
                    checked={draft.auto_start_server}
                    onChange={(e) =>
                      setDraft({ ...draft, auto_start_server: e.target.checked })
                    }
                  />
                  <span>Start Kadan Server automatically when this app opens</span>
                </label>

                <label className="checkbox-row">
                  <input
                    type="checkbox"
                    checked={draft.launch_at_login}
                    onChange={(e) =>
                      setDraft({ ...draft, launch_at_login: e.target.checked })
                    }
                  />
                  <span>Launch Kadan Server when I sign into my computer</span>
                </label>
              </div>

              <div className="save-bar">
                <button
                  className="btn"
                  disabled={busy}
                  onClick={() => void handleSaveSettings(false)}
                >
                  Save Only
                </button>
                <button
                  className="btn primary"
                  disabled={busy}
                  onClick={() => void handleSaveSettings(true)}
                >
                  Save &amp; Restart Server
                </button>
              </div>
            </section>
          )}
        </main>
      )}

      {tab === "logs" && (
        <main className="content logs-layout">
          <div className="log-toolbar">
            {["all", "api", "runner", "worker"].map((svc) => (
              <button
                key={svc}
                className={logFilter === svc ? "btn primary small" : "btn small"}
                onClick={() => setLogFilter(svc)}
              >
                {svc}
              </button>
            ))}
          </div>
          <pre className="log-box">
            {logs.length === 0
              ? "No logs yet."
              : logs.map((e) => `[${e.service}] ${e.line}`).join("\n")}
          </pre>
        </main>
      )}
    </div>
  );
}
