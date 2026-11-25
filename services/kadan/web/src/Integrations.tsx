import { useEffect, useRef, useState } from "react";
import { api } from "./api";
import {
  statusLabel,
  T3_TOOL_GROUPS,
  toggleGroupTools,
  type McpConnectResponse,
  type McpServer,
} from "./integrations";

interface Props {
  onClose: () => void;
}

export default function Integrations({ onClose }: Props) {
  const dialog = useRef<HTMLDialogElement>(null);
  const active = useRef(true);
  const [servers, setServers] = useState<McpServer[]>([]);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [pendingAuthUrl, setPendingAuthUrl] = useState<string | null>(null);

  // T3 Code quick-setup form
  const [t3Url, setT3Url] = useState("");

  // Generic MCP connection form
  const [customName, setCustomName] = useState("");
  const [customUrl, setCustomUrl] = useState("");
  const [customAuth, setCustomAuth] = useState<"oauth" | "bearer" | "none">("oauth");
  const [customToken, setCustomToken] = useState("");

  const t3Server = servers.find((s) => s.preset === "t3_code") ?? null;
  const customServers = servers.filter((s) => s.preset !== "t3_code");

  async function refresh() {
    setLoading(true);
    setError("");
    try {
      const list = await api<McpServer[]>("/mcp/servers");
      if (active.current) setServers(list);
    } catch (e) {
      if (active.current) setError(`Could not load integrations: ${(e as Error).message}`);
    } finally {
      if (active.current) setLoading(false);
    }
  }

  useEffect(() => {
    active.current = true;
    dialog.current?.showModal();
    void refresh();
    const onMessage = (event: MessageEvent) => {
      if (event.origin !== window.location.origin) return;
      if (event.data?.type === "kadan:mcp-oauth") {
        setPendingAuthUrl(null);
        setNotice(event.data.ok ? "Connected." : "Sign-in did not complete.");
        void refresh();
      }
    };
    window.addEventListener("message", onMessage);
    return () => {
      active.current = false;
      window.removeEventListener("message", onMessage);
    };
  }, []);

  async function connectServer(id: string) {
    setBusy(true);
    setError("");
    setNotice("");
    setPendingAuthUrl(null);
    try {
      const result = await api<McpConnectResponse>(`/mcp/servers/${id}/connect`, {}, "POST");
      if (!active.current) return;
      if (result.authorize_url) {
        setPendingAuthUrl(result.authorize_url);
        setNotice("Approve the connection in the new browser tab, then return here.");
        window.open(result.authorize_url, "_blank", "noopener");
      } else {
        setNotice(`Connected to ${result.server.name}.`);
      }
      await refresh();
    } catch (e) {
      if (active.current) setError((e as Error).message);
    } finally {
      if (active.current) setBusy(false);
    }
  }

  async function addAndConnectT3(event: React.FormEvent) {
    event.preventDefault();
    if (busy || !t3Url.trim()) return;
    setBusy(true);
    setError("");
    setNotice("");
    try {
      const created = await api<McpServer>(
        "/mcp/servers",
        { name: "T3 Code", url: t3Url.trim(), preset: "t3_code", auth_type: "oauth" },
        "POST",
      );
      if (!active.current) return;
      setT3Url("");
      const result = await api<McpConnectResponse>(`/mcp/servers/${created.id}/connect`, {}, "POST");
      if (!active.current) return;
      if (result.authorize_url) {
        setPendingAuthUrl(result.authorize_url);
        setNotice("Approve Kadan in T3 Code in the opened browser tab.");
        window.open(result.authorize_url, "_blank", "noopener");
      } else {
        setNotice("T3 Code is connected.");
      }
      await refresh();
    } catch (e) {
      if (active.current) setError((e as Error).message);
    } finally {
      if (active.current) setBusy(false);
    }
  }

  async function addCustomServer(event: React.FormEvent) {
    event.preventDefault();
    if (busy || !customName.trim() || !customUrl.trim()) return;
    setBusy(true);
    setError("");
    setNotice("");
    try {
      const created = await api<McpServer>(
        "/mcp/servers",
        {
          name: customName.trim(),
          url: customUrl.trim(),
          preset: "custom",
          auth_type: customAuth,
          bearer_token: customAuth === "bearer" ? customToken.trim() || null : null,
        },
        "POST",
      );
      if (!active.current) return;
      setCustomName("");
      setCustomUrl("");
      setCustomToken("");
      const result = await api<McpConnectResponse>(`/mcp/servers/${created.id}/connect`, {}, "POST");
      if (!active.current) return;
      if (result.authorize_url) {
        setPendingAuthUrl(result.authorize_url);
        setNotice(`Approve ${created.name} in the opened browser tab.`);
        window.open(result.authorize_url, "_blank", "noopener");
      } else {
        setNotice(`${created.name} is connected.`);
      }
      await refresh();
    } catch (e) {
      if (active.current) setError((e as Error).message);
    } finally {
      if (active.current) setBusy(false);
    }
  }

  async function refreshTools(server: McpServer) {
    setBusy(true);
    setError("");
    setNotice("");
    try {
      await api<McpServer>(`/mcp/servers/${server.id}/refresh`, {}, "POST");
      if (active.current) setNotice(`Updated tools from ${server.name}.`);
      await refresh();
    } catch (e) {
      if (active.current) setError((e as Error).message);
    } finally {
      if (active.current) setBusy(false);
    }
  }

  async function toggleEnabled(server: McpServer) {
    setBusy(true);
    setError("");
    try {
      await api<McpServer>(
        `/mcp/servers/${server.id}`,
        { expected_version: server.version, enabled: !server.enabled },
        "PUT",
      );
      await refresh();
    } catch (e) {
      if (active.current) setError((e as Error).message);
    } finally {
      if (active.current) setBusy(false);
    }
  }

  async function updatePinned(server: McpServer, pinned: string[]) {
    setBusy(true);
    setError("");
    try {
      await api<McpServer>(
        `/mcp/servers/${server.id}`,
        { expected_version: server.version, pinned_tools: pinned },
        "PUT",
      );
      await refresh();
    } catch (e) {
      if (active.current) setError((e as Error).message);
    } finally {
      if (active.current) setBusy(false);
    }
  }

  async function disconnect(server: McpServer) {
    setBusy(true);
    setError("");
    setNotice("");
    try {
      await api<McpServer>(`/mcp/servers/${server.id}/disconnect`, {}, "POST");
      if (active.current) setNotice(`Disconnected ${server.name}.`);
      await refresh();
    } catch (e) {
      if (active.current) setError((e as Error).message);
    } finally {
      if (active.current) setBusy(false);
    }
  }

  async function remove(server: McpServer) {
    setBusy(true);
    setError("");
    setNotice("");
    try {
      await api<void>(`/mcp/servers/${server.id}`, undefined, "DELETE");
      if (active.current) setNotice(`Removed ${server.name}.`);
      await refresh();
    } catch (e) {
      if (active.current) setError((e as Error).message);
    } finally {
      if (active.current) setBusy(false);
    }
  }

  return (
    <dialog
      ref={dialog}
      className="modal integrations-dialog"
      aria-labelledby="integrations-title"
      onCancel={(e) => { e.preventDefault(); if (!busy) onClose(); }}
      onClick={(e) => { if (e.target === dialog.current && !busy) onClose(); }}
    >
      <header className="modal-head">
        <h2 id="integrations-title">Integrations</h2>
        <button className="icon" type="button" aria-label="Close integrations" disabled={busy} onClick={onClose}>×</button>
      </header>
      <div className="integrations-content">
        <p className="field-hint">
          Connect external tools and coding assistants over MCP. Connected tools are available in chat.
        </p>

        {error && <p className="error" role="alert">{error}</p>}
        {notice && <p className="field-hint" role="status">{notice}</p>}
        {pendingAuthUrl && (
          <p>
            <a href={pendingAuthUrl} target="_blank" rel="noreferrer">
              Open approval page again
            </a>
          </p>
        )}

        <section className="integration-card" aria-label="T3 Code">
          <div className="integration-head">
            <div>
              <h3>T3 Code</h3>
              <p className="field-hint">
                Let your agent start coding threads, follow progress, and relay questions from T3 Code.
              </p>
            </div>
            {t3Server && (
              <span className={`integration-badge ${t3Server.status}`}>
                {statusLabel(t3Server)}
              </span>
            )}
          </div>

          {!t3Server ? (
            <form className="integration-form" onSubmit={addAndConnectT3}>
              <label>
                T3 Code MCP URL
                <input
                  type="url"
                  required
                  placeholder="https://your-t3-code/mcp"
                  value={t3Url}
                  disabled={busy}
                  onChange={(e) => setT3Url(e.target.value)}
                />
              </label>
              <p className="field-hint">
                In T3 Code, open Settings → Environments and choose Copy MCP URL.
              </p>
              <div className="integration-actions">
                <button className="primary" type="submit" disabled={busy || !t3Url.trim()}>
                  Connect T3 Code
                </button>
              </div>
            </form>
          ) : (
            <div className="integration-body">
              <p className="field-hint">{t3Server.url}</p>
              {t3Server.status_error && <p className="error">{t3Server.status_error}</p>}
              <div className="integration-actions">
                {t3Server.status !== "connected" && (
                  <button className="primary" type="button" disabled={busy} onClick={() => connectServer(t3Server.id)}>
                    Sign in to T3 Code
                  </button>
                )}
                {t3Server.app_url && (
                  <a className="button-link" href={t3Server.app_url} target="_blank" rel="noreferrer">
                    Open T3 Code
                  </a>
                )}
                <button type="button" disabled={busy} onClick={() => refreshTools(t3Server)}>
                  Refresh tools
                </button>
                <button type="button" disabled={busy} onClick={() => toggleEnabled(t3Server)}>
                  {t3Server.enabled ? "Pause" : "Resume"}
                </button>
                {t3Server.has_credentials && (
                  <button type="button" disabled={busy} onClick={() => disconnect(t3Server)}>
                    Disconnect
                  </button>
                )}
                <button type="button" disabled={busy} onClick={() => remove(t3Server)}>
                  Remove
                </button>
              </div>
              {t3Server.tools.length > 0 && (
                <details className="integration-tools">
                  <summary>
                    Tools ({t3Server.pinned_tools.length} active directly · {t3Server.tools.length} total available)
                  </summary>
                  <p className="field-hint">
                    Selected tool groups are ready immediately; all other T3 Code tools remain discoverable on demand.
                  </p>
                  <div className="integration-groups">
                    {T3_TOOL_GROUPS.map((group) => {
                      const available = group.tools.filter((name) =>
                        t3Server.tools.some((t) => t.name === name),
                      );
                      if (available.length === 0) return null;
                      const checked = available.every((name) => t3Server.pinned_tools.includes(name));
                      return (
                        <label key={group.id} className="integration-check">
                          <input
                            type="checkbox"
                            checked={checked}
                            disabled={busy}
                            onChange={(e) =>
                              updatePinned(
                                t3Server,
                                toggleGroupTools(t3Server.pinned_tools, t3Server.tools, group.tools, e.target.checked),
                              )
                            }
                          />
                          <span>{group.label}</span>
                        </label>
                      );
                    })}
                  </div>
                </details>
              )}
            </div>
          )}
        </section>

        <section className="integration-card" aria-label="Other MCP connections">
          <h3>Other MCP servers</h3>
          {loading ? (
            <p className="field-hint">Checking connections…</p>
          ) : customServers.length === 0 ? (
            <p className="field-hint">No other MCP servers added yet.</p>
          ) : (
            <ul className="integration-list">
              {customServers.map((server) => (
                <li key={server.id} className="integration-item">
                  <div className="integration-head">
                    <div>
                      <strong>{server.name}</strong>
                      <p className="field-hint">
                        {server.url} · {server.tools.length} tools
                      </p>
                    </div>
                    <span className={`integration-badge ${server.status}`}>
                      {statusLabel(server)}
                    </span>
                  </div>
                  {server.status_error && <p className="error">{server.status_error}</p>}
                  <div className="integration-actions">
                    {server.status !== "connected" && (
                      <button type="button" disabled={busy} onClick={() => connectServer(server.id)}>
                        Connect
                      </button>
                    )}
                    <button type="button" disabled={busy} onClick={() => refreshTools(server)}>
                      Refresh
                    </button>
                    <button type="button" disabled={busy} onClick={() => toggleEnabled(server)}>
                      {server.enabled ? "Pause" : "Resume"}
                    </button>
                    <button type="button" disabled={busy} onClick={() => remove(server)}>
                      Remove
                    </button>
                  </div>
                </li>
              ))}
            </ul>
          )}

          <details>
            <summary>Add another MCP server</summary>
            <form className="integration-form" onSubmit={addCustomServer}>
              <label>
                Name
                <input
                  type="text"
                  required
                  placeholder="Linear, Home Assistant, etc."
                  value={customName}
                  disabled={busy}
                  onChange={(e) => setCustomName(e.target.value)}
                />
              </label>
              <label>
                MCP URL
                <input
                  type="url"
                  required
                  placeholder="https://example.com/mcp"
                  value={customUrl}
                  disabled={busy}
                  onChange={(e) => setCustomUrl(e.target.value)}
                />
              </label>
              <label>
                Sign-in method
                <select
                  value={customAuth}
                  disabled={busy}
                  onChange={(e) => setCustomAuth(e.target.value as "oauth" | "bearer" | "none")}
                >
                  <option value="oauth">Browser sign-in (OAuth)</option>
                  <option value="bearer">Access token</option>
                  <option value="none">No sign-in required</option>
                </select>
              </label>
              {customAuth === "bearer" && (
                <label>
                  Access token
                  <input
                    type="password"
                    required
                    value={customToken}
                    disabled={busy}
                    onChange={(e) => setCustomToken(e.target.value)}
                  />
                </label>
              )}
              <div className="integration-actions">
                <button className="primary" type="submit" disabled={busy || !customName.trim() || !customUrl.trim()}>
                  Add & connect
                </button>
              </div>
            </form>
          </details>
        </section>
      </div>
    </dialog>
  );
}
