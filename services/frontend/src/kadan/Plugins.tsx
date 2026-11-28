import { useEffect, useRef, useState } from "react";
import { ArrowLeft, ArrowUpRight, Check, Plus, Search } from "lucide-react";
import { api } from "./api";
import PluginIcon from "./PluginIcon";
import {
  statusLabel,
  T3_TOOL_GROUPS,
  toggleGroupTools,
  type McpConnectResponse,
  type McpServer,
} from "./integrations";

interface Props {
  agentName: string;
}

/** What the plugin page has open: the catalog, an installed plugin, or a setup form. */
type Selection = { name: "catalog" } | { name: "server"; id: string } | { name: "setup"; preset: "t3_code" | "custom" };

const T3_DESCRIPTION = "Hand coding work to T3 Code, follow its threads and answer its questions from chat.";
const CUSTOM_DESCRIPTION = "Connect any app or service that gives you a connection address.";

export default function Plugins({ agentName }: Props) {
  const active = useRef(true);
  const [servers, setServers] = useState<McpServer[]>([]);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [pendingAuthUrl, setPendingAuthUrl] = useState<string | null>(null);
  const [manualCallback, setManualCallback] = useState(false);
  const [callbackUrl, setCallbackUrl] = useState("");
  const [query, setQuery] = useState("");
  const [selection, setSelection] = useState<Selection>({ name: "catalog" });

  function savePendingAuth(serverId: string, authorizeUrl: string, manual: boolean) {
    setPendingAuthUrl(authorizeUrl);
    setManualCallback(manual);
    setCallbackUrl("");
    try {
      sessionStorage.setItem(
        `kadan:mcp-pending-auth:${serverId}`,
        JSON.stringify({ authorizeUrl, manual }),
      );
    } catch {
      // Ignore storage errors.
    }
  }

  function clearPendingAuth(serverId?: string) {
    setPendingAuthUrl(null);
    setManualCallback(false);
    setCallbackUrl("");
    if (serverId) {
      try {
        sessionStorage.removeItem(`kadan:mcp-pending-auth:${serverId}`);
      } catch {
        // Ignore storage errors.
      }
    }
  }

  const [t3Url, setT3Url] = useState("");
  const [customName, setCustomName] = useState("");
  const [customUrl, setCustomUrl] = useState("");
  const [customAuth, setCustomAuth] = useState<"oauth" | "bearer" | "none">("oauth");
  const [customToken, setCustomToken] = useState("");

  const t3Server = servers.find((s) => s.preset === "t3_code") ?? null;
  const selected = selection.name === "server" ? servers.find((s) => s.id === selection.id) ?? null : null;

  async function refresh() {
    setLoading(true);
    try {
      const list = await api<McpServer[]>("/mcp/servers");
      if (active.current) {
        setServers(list);
        setError("");
      }
    } catch (e) {
      if (active.current) setError(`Could not load plugins: ${(e as Error).message}`);
    } finally {
      if (active.current) setLoading(false);
    }
  }

  useEffect(() => {
    active.current = true;
    void refresh();
    const onMessage = (event: MessageEvent) => {
      if (event.origin !== window.location.origin) return;
      if (event.data?.type === "kadan:mcp-oauth") {
        setPendingAuthUrl(null);
        setManualCallback(false);
        setCallbackUrl("");
        setNotice(event.data.ok ? "Connected." : "Sign-in did not complete. Try signing in again.");
        void refresh();
      }
    };
    window.addEventListener("message", onMessage);
    return () => {
      active.current = false;
      window.removeEventListener("message", onMessage);
    };
  }, []);

  useEffect(() => {
    if (selection.name !== "server") return;
    try {
      const raw = sessionStorage.getItem(`kadan:mcp-pending-auth:${selection.id}`);
      if (!raw) return;
      const saved = JSON.parse(raw) as { authorizeUrl?: string; manual?: boolean };
      if (typeof saved.authorizeUrl === "string" && saved.authorizeUrl) {
        setPendingAuthUrl(saved.authorizeUrl);
        setManualCallback(Boolean(saved.manual));
      }
    } catch {
      // Ignore invalid saved state.
    }
  }, [selection]);

  useEffect(() => {
    if (!pendingAuthUrl) return;
    const checkConnected = async () => {
      try {
        const list = await api<McpServer[]>("/mcp/servers");
        if (!active.current) return;
        setServers(list);
        const target = selection.name === "server" ? list.find((s) => s.id === selection.id) : null;
        if (target?.status === "connected") {
          clearPendingAuth(target.id);
          setNotice(`${target.name} is connected.`);
        }
      } catch {
        // Ignore transient poll errors while waiting for browser approval.
      }
    };
    const onFocus = () => {
      if (document.visibilityState !== "hidden") void checkConnected();
    };
    window.addEventListener("focus", onFocus);
    document.addEventListener("visibilitychange", onFocus);
    return () => {
      window.removeEventListener("focus", onFocus);
      document.removeEventListener("visibilitychange", onFocus);
    };
  }, [pendingAuthUrl, selection]);

  function open(next: Selection) {
    setSelection(next);
    setError("");
    setNotice("");
    setPendingAuthUrl(null);
    setManualCallback(false);
    setCallbackUrl("");
  }

  /** Run one plugin action, keeping its message visible and reloading the list afterwards. */
  async function run(action: () => Promise<string | void>) {
    if (busy) return;
    setBusy(true);
    setError("");
    setNotice("");
    try {
      const message = await action();
      if (active.current && message) setNotice(message);
      await refresh();
    } catch (e) {
      if (active.current) setError((e as Error).message);
    } finally {
      if (active.current) setBusy(false);
    }
  }

  async function connect(server: McpServer) {
    clearPendingAuth(server.id);
    const result = await api<McpConnectResponse>(`/mcp/servers/${server.id}/connect`, {}, "POST");
    if (result.authorize_url) {
      const manual = Boolean(result.manual_callback);
      savePendingAuth(server.id, result.authorize_url, manual);
      if (!manual) {
        window.open(result.authorize_url, "_blank", "noopener");
        return `Approve ${server.name} in the new browser tab, then come back here.`;
      }
      return `Follow the two steps below to finish connecting ${server.name}.`;
    }
    return `${result.server.name} is connected.`;
  }

  function finishManualCallback(event: React.FormEvent) {
    event.preventDefault();
    if (!callbackUrl.trim()) return;
    void run(async () => {
      const connected = await api<McpServer>(
        "/mcp/oauth/complete",
        { callback_url: callbackUrl.trim() },
        "POST",
      );
      if (!active.current) return;
      clearPendingAuth(connected.id);
      return `${connected.name} is connected.`;
    });
  }

  function addT3(event: React.FormEvent) {
    event.preventDefault();
    if (!t3Url.trim()) return;
    void run(async () => {
      const created = await api<McpServer>(
        "/mcp/servers",
        { name: "T3 Code", url: t3Url.trim(), preset: "t3_code", auth_type: "oauth" },
        "POST",
      );
      if (!active.current) return;
      setServers((prev) => [...prev.filter((s) => s.id !== created.id), created]);
      setT3Url("");
      setSelection({ name: "server", id: created.id });
      return connect(created);
    });
  }

  function addCustom(event: React.FormEvent) {
    event.preventDefault();
    if (!customName.trim() || !customUrl.trim()) return;
    void run(async () => {
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
      setServers((prev) => [...prev.filter((s) => s.id !== created.id), created]);
      setCustomName("");
      setCustomUrl("");
      setCustomToken("");
      setSelection({ name: "server", id: created.id });
      return connect(created);
    });
  }

  const update = (server: McpServer, change: Record<string, unknown>) =>
    api<McpServer>(`/mcp/servers/${server.id}`, { expected_version: server.version, ...change }, "PUT");

  const messages = (
    <>
      {error && <p className="error" role="alert">{error}</p>}
      {notice && <p className="plugin-notice" role="status">{notice}</p>}
      {pendingAuthUrl && !manualCallback && (
        <a className="text-link" href={pendingAuthUrl} target="_blank" rel="noreferrer">
          Open the approval page again <ArrowUpRight size={14} />
        </a>
      )}
      {pendingAuthUrl && manualCallback && (
        <form className="plugin-form" onSubmit={finishManualCallback}>
          <p>
            <strong>Step 1:</strong>{" "}
            <a className="text-link" href={pendingAuthUrl} target="_blank" rel="noreferrer">
              Open the approval page <ArrowUpRight size={14} />
            </a>{" "}
            and approve access.
          </p>
          <label>
            Step 2: Paste the final address from that tab
            <input
              type="text"
              autoComplete="off"
              required
              placeholder="http://localhost:8630/api/mcp/oauth/callback?code=...&state=..."
              value={callbackUrl}
              disabled={busy}
              onChange={(e) => setCallbackUrl(e.target.value)}
            />
          </label>
          <p className="field-hint">
            Over a local network address without HTTPS, sign-in redirects to localhost, which won&apos;t load on another device. That&apos;s expected — copy the full address from that tab&apos;s address bar and paste it here.
          </p>
          <div className="form-actions">
            <button className="primary" disabled={busy || !callbackUrl.trim()}>
              {busy ? "Finishing…" : "Finish sign-in"}
            </button>
          </div>
        </form>
      )}
    </>
  );
  const back = (
    <button className="page-back" onClick={() => open({ name: "catalog" })}>
      <ArrowLeft size={16} />
      Plugins
    </button>
  );

  if (selection.name === "setup") {
    const t3 = selection.preset === "t3_code";
    return (
      <section className="page plugins-page" aria-labelledby="plugin-title">
        <div className="page-inner narrow">
          {back}
          <header className="plugin-hero">
            <PluginIcon preset={selection.preset} name={t3 ? "T3 Code" : "Custom"} size={56} />
            <div>
              <h1 id="plugin-title">{t3 ? "T3 Code" : "Custom plugin"}</h1>
              <p>{t3 ? T3_DESCRIPTION : CUSTOM_DESCRIPTION}</p>
            </div>
          </header>
          {messages}
          {t3 ? (
            <form className="plugin-form" onSubmit={addT3}>
              <label>
                T3 Code address
                <input type="url" required autoFocus placeholder="https://your-t3-code/mcp" value={t3Url}
                  disabled={busy} onChange={(e) => setT3Url(e.target.value)} />
              </label>
              <p className="field-hint">In T3 Code, open Settings → Environments and choose Copy MCP URL.</p>
              <div className="form-actions">
                <button className="primary" disabled={busy || !t3Url.trim()}>
                  {busy ? "Connecting…" : "Connect T3 Code"}
                </button>
              </div>
            </form>
          ) : (
            <form className="plugin-form" onSubmit={addCustom}>
              <label>
                Name
                <input type="text" required autoFocus placeholder="Linear, Home Assistant…" value={customName}
                  disabled={busy} onChange={(e) => setCustomName(e.target.value)} />
              </label>
              <label>
                Address
                <input type="url" required placeholder="https://example.com/mcp" value={customUrl}
                  disabled={busy} onChange={(e) => setCustomUrl(e.target.value)} />
              </label>
              <label>
                Sign-in
                <select value={customAuth} disabled={busy}
                  onChange={(e) => setCustomAuth(e.target.value as "oauth" | "bearer" | "none")}>
                  <option value="oauth">Sign in through the browser</option>
                  <option value="bearer">Paste an access key</option>
                  <option value="none">No sign-in needed</option>
                </select>
              </label>
              {customAuth === "bearer" && (
                <label>
                  Access key
                  <input type="password" required value={customToken} disabled={busy}
                    onChange={(e) => setCustomToken(e.target.value)} />
                </label>
              )}
              <div className="form-actions">
                <button className="primary" disabled={busy || !customName.trim() || !customUrl.trim()}>
                  {busy ? "Connecting…" : "Add plugin"}
                </button>
              </div>
            </form>
          )}
        </div>
      </section>
    );
  }

  if (selection.name === "server") {
    if (!selected) {
      return (
        <section className="page plugins-page" aria-label="Plugin">
          <div className="page-inner narrow">
            {back}
            {messages}
            <p className="page-empty">{loading ? "Loading plugin…" : "This plugin was removed."}</p>
          </div>
        </section>
      );
    }
    const server = selected;
    const t3 = server.preset === "t3_code";
    return (
      <section className="page plugins-page" aria-labelledby="plugin-title">
        <div className="page-inner narrow">
          {back}
          <header className="plugin-hero">
            <PluginIcon preset={server.preset} name={server.name} size={56} />
            <div>
              <h1 id="plugin-title">{server.name}</h1>
              <p>{t3 ? T3_DESCRIPTION : server.url}</p>
            </div>
            <span className={`plugin-badge ${server.enabled ? server.status : "paused"}`}>{statusLabel(server)}</span>
          </header>
          {messages}
          {!pendingAuthUrl && server.status_error && <p className="error">{server.status_error}</p>}
          <div className="plugin-actions">
            {server.status !== "connected" && (
              <button className="primary" disabled={busy} onClick={() => void run(() => connect(server))}>
                {t3 ? "Sign in to T3 Code" : "Connect"}
              </button>
            )}
            {server.app_url && (
              <a className="button" href={server.app_url} target="_blank" rel="noreferrer">
                Open {server.name} <ArrowUpRight size={14} />
              </a>
            )}
            <button disabled={busy} onClick={() => void run(async () => {
              await api<McpServer>(`/mcp/servers/${server.id}/refresh`, {}, "POST");
              return `Updated tools from ${server.name}.`;
            })}>Refresh tools</button>
            <button disabled={busy} onClick={() => void run(async () => {
              await update(server, { enabled: !server.enabled });
              return server.enabled ? `${server.name} is paused.` : `${server.name} is back on.`;
            })}>{server.enabled ? "Pause" : "Turn on"}</button>
            {server.has_credentials && (
              <button disabled={busy} onClick={() => void run(async () => {
                await api<McpServer>(`/mcp/servers/${server.id}/disconnect`, {}, "POST");
                return `Signed out of ${server.name}.`;
              })}>Sign out</button>
            )}
            <button className="danger" disabled={busy} onClick={() => void run(async () => {
              await api<void>(`/mcp/servers/${server.id}`, undefined, "DELETE");
              if (active.current) setSelection({ name: "catalog" });
              return `Removed ${server.name}.`;
            })}>Remove</button>
          </div>
          {t3 && server.tools.length > 0 && (
            <section className="plugin-section" aria-labelledby="tool-groups">
              <h2 id="tool-groups">Always ready</h2>
              <p className="field-hint">
                Turned-on groups are ready in every chat. {agentName} can still find any other {server.name} tool when it needs one.
              </p>
              <div className="plugin-toggles">
                {T3_TOOL_GROUPS.map((group) => {
                  const available = group.tools.filter((name) => server.tools.some((t) => t.name === name));
                  if (available.length === 0) return null;
                  const checked = available.every((name) => server.pinned_tools.includes(name));
                  return (
                    <label key={group.id} className="switch-row">
                      <span>{group.label}</span>
                      <input type="checkbox" role="switch" checked={checked} disabled={busy}
                        onChange={(e) => void run(() => update(server,
                          { pinned_tools: toggleGroupTools(server.pinned_tools, server.tools, group.tools, e.target.checked) })
                          .then(() => undefined))} />
                    </label>
                  );
                })}
              </div>
            </section>
          )}
          <section className="plugin-section" aria-labelledby="tool-list">
            <h2 id="tool-list">Tools <span className="count">{server.tools.length}</span></h2>
            {server.tools.length === 0 ? (
              <p className="field-hint">
                {server.status === "connected" ? "No tools yet. Try Refresh tools." : "Connect to see what this plugin can do."}
              </p>
            ) : (
              <ul className="tool-list">
                {server.tools.map((tool) => (
                  <li key={tool.name}>
                    <code>{tool.name}</code>
                    {tool.description && <span>{tool.description}</span>}
                  </li>
                ))}
              </ul>
            )}
          </section>
        </div>
      </section>
    );
  }

  const needle = query.trim().toLowerCase();
  const matches = (text: string) => !needle || text.toLowerCase().includes(needle);
  const installed = servers.filter((server) => matches(`${server.name} ${server.url}`));
  const catalog = [
    ...(!t3Server ? [{ preset: "t3_code" as const, name: "T3 Code", description: T3_DESCRIPTION }] : []),
    { preset: "custom" as const, name: "Custom plugin", description: CUSTOM_DESCRIPTION },
  ].filter((entry) => matches(`${entry.name} ${entry.description}`));

  return (
    <section className="page plugins-page" aria-labelledby="plugins-title">
      <div className="page-inner">
        <header className="page-head stacked">
          <h1 id="plugins-title">Plugins</h1>
          <p>Give {agentName} new abilities by connecting apps and services. Plugins work in text chats.</p>
        </header>
        <label className="search-field plugin-search">
          <Search size={16} />
          <span className="sr-only">Search plugins</span>
          <input type="search" placeholder="Search plugins" value={query} onChange={(e) => setQuery(e.target.value)} />
        </label>
        {messages}
        {(installed.length > 0 || (loading && !servers.length)) && (
          <>
            <h2 className="page-section-title">Installed</h2>
            {loading && !servers.length ? <p className="page-empty">Loading plugins…</p> : (
              <div className="plugin-grid">
                {installed.map((server) => (
                  <button key={server.id} className="plugin-card" onClick={() => open({ name: "server", id: server.id })}>
                    <PluginIcon preset={server.preset} name={server.name} />
                    <span className="plugin-card-text">
                      <strong>{server.name}</strong>
                      <span>{server.preset === "t3_code" ? T3_DESCRIPTION : `${server.tools.length} tools`}</span>
                    </span>
                    <span className={`plugin-badge ${server.enabled ? server.status : "paused"}`}>
                      {server.enabled && server.status === "connected" && <Check size={12} />}
                      {statusLabel(server)}
                    </span>
                  </button>
                ))}
              </div>
            )}
          </>
        )}
        <h2 className="page-section-title">Available</h2>
        {catalog.length === 0 ? <p className="page-empty">No plugins match “{query}”.</p> : (
          <div className="plugin-grid">
            {catalog.map((entry) => (
              <button key={entry.preset} className="plugin-card" onClick={() => open({ name: "setup", preset: entry.preset })}>
                <PluginIcon preset={entry.preset} name={entry.name} />
                <span className="plugin-card-text">
                  <strong>{entry.name}</strong>
                  <span>{entry.description}</span>
                </span>
                <span className="plugin-add" aria-hidden="true"><Plus size={16} /></span>
              </button>
            ))}
          </div>
        )}
      </div>
    </section>
  );
}
