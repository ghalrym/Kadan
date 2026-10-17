import { useEffect, useRef, useState } from "react";
import {
  ArrowUpRight,
  Check,
  ChevronDown,
  FolderOpen,
  Menu,
  MessageSquare,
  Plus,
  RefreshCw,
  Settings2,
  SquarePen,
  X,
} from "lucide-react";
import { agentPath, allResources, api } from "./api";
import type { Agent, Resource, Status } from "./types";
import Chat from "./Chat";
import Workspace from "./Workspace";
import Personas from "./Personas";
import AgentAvatar from "./AgentAvatar";
import AvatarEditor from "./AvatarEditor";
import { activityLabel } from "./avatar";
import type { AvatarActivity } from "./avatar";

export default function App() {
  const [agents, setAgents] = useState<Agent[]>([]);
  const [key, setKey] = useState(localStorage.getItem("kadan.agent") || "");
  const [threads, setThreads] = useState<Resource[]>([]);
  const [personas, setPersonas] = useState<Resource[]>([]);
  const [thread, setThread] = useState<Resource | null>(null);
  const [view, setView] = useState<"chat" | "workspace" | "personas">("chat");
  const [status, setStatus] = useState<Status | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [activity, setActivity] = useState<AvatarActivity>("idle");
  const [editingAvatar, setEditingAvatar] = useState(false);
  const [avatarEditors, setAvatarEditors] = useState<string[]>([]);
  const [generatingAvatars, setGeneratingAvatars] = useState<string[]>([]);
  const [menu, setMenu] = useState(false);
  const [mobile, setMobile] = useState(
    () => matchMedia("(max-width: 700px)").matches,
  );
  const navigation = useRef<HTMLElement>(null);
  const openNavigation = useRef<HTMLButtonElement>(null);
  const closeNavigation = useRef<HTMLButtonElement>(null);
  const [creating, setCreating] = useState(false);
  const [agentError, setAgentError] = useState("");
  const [savingAgent, setSavingAgent] = useState(false);
  const [creatingThread, setCreatingThread] = useState(false);
  const [resourceRevision, setResourceRevision] = useState(0);
  const activeAgent = useRef(key);
  activeAgent.current = key;
  const dialog = useRef<HTMLDialogElement>(null);
  const selectedAgent = agents.find((agent) => agent.key === key);

  useEffect(() => {
    const query = matchMedia("(max-width: 700px)");
    const changed = () => {
      const focused = document.activeElement;
      if (query.matches && navigation.current?.contains(focused)) {
        openNavigation.current?.focus();
      } else if (!query.matches && focused === openNavigation.current) {
        document.getElementById("main")?.focus();
      }
      setMobile(query.matches);
      if (!query.matches) setMenu(false);
    };
    query.addEventListener("change", changed);
    return () => query.removeEventListener("change", changed);
  }, []);
  useEffect(() => {
    if (!menu || !mobile) return;
    closeNavigation.current?.focus();
    return () => {
      if (matchMedia("(max-width: 700px)").matches)
        openNavigation.current?.focus();
      else document.getElementById("main")?.focus();
    };
  }, [menu, mobile]);

  async function refreshStatus() {
    try {
      setStatus(await api<Status>("/status"));
    } catch {
      setStatus((current) => ({
        tsuchi: {
          state: "unavailable",
          message: "Kadan cannot reach its services.",
          browser_url: current?.tsuchi.browser_url ?? null,
        },
        hikari: {
          state: "unavailable",
          message: "Kadan cannot reach Hikari.",
          tier: null,
          model_id: null,
          settings_url: current?.hikari.settings_url ?? null,
        },
      }));
    }
  }
  async function refreshAgents() {
    setError("");
    setLoading(true);
    try {
      const rows = await api<Agent[]>("/agents");
      setAgents(rows);
      setKey((current) =>
        rows.some((a) => a.key === current) ? current : rows[0]?.key || "",
      );
      setResourceRevision((revision) => revision + 1);
      if (!rows.length) setLoading(false);
    } catch (e) {
      setError((e as Error).message);
      setLoading(false);
    }
  }
  useEffect(() => {
    void refreshAgents();
    void refreshStatus();
    const timer = setInterval(() => void refreshStatus(), 30000);
    return () => clearInterval(timer);
  }, []);
  useEffect(() => {
    localStorage.setItem("kadan.agent", key);
    setThread(null);
    setActivity("idle");
    setEditingAvatar(false);
    setThreads([]);
    setPersonas([]);
    setError("");
  }, [key]);
  useEffect(() => {
    if (!key) return;
    let current = true;
    setLoading(true);
    Promise.all([
      allResources(key, "conversation"),
      allResources(key, "persona"),
    ])
      .then(([conversations, profiles]) => {
        if (current) {
          setThreads(conversations);
          setPersonas(profiles);
          setThread((current) =>
            current
              ? conversations.find((item) => item.id === current.id) || null
              : null,
          );
        }
      })
      .catch((e) => {
        if (current) setError(e.message);
      })
      .finally(() => {
        if (current) setLoading(false);
      });
    return () => {
      current = false;
    };
  }, [key, resourceRevision]);
  useEffect(() => {
    if (creating) dialog.current?.showModal();
    else dialog.current?.close();
  }, [creating]);

  function updateThread(value: Resource) {
    if (value.agent_key !== activeAgent.current) return;
    setThread((current) =>
      current && current.id !== value.id ? current : value,
    );
    setThreads((rows) => [value, ...rows.filter((row) => row.id !== value.id)]);
  }
  async function openThread(value: Resource) {
    setError("");
    setMenu(false);
    setView("chat");
    setBusy(true);
    try {
      setThread(await api<Resource>(`${agentPath(key)}/resources/${value.id}`));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  async function createThread() {
    if (!key || creatingThread || busy || loading) return;
    setCreatingThread(true);
    setError("");
    try {
      const value = await api<Resource>(`${agentPath(key)}/resources`, {
        kind: "conversation",
        title: `Thread · ${new Date().toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })}`,
      });
      setThread(value);
      updateThread(value);
      setView("chat");
      setMenu(false);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setCreatingThread(false);
    }
  }
  async function createAgent(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const data = new FormData(event.currentTarget);
    setSavingAgent(true);
    setAgentError("");
    try {
      const agent = await api<Agent>("/agents", {
        name: data.get("name"),
        key: data.get("key"),
      });
      setAgents((rows) => [...rows, agent]);
      setKey(agent.key);
      setView("chat");
      setCreating(false);
    } catch (e) {
      setAgentError((e as Error).message);
    } finally {
      setSavingAgent(false);
    }
  }
  function openAvatar() {
    setAvatarEditors((keys) => (keys.includes(key) ? keys : [...keys, key]));
    setEditingAvatar(true);
  }
  const generatingAvatar = generatingAvatars.includes(key);
  const disabled = busy || creatingThread || loading;
  const avatarActivity =
    activity !== "idle"
      ? activity
      : busy || creatingThread || generatingAvatar
        ? "working"
        : "idle";
  return (
    <div className="app">
      <a className="skip-link" href="#main" inert={mobile && menu}>
        Skip to content
      </a>
      {menu && (
        <button
          className="scrim"
          tabIndex={-1}
          aria-label="Close navigation"
          onClick={() => setMenu(false)}
        />
      )}
      <aside
        id="navigation"
        ref={navigation}
        inert={mobile && !menu}
        role={mobile && menu ? "dialog" : undefined}
        aria-modal={mobile && menu ? true : undefined}
        onKeyDown={(event) => {
          if (!mobile || !menu) return;
          if (event.key === "Escape") {
            event.preventDefault();
            setMenu(false);
          } else if (event.key === "Tab") {
            const controls = navigation.current?.querySelectorAll<HTMLElement>(
              "button:not(:disabled), select:not(:disabled), a[href]",
            );
            const first = controls?.[0];
            const last = controls?.[controls.length - 1];
            if (event.shiftKey && document.activeElement === first) {
              event.preventDefault();
              last?.focus();
            } else if (!event.shiftKey && document.activeElement === last) {
              event.preventDefault();
              first?.focus();
            }
          }
        }}
        className={`sidebar ${menu ? "is-open" : ""}`}
        aria-label="Navigation"
      >
        <div className="brand">
          <span className="brand-mark">
            <span />
            <span />
            <span />
          </span>
          <strong>Kadan</strong>
          <button
            className="icon mobile-only"
            aria-label="Close navigation"
            ref={closeNavigation}
            onClick={() => setMenu(false)}
          >
            <X size={18} />
          </button>
        </div>
        <div className="agent-control">
          <label className="sr-only" htmlFor="agent">
            Agent
          </label>
          <select
            id="agent"
            value={key}
            disabled={disabled || !agents.length}
            onChange={(event) => setKey(event.target.value)}
          >
            {!agents.length && <option value="">No agents</option>}
            {agents.map((agent) => (
              <option value={agent.key} key={agent.key}>
                {agent.name}
              </option>
            ))}
          </select>
          <ChevronDown size={14} className="select-chevron" />
          <button
            className="icon"
            aria-label="Create agent"
            disabled={disabled}
            onClick={() => {
              setAgentError("");
              setCreating(true);
            }}
          >
            <Plus size={18} />
          </button>
        </div>
        <button
          className="nav-button new-thread"
          disabled={disabled || !key}
          onClick={() => void createThread()}
        >
          <SquarePen size={17} />
          {creatingThread ? "Creating thread…" : "New thread"}
        </button>
        <nav aria-label="Agent views">
          <button
            className={`nav-button ${view === "workspace" ? "selected" : ""}`}
            disabled={!key || disabled}
            onClick={() => {
              setView("workspace");
              setMenu(false);
            }}
          >
            <FolderOpen size={17} />
            Workspace
          </button>
          <button
            className={`nav-button ${view === "personas" ? "selected" : ""}`}
            disabled={!key || disabled}
            onClick={() => {
              setView("personas");
              setMenu(false);
            }}
          >
            <Settings2 size={17} />
            Personas
          </button>
        </nav>
        <div className="thread-list" aria-label="Threads">
          {loading && key ? (
            <p className="quiet loading">Loading threads…</p>
          ) : threads.length === 0 ? (
            <p className="quiet empty-sidebar">No threads yet</p>
          ) : (
            threads.map((item) => (
              <button
                title={item.title}
                key={item.id}
                disabled={disabled}
                className={`thread-button ${view === "chat" && thread?.id === item.id ? "selected" : ""}`}
                onClick={() => void openThread(item)}
              >
                <MessageSquare size={15} />
                <span>{item.title}</span>
              </button>
            ))
          )}
        </div>
        <div className="sidebar-bottom">
          <a
            href={status?.tsuchi.browser_url ?? undefined}
            aria-disabled={!status?.tsuchi.browser_url}
            target="_blank"
            rel="noreferrer"
          >
            <span
              className={`status-dot ${status?.tsuchi.state === "ready" ? "ready" : ""}`}
            />
            <span>Tsuchi</span>
            <span className="quiet">
              {status?.tsuchi.state === "ready" ? "Connected" : "Unavailable"}
            </span>
            <ArrowUpRight size={14} />
          </a>
          <a
            href={status?.hikari.settings_url ?? undefined}
            aria-disabled={!status?.hikari.settings_url}
            target="_blank"
            rel="noreferrer"
          >
            <span
              className={`status-dot ${status?.hikari.state === "ready" ? "ready" : ""}`}
            />
            <span>Hikari</span>
            <span className="quiet">
              {status?.hikari.state === "ready"
                ? "Ready"
                : status?.hikari.state === "tier_required"
                  ? "Choose tier"
                  : "Unavailable"}
            </span>
            <ArrowUpRight size={14} />
          </a>
        </div>
      </aside>
      <main id="main" tabIndex={-1} inert={mobile && menu}>
        <header className="topbar">
          <button
            className="icon mobile-only"
            aria-label="Open navigation"
            ref={openNavigation}
            aria-expanded={menu}
            aria-controls="navigation"
            onClick={() => setMenu(true)}
          >
            <Menu size={20} />
          </button>
          {selectedAgent && (
            <button
              className="avatar-trigger"
              aria-label={`Edit avatar for ${selectedAgent.name}`}
              title="Edit avatar"
              disabled={disabled}
              onClick={openAvatar}
            >
              <AgentAvatar
                agentKey={key}
                name={selectedAgent.name}
                assetId={selectedAgent.avatar_asset_id}
                activity={avatarActivity}
                follow={!editingAvatar}
              />
            </button>
          )}
          <div className="breadcrumb">
            <span>{selectedAgent?.name || "Kadan"}</span>
            <span className="separator">/</span>
            <h1>
              {view === "workspace"
                ? "Workspace"
                : view === "personas"
                  ? "Personas"
                  : thread?.title || "Threads"}
            </h1>
          </div>
          {selectedAgent && (
            <span className="agent-activity-label" role="status">
              {generatingAvatar && activity === "idle" && !busy
                ? "Generating avatar"
                : activityLabel[avatarActivity]}
            </span>
          )}
          <button
            className="icon refresh"
            aria-label="Refresh services and agents"
            disabled={disabled}
            onClick={() => {
              void refreshAgents();
              void refreshStatus();
            }}
          >
            <RefreshCw size={16} />
          </button>
        </header>
        {error && (
          <div className="error banner" role="alert">
            {error}
            <button disabled={disabled} onClick={() => void refreshAgents()}>
              Retry
            </button>
          </div>
        )}
        {loading && !key ? (
          <div className="center-empty" role="status">
            Loading agents…
          </div>
        ) : !key ? (
          <div className="center-empty">
            <div className="empty-icon">
              <MessageSquare size={28} />
            </div>
            <h2>No agents yet</h2>
            <button className="primary" onClick={() => setCreating(true)}>
              <Plus size={16} />
              Create agent
            </button>
          </div>
        ) : view === "workspace" ? (
          <Workspace
            key={key}
            agentKey={key}
            browserUrl={status?.tsuchi.browser_url ?? null}
          />
        ) : view === "personas" ? (
          <Personas
            key={key}
            agentKey={key}
            personas={personas}
            onChange={(rows) => {
              if (activeAgent.current === key) setPersonas(rows);
            }}
            onBusy={setBusy}
            blocked={loading}
          />
        ) : thread ? (
          <Chat
            key={`${key}:${thread.id}`}
            agentKey={key}
            thread={thread}
            personas={personas}
            status={status}
            onChange={updateThread}
            onBusy={setBusy}
            blocked={loading}
            refreshStatus={refreshStatus}
            onActivity={setActivity}
          />
        ) : (
          <div className="center-empty">
            {selectedAgent && (
              <button
                className="avatar-trigger avatar-empty-trigger"
                aria-label={`Edit avatar for ${selectedAgent.name}`}
                disabled={disabled}
                onClick={openAvatar}
              >
                <AgentAvatar
                  agentKey={key}
                  name={selectedAgent.name}
                  assetId={selectedAgent.avatar_asset_id}
                  activity={avatarActivity}
                  size="large"
                  follow={!editingAvatar}
                />
              </button>
            )}
            <h2>No thread selected</h2>
            <button
              className="primary"
              onClick={() => void createThread()}
              disabled={disabled}
            >
              <SquarePen size={16} />
              New thread
            </button>
          </div>
        )}
      </main>
      {agents
        .filter((agent) => avatarEditors.includes(agent.key))
        .map((agent) => (
          <AvatarEditor
            key={agent.key}
            agent={agent}
            open={editingAvatar && agent.key === key}
            browserUrl={status?.tsuchi.browser_url ?? null}
            onChange={(updated) =>
              setAgents((rows) =>
                rows.map((row) => (row.key === updated.key ? updated : row)),
              )
            }
            onClose={() => setEditingAvatar(false)}
            onGenerating={(generating) =>
              setGeneratingAvatars((keys) =>
                generating
                  ? [...keys.filter((value) => value !== agent.key), agent.key]
                  : keys.filter((value) => value !== agent.key),
              )
            }
          />
        ))}
      <dialog
        ref={dialog}
        onClose={() => {
          if (mobile && !menu) openNavigation.current?.focus();
        }}
        onCancel={(event) => {
          if (savingAgent) event.preventDefault();
          else setCreating(false);
        }}
        aria-labelledby="create-agent-title"
      >
        <form onSubmit={createAgent}>
          <div className="dialog-header">
            <h2 id="create-agent-title">Create agent</h2>
            <button
              type="button"
              className="icon"
              aria-label="Close"
              disabled={savingAgent}
              onClick={() => setCreating(false)}
            >
              <X size={18} />
            </button>
          </div>
          <label>
            Name
            <input
              name="name"
              autoFocus
              required
              maxLength={120}
              placeholder="Research"
            />
          </label>
          <label>
            Agent key
            <input
              name="key"
              required
              pattern="[a-z0-9][a-z0-9_-]{0,63}"
              maxLength={64}
              placeholder="research"
              autoCapitalize="none"
              autoComplete="off"
              spellCheck={false}
            />
          </label>
          <p className="field-hint">
            Lowercase letters, numbers, hyphens and underscores. The key cannot
            be changed after creation.
          </p>
          {agentError && (
            <p className="error" role="alert">
              {agentError}
            </p>
          )}
          <div className="form-actions">
            <button
              type="button"
              disabled={savingAgent}
              onClick={() => setCreating(false)}
            >
              Cancel
            </button>
            <button className="primary" disabled={savingAgent}>
              <Check size={16} />
              {savingAgent ? "Creating…" : "Create agent"}
            </button>
          </div>
        </form>
      </dialog>
    </div>
  );
}
