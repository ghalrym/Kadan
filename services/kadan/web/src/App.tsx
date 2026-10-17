import { useCallback, useEffect, useRef, useState } from "react";
import { flushSync } from "react-dom";
import {
  BookUser,
  FolderOpen,
  Menu,
  Plus,
  Palette,
  RefreshCw,
  SquarePen,
  X,
} from "lucide-react";
import { agentPath, allResources, api } from "./api";
import type { Agent, PetJob, Resource, Status } from "./types";
import Chat, { type Opening } from "./Chat";
import Home from "./Home";
import Workspace from "./Workspace";
import Personas from "./Personas";
import Hatch from "./Hatch";
import Pet, { agentPet, useMood } from "./Pet";
import { agentHue } from "./pet";
import PetEditor from "./PetEditor";
import { PetContext, useAgentPet } from "./PetContext";

/** Animate a view change, letting the agent's pet glide between layouts. */
function transition(update: () => void) {
  if (!document.startViewTransition || matchMedia("(prefers-reduced-motion: reduce)").matches)
    return update();
  document.startViewTransition(() => flushSync(update));
}

function ago(value: string) {
  const minutes = Math.round((Date.now() - new Date(value).getTime()) / 60000);
  if (minutes < 1) return "now";
  if (minutes < 60) return `${minutes}m`;
  if (minutes < 1440) return `${Math.round(minutes / 60)}h`;
  if (minutes < 10080) return `${Math.round(minutes / 1440)}d`;
  return new Date(value).toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

function RosterPet({
  agent,
  selected,
  disabled,
  onSelect,
}: {
  agent: Agent;
  selected: boolean;
  disabled: boolean;
  onSelect: () => void;
}) {
  const [mood, react] = useMood("idle");
  const sheet = useAgentPet(agent.key);
  return (
    <button
      className={`roster-pet ${selected ? "selected" : ""}`}
      style={{ "--agent-h": agentHue(agent.key) } as React.CSSProperties}
      aria-pressed={selected}
      aria-label={agent.name}
      title={agent.name}
      disabled={disabled}
      onPointerEnter={() => react("waving")}
      onClick={() => {
        react("jumping");
        onSelect();
      }}
    >
      <Pet sheet={sheet} state={mood} size={48} look />
    </button>
  );
}

export default function App() {
  const [agents, setAgents] = useState<Agent[]>([]);
  const [pets, setPets] = useState<Record<string, PetJob | null>>({});
  const [editingPet, setEditingPet] = useState(false);
  const petSelectionRevision = useRef<Record<string, number>>({});
  const pendingPetSelection = useRef<Record<string, number>>({});
  const [key, setKey] = useState(localStorage.getItem("kadan.agent") || "");
  const [threads, setThreads] = useState<Resource[]>([]);
  const [personas, setPersonas] = useState<Resource[]>([]);
  const [thread, setThread] = useState<Resource | null>(null);
  const [opening, setOpening] = useState<Opening | null>(null);
  const [view, setView] = useState<"chat" | "workspace" | "personas">("chat");
  const [status, setStatus] = useState<Status | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [menu, setMenu] = useState(false);
  const [mobile, setMobile] = useState(() => matchMedia("(max-width: 760px)").matches);
  const navigation = useRef<HTMLElement>(null);
  const openNavigation = useRef<HTMLButtonElement>(null);
  const closeNavigation = useRef<HTMLButtonElement>(null);
  const [creating, setCreating] = useState(false);
  const [hatched, setHatched] = useState("");
  const [creatingThread, setCreatingThread] = useState(false);
  const [resourceRevision, setResourceRevision] = useState(0);
  const activeAgent = useRef(key);
  activeAgent.current = key;
  const selectedAgent = agents.find((agent) => agent.key === key);

  const refreshPetSelection = useCallback(async (agentKey: string) => {
    if (pendingPetSelection.current[agentKey]) return;
    const revision = petSelectionRevision.current[agentKey] ?? 0;
    const pet = await api<PetJob | null>(`${agentPath(agentKey)}/pet`);
    if (revision === (petSelectionRevision.current[agentKey] ?? 0) && !pendingPetSelection.current[agentKey]) {
      setPets((rows) => rows[agentKey]?.id === pet?.id && rows[agentKey]?.version === pet?.version
        ? rows : { ...rows, [agentKey]: pet });
    }
  }, []);

  const selectPet = useCallback(async (agentKey: string, id: string | null) => {
    const revision = (petSelectionRevision.current[agentKey] ?? 0) + 1;
    petSelectionRevision.current[agentKey] = revision;
    pendingPetSelection.current[agentKey] = revision;
    const apply = (pet: PetJob | null) => {
      if (revision === petSelectionRevision.current[agentKey]) {
        setPets((rows) => ({ ...rows, [agentKey]: pet }));
      }
    };
    try {
      try {
        apply(await api<PetJob | null>(`${agentPath(agentKey)}/pet`, { pet_id: id }, "PUT"));
      } catch (error) {
        // The write may have committed even when its response never arrived.
        const saved = await api<PetJob | null>(`${agentPath(agentKey)}/pet`);
        apply(saved);
        if ((saved?.id ?? null) !== id) throw error;
      }
    } finally {
      if (pendingPetSelection.current[agentKey] === revision) delete pendingPetSelection.current[agentKey];
    }
  }, []);
  const refreshEditedPet = useCallback(() => refreshPetSelection(key), [key, refreshPetSelection]);


  useEffect(() => {
    const query = matchMedia("(max-width: 760px)");
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
      if (matchMedia("(max-width: 760px)").matches) openNavigation.current?.focus();
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
    let current = true;
    for (const agent of agents) {
      refreshPetSelection(agent.key)
        .catch((error: Error) => {
          if (current) setError(`Could not load ${agent.name}’s selected pet: ${error.message}`);
        });
    }
    return () => { current = false; };
  }, [agents, refreshPetSelection]);
  useEffect(() => {
    setEditingPet(false);
    localStorage.setItem("kadan.agent", key);
    setThread(null);
    setThreads([]);
    setPersonas([]);
    setOpening(null);
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

  function updateThread(value: Resource) {
    if (value.agent_key !== activeAgent.current) return;
    setThread((current) =>
      current && current.id !== value.id ? current : value,
    );
    setThreads((rows) => [value, ...rows.filter((row) => row.id !== value.id)]);
  }
  function show(next: typeof view) {
    setMenu(false);
    transition(() => setView(next));
  }
  async function openThread(value: Resource) {
    setError("");
    setMenu(false);
    setBusy(true);
    try {
      const loaded = await api<Resource>(`${agentPath(key)}/resources/${value.id}`);
      transition(() => {
        setView("chat");
        setThread(loaded);
      });
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  function goHome() {
    setMenu(false);
    transition(() => {
      setView("chat");
      setThread(null);
    });
  }
  async function createThread(first: Opening) {
    if (!key || creatingThread || busy || loading) return;
    setCreatingThread(true);
    setError("");
    try {
      const value = await api<Resource>(`${agentPath(key)}/resources`, {
        kind: "conversation",
        title: `Thread · ${new Date().toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })}`,
      });
      transition(() => {
        setOpening(first);
        setThread(value);
        updateThread(value);
        setView("chat");
      });
      setMenu(false);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setCreatingThread(false);
    }
  }
  const disabled = busy || creatingThread || loading;
  const hue = key ? agentHue(key) : 252;
  return (
    <PetContext.Provider value={pets}>
    <div className="app" style={{ "--agent-h": hue } as React.CSSProperties}>
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
              'button:not(:disabled), select:not(:disabled), a[href]',
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
        <div className="rail">
          <span className="logo" aria-hidden="true">
            <i />
            <i />
            <i />
            <i />
          </span>
          <nav className="roster" aria-label="Agents">
            {agents.map((agent) => (
              <RosterPet
                key={agent.key}
                agent={agent}
                selected={agent.key === key}
                disabled={disabled}
                onSelect={() => {
                  if (agent.key !== key) transition(() => setKey(agent.key));
                  else goHome();
                }}
              />
            ))}
            <button
              className="roster-add"
              aria-label="Hatch agent"
              title="Hatch agent"
              disabled={disabled}
              onClick={() => setCreating(true)}
            >
              <Plus size={20} />
            </button>
          </nav>
          <div className="services">
            <a
              href={status?.tsuchi.browser_url ?? undefined}
              aria-disabled={!status?.tsuchi.browser_url}
              target="_blank"
              rel="noreferrer"
              title={`Tsuchi · ${status?.tsuchi.message ?? "Checking"}`}
            >
              <span className={`dot ${status?.tsuchi.state === "ready" ? "ready" : status ? "down" : ""}`} />
              <span className="sr-only">
                Tsuchi {status?.tsuchi.state === "ready" ? "connected" : "unavailable"}
              </span>
            </a>
            <a
              href={status?.hikari.settings_url ?? undefined}
              aria-disabled={!status?.hikari.settings_url}
              target="_blank"
              rel="noreferrer"
              title={`Hikari · ${status?.hikari.message ?? "Checking"}`}
            >
              <span
                className={`dot ${status?.hikari.state === "ready" ? "ready" : status?.hikari.state === "unavailable" ? "down" : status ? "warn" : ""}`}
              />
              <span className="sr-only">
                Hikari {status?.hikari.state === "ready" ? "ready" : "needs attention"}
              </span>
            </a>
          </div>
        </div>
        <div className="panel">
          <header className="panel-head">
            <button className="agent-title" disabled={!key} onClick={goHome}>
              <strong>{selectedAgent?.name || "Kadan"}</strong>
              {selectedAgent && <code>{selectedAgent.key}</code>}
            </button>
            <button
              className="icon mobile-only"
              aria-label="Close navigation"
              ref={closeNavigation}
              onClick={() => setMenu(false)}
            >
              <X size={18} />
            </button>
          </header>
          <nav className="panel-nav" aria-label="Agent views">
            <button disabled={!key} onClick={() => { setMenu(false); setEditingPet(true); }}>
              <Palette size={17} />
              Edit pet
            </button>
            <button
              className={key && view === "chat" && !thread ? "selected" : ""}
              disabled={disabled || !key}
              onClick={goHome}
            >
              <SquarePen size={17} />
              New thread
            </button>
            <button
              className={view === "workspace" ? "selected" : ""}
              disabled={!key || disabled}
              onClick={() => show("workspace")}
            >
              <FolderOpen size={17} />
              Workspace
            </button>
            <button
              className={view === "personas" ? "selected" : ""}
              disabled={!key || disabled}
              onClick={() => show("personas")}
            >
              <BookUser size={17} />
              Personas
            </button>
          </nav>
          <div className="thread-list" aria-label="Threads" role="group">
            <h3>Threads</h3>
            {loading && key ? (
              <div className="skeleton" role="status">
                <span className="sr-only">Loading threads</span>
                <i />
                <i />
                <i />
              </div>
            ) : threads.length === 0 ? (
              <p className="quiet">None yet</p>
            ) : (
              threads.map((item, index) => (
                <button
                  title={item.title}
                  key={item.id}
                  disabled={disabled}
                  style={{ "--i": Math.min(index, 12) } as React.CSSProperties}
                  className={`thread ${view === "chat" && thread?.id === item.id ? "selected" : ""}`}
                  onClick={() => void openThread(item)}
                >
                  <span>{item.title}</span>
                  <time dateTime={item.updated_at}>{ago(item.updated_at)}</time>
                </button>
              ))
            )}
          </div>
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
          <h1>
            {view === "workspace"
              ? "Workspace"
              : view === "personas"
                ? "Personas"
                : thread?.title || selectedAgent?.name || "Kadan"}
          </h1>
          <button
            className="icon"
            aria-label="Refresh services and agents"
            title="Refresh"
            disabled={disabled}
            onClick={() => {
              void refreshAgents();
              void refreshStatus();
            }}
          >
            <RefreshCw size={16} className={loading ? "spin" : ""} />
          </button>
        </header>
        {error && (
          <div className="banner" role="alert">
            {error}
            <button disabled={disabled} onClick={() => void refreshAgents()}>
              Retry
            </button>
          </div>
        )}
        {loading && !key ? (
          <div className="center-empty" role="status">
            <span className="sr-only">Loading agents</span>
            <span className="loader" aria-hidden="true">
              <i />
              <i />
              <i />
            </span>
          </div>
        ) : !key ? (
          <div className="center-empty">
            <Pet sheet={agentPet("kadan")} state="waiting" size={144} look grounded className="hero-pet" />
            <h2>No agents yet</h2>
            <button className="primary" onClick={() => setCreating(true)}>
              <Plus size={16} />
              Hatch agent
            </button>
          </div>
        ) : view === "workspace" ? (
          <Workspace key={key} agentKey={key} browserUrl={status?.tsuchi.browser_url ?? null} />
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
            agentName={selectedAgent?.name || key}
            thread={thread}
            personas={personas}
            status={status}
            opening={opening}
            onOpened={() => setOpening(null)}
            onChange={updateThread}
            onBusy={setBusy}
            blocked={loading}
            refreshStatus={refreshStatus}
          />
        ) : (
          <Home
            key={key}
            agentKey={key}
            agentName={selectedAgent?.name || key}
            entrance={hatched === key ? "jumping" : "waving"}
            personas={personas}
            status={status}
            disabled={disabled}
            onStart={(first) => void createThread(first)}
            onEditPet={() => setEditingPet(true)}
          />
        )}
      </main>
      {editingPet && selectedAgent && <PetEditor
        key={key}
        agentKey={key}
        agentName={selectedAgent.name}
        selected={pets[key] ?? null}
        onSelect={(id) => selectPet(key, id)}
        onRefreshSelection={refreshEditedPet}
        onClose={() => setEditingPet(false)}
      />}
      <Hatch
        open={creating}
        onClose={() => {
          setCreating(false);
          if (mobile && !menu) openNavigation.current?.focus();
        }}
        onHatched={(agent) => {
          setAgents((rows) => [...rows, agent]);
          setHatched(agent.key);
          setCreating(false);
          transition(() => {
            setKey(agent.key);
            setView("chat");
          });
        }}
      />
    </div>
    </PetContext.Provider>
  );
}
