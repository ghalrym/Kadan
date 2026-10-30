import { useCallback, useEffect, useRef, useState } from "react";
import { flushSync } from "react-dom";
import {
  Egg,
  FolderOpen,
  Menu,
  MessageCircle,
  Palette,
  Pencil,
  Server,
  Code2,
  RefreshCw,
  X,
} from "lucide-react";
import { agentPath, allResources, api } from "./api";
import { latestConversation } from "./agentRuns";
import type { Agent, PetJob, Resource, Status } from "./types";
import Chat from "./Chat";
import Workspace from "./Workspace";
import Hatch from "./Hatch";
import Pet, { agentPet, useMood } from "./Pet";
import { agentHue } from "./pet";
import PetEditor from "./PetEditor";
import RenameAgent from "./RenameAgent";
import RunnerAssignment from "./RunnerAssignment";
import Coding from "./Coding";
import { PetContext, useAgentPet } from "./PetContext";

/** Animate a view change, letting the agent's pet glide between layouts. */
function transition(update: () => void) {
  if (!document.startViewTransition || matchMedia("(prefers-reduced-motion: reduce)").matches)
    return update();
  document.startViewTransition(() => flushSync(update));
}

function AgentIdentity({ agent, onSelect }: { agent: Agent; onSelect: () => void }) {
  const [mood, react] = useMood("idle");
  const sheet = useAgentPet(agent.key);
  return (
    <button
      className="agent-identity"
      onPointerEnter={() => react("waving")}
      onClick={() => {
        react("jumping");
        onSelect();
      }}
    >
      <span className="agent-identity-pet">
        <Pet sheet={sheet} state={mood} size={48} look />
      </span>
      <span className="agent-identity-text">
        <strong>{agent.name}</strong>
        <code>{agent.key}</code>
      </span>
    </button>
  );
}

export default function App() {
  // Kadan has a single agent: the first one the API returns.
  const [agent, setAgent] = useState<Agent | null>(null);
  const [pets, setPets] = useState<Record<string, PetJob | null>>({});
  const [editingPet, setEditingPet] = useState(false);
  const [renaming, setRenaming] = useState(false);
  const [assigningRunner, setAssigningRunner] = useState(false);
  const [coding, setCoding] = useState(false);
  const petSelectionRevision = useRef(0);
  const petReadRevision = useRef(0);
  const pendingPetSelection = useRef(0);
  const [personas, setPersonas] = useState<Resource[]>([]);
  // The agent has one continuous conversation rather than separate threads.
  const [chain, setChain] = useState<Resource | null>(null);
  const [view, setView] = useState<"chat" | "workspace">("chat");
  const [workspaceVisited, setWorkspaceVisited] = useState(false);
  const [status, setStatus] = useState<Status | null>(null);
  const [error, setError] = useState("");
  const [petError, setPetError] = useState<{ key: string; message: string } | null>(null);
  const [petRetry, setPetRetry] = useState(0);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [menu, setMenu] = useState(false);
  const [mobile, setMobile] = useState(() => matchMedia("(max-width: 760px)").matches);
  const navigation = useRef<HTMLElement>(null);
  const openNavigation = useRef<HTMLButtonElement>(null);
  const closeNavigation = useRef<HTMLButtonElement>(null);
  const [creating, setCreating] = useState(false);
  const [hatched, setHatched] = useState(false);
  const [resourceRevision, setResourceRevision] = useState(0);
  const key = agent?.key ?? "";
  const [resourceOwner, setResourceOwner] = useState("");
  const currentAgentKey = useRef(key);
  const agentKey = currentAgentKey;
  const refreshRevision = useRef(0);
  if (currentAgentKey.current !== key) {
    currentAgentKey.current = key;
    ++petSelectionRevision.current;
    pendingPetSelection.current = 0;
  }

  const refreshPetSelection = useCallback(async (agentKey: string, current: () => boolean = () => true, reportError = false) => {
    if (pendingPetSelection.current) return false;
    const revision = petSelectionRevision.current;
    const read = ++petReadRevision.current;
    const isCurrent = () => current() && agentKey === currentAgentKey.current
      && read === petReadRevision.current && revision === petSelectionRevision.current && !pendingPetSelection.current;
    try {
      const pet = await api<PetJob | null>(`${agentPath(agentKey)}/pet`);
      if (!isCurrent()) return false;
      setPets((rows) => rows[agentKey]?.id === pet?.id && rows[agentKey]?.version === pet?.version
        ? rows : { [agentKey]: pet });
      setPetError(null);
      return true;
    } catch (error) {
      if (reportError && isCurrent()) setPetError({ key: agentKey, message: `Could not load the selected pet: ${(error as Error).message}` });
      throw error;
    }
  }, []);

  const selectPet = useCallback(async (agentKey: string, id: string | null) => {
    if (agentKey !== currentAgentKey.current) return;
    const revision = petSelectionRevision.current + 1;
    petSelectionRevision.current = revision;
    pendingPetSelection.current = revision;
    const apply = (pet: PetJob | null) => {
      if (agentKey === currentAgentKey.current && revision === petSelectionRevision.current) {
        setPets({ [agentKey]: pet });
        setPetError(null);
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
      if (pendingPetSelection.current === revision) pendingPetSelection.current = 0;
    }
  }, []);
  const refreshEditedPet = useCallback(async () => { await refreshPetSelection(key); }, [key, refreshPetSelection]);

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
  async function refreshAgent() {
    const revision = ++refreshRevision.current;
    setError("");
    setLoading(true);
    try {
      const [first] = await api<Agent[]>("/agents");
      if (revision !== refreshRevision.current) return;
      if (agentKey.current !== (first?.key ?? "")) {
        setBusy(false);
        setEditingPet(false);
        setRenaming(false);
        setAssigningRunner(false);
      }
      setAgent(first ?? null);
      setResourceRevision((revision) => revision + 1);
      if (!first) setLoading(false);
    } catch (e) {
      if (revision === refreshRevision.current) {
        setError((e as Error).message);
        setLoading(false);
      }
    }
  }
  useEffect(() => {
    void refreshAgent();
    void refreshStatus();
    const timer = setInterval(() => void refreshStatus(), 30000);
    return () => clearInterval(timer);
  }, []);
  useEffect(() => {
    ++petSelectionRevision.current;
    pendingPetSelection.current = 0;
    setPetError(null);
  }, [key]);
  useEffect(() => {
    if (!key) return;
    let current = true;
    void refreshPetSelection(key, () => current, true).catch(() => {});
    return () => { current = false; };
  }, [key, refreshPetSelection, petRetry]);
  useEffect(() => {
    if (!key) return;
    let current = true;
    const revision = refreshRevision.current;
    setLoading(true);
    Promise.all([
      api<Resource>(`${agentPath(key)}/chain`),
      allResources(key, "persona"),
    ])
      .then(([conversation, profiles]) => {
        if (current && revision === refreshRevision.current) {
          setChain((current) => current?.agent_key === key && current.id === conversation.id
            ? latestConversation(current, conversation) : conversation);
          setPersonas(profiles);
          setResourceOwner(key);
        }
      })
      .catch((e) => {
        if (current && revision === refreshRevision.current) setError(e.message);
      })
      .finally(() => {
        if (current && revision === refreshRevision.current) setLoading(false);
      });
    return () => {
      current = false;
    };
  }, [key, resourceRevision]);

  function updateChain(value: Resource) {
    if (agentKey.current !== key) return;
    setChain((current) => current ? latestConversation(current, value) : current);
  }
  function show(next: typeof view) {
    setMenu(false);
    if (next === "workspace") setWorkspaceVisited(true);
    transition(() => setView(next));
  }
  function restoreFocus() {
    if (mobile && !menu) openNavigation.current?.focus();
  }
  const disabled = busy || loading;
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
        <header className="panel-head">
          {agent ? (
            <AgentIdentity agent={agent} onSelect={() => show("chat")} />
          ) : (
            <span className="brand">
              <span className="logo" aria-hidden="true">
                <i />
                <i />
                <i />
                <i />
              </span>
              Kadan
            </span>
          )}
          <button
            className="icon mobile-only"
            aria-label="Close navigation"
            ref={closeNavigation}
            onClick={() => setMenu(false)}
          >
            <X size={18} />
          </button>
        </header>
        <nav className="panel-nav" aria-label="Views">
          <button
            className={key && view === "chat" ? "selected" : ""}
            disabled={!key}
            onClick={() => show("chat")}
          >
            <MessageCircle size={17} />
            Messages
          </button>
          <button
            className={view === "workspace" ? "selected" : ""}
            disabled={!key || disabled}
            onClick={() => show("workspace")}
          >
            <FolderOpen size={17} />
            Workspace
          </button>
          <button disabled={!key} onClick={() => { setMenu(false); setCoding(true); }}>
            <Code2 size={17} />
            Coding
          </button>
        </nav>
        <nav className="panel-nav" aria-label="Agent settings">
          <h2 className="panel-label">Agent</h2>
          <button disabled={!key} onClick={() => { setMenu(false); setEditingPet(true); }}>
            <Palette size={17} />
            Edit pet
          </button>
          <button disabled={!key || disabled} onClick={() => { setMenu(false); setRenaming(true); }}>
            <Pencil size={17} />
            Rename agent
          </button>
          <button disabled={!key || disabled} onClick={() => { setMenu(false); setAssigningRunner(true); }}>
            <Server size={17} />
            Runner assignment
          </button>
        </nav>
        <footer className="services">
          <a
            href={status?.tsuchi.browser_url ?? undefined}
            aria-disabled={!status?.tsuchi.browser_url}
            target="_blank"
            rel="noreferrer"
            title={status?.tsuchi.message ?? "Checking"}
          >
            <span className={`dot ${status?.tsuchi.state === "ready" ? "ready" : status ? "down" : ""}`} />
            Tsuchi
            <span className="sr-only">
              {!status ? "checking" : status.tsuchi.state === "ready" ? "connected" : "unavailable"}
            </span>
          </a>
          <a
            href={status?.hikari.settings_url ?? undefined}
            aria-disabled={!status?.hikari.settings_url}
            target="_blank"
            rel="noreferrer"
            title={status?.hikari.message ?? "Checking"}
          >
            <span
              className={`dot ${status?.hikari.state === "ready" ? "ready" : status?.hikari.state === "unavailable" ? "down" : status ? "warn" : ""}`}
            />
            Hikari
            <span className="sr-only">
              {!status ? "checking" : status.hikari.state === "ready" ? "ready" : "needs attention"}
            </span>
          </a>
        </footer>
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
            {view === "workspace" ? "Workspace" : agent?.name || "Kadan"}
          </h1>
          <button
            className="icon"
            aria-label="Refresh services and agent"
            title="Refresh"
            disabled={disabled}
            onClick={() => {
              void refreshAgent();
              void refreshStatus();
            }}
          >
            <RefreshCw size={16} className={loading ? "spin" : ""} />
          </button>
        </header>
        {error && (
          <div className="banner" role="alert">
            {error}
            <button disabled={disabled} onClick={() => void refreshAgent()}>
              Retry
            </button>
          </div>
        )}
        {petError?.key === key && (
          <div className="banner" role="alert">
            {petError.message}
            <button onClick={() => setPetRetry((revision) => revision + 1)}>Retry</button>
          </div>
        )}
        {key && workspaceVisited && (
          <div className="workspace-pane" hidden={view !== "workspace"}>
            <Workspace key={key} agentKey={key} browserUrl={status?.tsuchi.browser_url ?? null} />
          </div>
        )}
        {loading && !key ? (
          <div className="center-empty" role="status">
            <span className="sr-only">Loading agent</span>
            <span className="loader" aria-hidden="true">
              <i />
              <i />
              <i />
            </span>
          </div>
        ) : !key ? (
          <div className="center-empty">
            <Pet sheet={agentPet("kadan")} state="waiting" size={144} look grounded className="hero-pet" />
            <h2>Meet your agent</h2>
            <p className="center-hint">Give it a name to get started.</p>
            <button className="primary" disabled={!!error} onClick={() => setCreating(true)}>
              <Egg size={16} />
              Hatch your agent
            </button>
          </div>
        ) : view === "workspace" ? null : chain && resourceOwner === key ? (
          <Chat
            key={`${key}:${chain.id}`}
            agentKey={key}
            agentName={agent?.name || key}
            chain={chain}
            personas={personas}
            status={status}
            entrance={hatched ? "jumping" : "waving"}
            onChange={updateChain}
            onBusy={(value) => { if (agentKey.current === key) setBusy(value); }}
            onEditPet={() => setEditingPet(true)}
            blocked={loading}
            refreshStatus={refreshStatus}
          />
        ) : (
          <div className="center-empty" role="status">
            <span className="sr-only">Loading messages</span>
            <span className="loader" aria-hidden="true">
              <i />
              <i />
              <i />
            </span>
          </div>
        )}
      </main>
      {renaming && agent && <RenameAgent
        agent={agent}
        onUpdated={(saved) => setAgent((current) => current?.key === saved.key ? { ...current, name: saved.name } : current)}
        onClose={() => {
          setRenaming(false);
          restoreFocus();
        }}
      />}
      {coding && <Coding onClose={() => { setCoding(false); restoreFocus(); }} />}
      {assigningRunner && agent && <RunnerAssignment
        agent={agent}
        onClose={() => {
          setAssigningRunner(false);
          restoreFocus();
        }}
      />}
      {editingPet && agent && <PetEditor
        agentKey={key}
        agentName={agent.name}
        selected={pets[key] ?? null}
        onSelect={(id) => selectPet(key, id)}
        onRefreshSelection={refreshEditedPet}
        onClose={() => setEditingPet(false)}
      />}
      {!agent && <Hatch
        open={creating}
        onClose={() => {
          setCreating(false);
          restoreFocus();
        }}
        onHatched={(created) => {
          ++refreshRevision.current;
          setHatched(true);
          setCreating(false);
          transition(() => {
            setAgent(created);
            setView("chat");
          });
        }}
      />}
    </div>
    </PetContext.Provider>
  );
}
