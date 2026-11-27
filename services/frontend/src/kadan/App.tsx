import { useCallback, useEffect, useRef, useState } from "react";
import { flushSync } from "react-dom";
import { Egg, Menu } from "lucide-react";
import { agentPath, api } from "./api";
import { getDesktopServerConfig, isDesktopApp } from "./desktop";
import { latestConversation } from "./agentRuns";
import { emptyConversation, pickConversation, summarize, upsertConversation } from "./conversations";
import type { Agent, ConversationSummary, PetJob, Resource, Status } from "./types";
import Chat from "./Chat";
import Hatch from "./Hatch";
import Pet, { agentPet } from "./Pet";
import { agentHue } from "./pet";
import PetEditor from "./PetEditor";
import PersonaEditor from "./PersonaEditor";
import RenameAgent from "./RenameAgent";
import RunnerAssignment from "./RunnerAssignment";
import Plugins from "./Plugins";
import Sidebar, { type Page } from "./Sidebar";
import SpaceLibrary from "./SpaceLibrary";
import ConfirmDialog from "./ConfirmDialog";
import AgentPage from "./AgentPage";
import ServerSettings from "./ServerSettings";
import { PetContext } from "./PetContext";

/** Animate a view change, letting the agent's pet glide between layouts. */
function transition(update: () => void) {
  if (!document.startViewTransition || matchMedia("(prefers-reduced-motion: reduce)").matches)
    return update();
  document.startViewTransition(() => flushSync(update));
}

export default function App() {
  // Kadan has a single agent: the first one the API returns.
  const [agent, setAgent] = useState<Agent | null>(null);
  const [pets, setPets] = useState<Record<string, PetJob | null>>({});
  const [editingPet, setEditingPet] = useState(false);
  const [editingPersona, setEditingPersona] = useState(false);
  const [renaming, setRenaming] = useState(false);
  const [assigningRunner, setAssigningRunner] = useState(false);
  const [confirm, setConfirm] = useState<{ title: string; message: string; confirm: string; run: () => Promise<void> } | null>(null);
  const petSelectionRevision = useRef(0);
  const petReadRevision = useRef(0);
  const pendingPetSelection = useRef(0);
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  /** The open chat, remembered with the agent it belongs to. */
  const [selection, setSelection] = useState<{ owner: string; id: string } | null>(null);
  const [chain, setChain] = useState<Resource | null>(null);
  const [page, setPage] = useState<Page>({ name: "chat" });
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
  const [serverSettingsOpen, setServerSettingsOpen] = useState(false);
  const [desktopHost, setDesktopHost] = useState("");
  const desktopMode = isDesktopApp();
  const currentAgentKey = useRef(key);
  const agentKey = currentAgentKey;
  const refreshRevision = useRef(0);
  const selectedRef = useRef(selection);
  selectedRef.current = selection;
  const selectedId = selection?.owner === key ? selection.id : null;
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
    if (!desktopMode) return;
    void getDesktopServerConfig()
      .then((cfg) => setDesktopHost(cfg.active_host))
      .catch(() => {});
  }, [desktopMode]);

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
      setStatus({
        tsuchi: { state: "unavailable", message: "Kadan cannot reach its services." },
        hikari: { state: "unavailable", message: "Kadan cannot reach Hikari.", model_id: null },
      });
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
        setEditingPersona(false);
        setRenaming(false);
        setAssigningRunner(false);
        setConfirm(null);
        setPage({ name: "chat" });
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
  // Chats load first; the open chat loads separately below.
  useEffect(() => {
    if (!key) return;
    let current = true;
    const revision = refreshRevision.current;
    const isCurrent = () => current && revision === refreshRevision.current;
    setLoading(true);
    api<ConversationSummary[]>(`${agentPath(key)}/conversations`)
      .then(async (rows) => {
        if (!isCurrent()) return;
        const preferred = selectedRef.current?.owner === key ? selectedRef.current.id : null;
        let id = pickConversation(rows, preferred);
        if (!id) {
          const created = await api<Resource>(`${agentPath(key)}/conversations`, {}, "POST");
          if (!isCurrent()) return;
          rows = [summarize(created)];
          id = created.id;
        }
        setConversations(rows);
        setSelection((value) => value?.owner === key && value.id === id ? value : { owner: key, id });
      })
      .catch((e) => {
        if (isCurrent()) {
          setError(e.message);
          setLoading(false);
        }
      });
    return () => {
      current = false;
    };
  }, [key, resourceRevision]);
  useEffect(() => {
    if (!key || !selectedId) return;
    let current = true;
    const revision = refreshRevision.current;
    const isCurrent = () => current && revision === refreshRevision.current && currentAgentKey.current === key;
    setLoading(true);
    api<Resource>(`${agentPath(key)}/resources/${encodeURIComponent(selectedId)}`)
      .then((conversation) => {
        if (!isCurrent()) return;
        setChain((current) => current?.agent_key === key && current.id === conversation.id
          ? latestConversation(current, conversation) : conversation);
        setResourceOwner(key);
      })
      .catch((e) => {
        if (isCurrent()) setError(e.message);
      })
      .finally(() => {
        if (isCurrent()) setLoading(false);
      });
    return () => {
      current = false;
    };
  }, [key, selectedId, resourceRevision]);

  function updateChain(value: Resource) {
    if (agentKey.current !== key) return;
    setChain((current) => current ? latestConversation(current, value) : current);
    setConversations((rows) => upsertConversation(rows, summarize(value)));
  }
  function show(next: Page) {
    setMenu(false);
    transition(() => setPage(next));
  }
  function openChat(id: string) {
    setSelection({ owner: key, id });
    show({ name: "chat" });
  }
  async function newChat() {
    const reuse = emptyConversation(conversations);
    if (reuse) {
      openChat(reuse.id);
      return;
    }
    const created = await api<Resource>(`${agentPath(key)}/conversations`, {}, "POST");
    if (agentKey.current !== created.agent_key) return;
    setConversations((rows) => upsertConversation(rows, summarize(created)));
    setChain(created);
    setResourceOwner(key);
    openChat(created.id);
  }
  function startNewChat() {
    newChat().catch((e) => setError((e as Error).message));
  }
  async function updateConversation(id: string, change: { title: string }) {
    try {
      const saved = await api<ConversationSummary>(`${agentPath(key)}/conversations/${encodeURIComponent(id)}`, change, "PUT");
      setConversations((rows) => upsertConversation(rows, saved));
    } catch (e) {
      setError((e as Error).message);
      throw e;
    }
  }
  /** After the open chat disappears, open the next one rather than an empty screen. */
  function reopenAfterRemoval(remaining: ConversationSummary[]) {
    const next = pickConversation(remaining.filter((row) => row.message_count > 0), null);
    if (next) openChat(next);
    else startNewChat();
  }
  function deleteChat(id: string) {
    const chat = conversations.find((row) => row.id === id);
    setConfirm({
      title: "Delete chat?",
      message: `“${chat?.title ?? "This chat"}” will be moved to Tsuchi’s trash, where it can be restored.`,
      confirm: "Delete",
      run: async () => {
        await api<void>(`${agentPath(key)}/conversations/${encodeURIComponent(id)}`, undefined, "DELETE");
        const remaining = conversations.filter((row) => row.id !== id);
        setConversations(remaining);
        if (selectedId === id) reopenAfterRemoval(remaining);
      },
    });
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
        <Sidebar
          agent={agent}
          page={page}
          conversations={key && resourceOwner === key ? conversations : []}
          selectedId={selectedId}
          status={status}
          loading={loading}
          disabled={disabled}
          closeRef={closeNavigation}
          onClose={() => setMenu(false)}
          onRefresh={() => {
            void refreshAgent();
            void refreshStatus();
          }}
          onNewChat={startNewChat}
          onOpenChat={openChat}
          onPage={(name) => show({ name })}
          onRenameChat={(id, title) => updateConversation(id, { title })}
          onDeleteChat={deleteChat}
          desktopHost={desktopHost}
          onServerSettings={desktopMode ? () => { setMenu(false); setServerSettingsOpen(true); } : undefined}
        />
      </aside>
      <main id="main" tabIndex={-1} inert={mobile && menu}>
        <button
          className="icon mobile-only mobile-menu"
          aria-label="Open navigation"
          ref={openNavigation}
          aria-expanded={menu}
          aria-controls="navigation"
          onClick={() => setMenu(true)}
        >
          <Menu size={20} />
        </button>
        {error && (
          <div className="banner" role="alert">
            {error}
            {desktopMode && (
              <button onClick={() => setServerSettingsOpen(true)}>
                Server settings
              </button>
            )}
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
        ) : page.name === "plugins" ? (
          <Plugins agentName={agent?.name || key} />
        ) : page.name === "agent" && agent ? (
          <AgentPage
            agent={agent}
            disabled={disabled}
            onEditPet={() => setEditingPet(true)}
            onEditPersona={() => setEditingPersona(true)}
            onRename={() => setRenaming(true)}
            onRunner={() => setAssigningRunner(true)}
          />
        ) : page.name === "space" ? (
          <SpaceLibrary key={key} agentKey={key} />
        ) : chain && resourceOwner === key && chain.id === selectedId ? (
          <Chat
            key={`${key}:${chain.id}`}
            agentKey={key}
            agentName={agent?.name || key}
            chain={chain}
            status={status}
            entrance={hatched ? "jumping" : "waving"}
            onChange={updateChain}
            onBusy={(value) => { if (agentKey.current === key) setBusy(value); }}
            onEditPet={() => setEditingPet(true)}
            onEditPersona={() => setEditingPersona(true)}
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
      {confirm && <ConfirmDialog
        title={confirm.title}
        message={confirm.message}
        confirm={confirm.confirm}
        onConfirm={confirm.run}
        onClose={() => setConfirm(null)}
      />}
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
      {editingPersona && agent && <PersonaEditor
        agentKey={key}
        agentName={agent.name}
        onClose={() => {
          setEditingPersona(false);
          restoreFocus();
        }}
      />}
      {serverSettingsOpen && <ServerSettings
        onConnected={(nextHost) => {
          setDesktopHost(nextHost);
          setAgent(null);
          setChain(null);
          setConversations([]);
          setError("");
          void refreshAgent();
          void refreshStatus();
        }}
        onClose={() => { setServerSettingsOpen(false); restoreFocus(); }}
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
            setPage({ name: "chat" });
          });
        }}
      />}
    </div>
    </PetContext.Provider>
  );
}
