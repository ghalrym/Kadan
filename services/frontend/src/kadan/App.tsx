import { useCallback, useEffect, useRef, useState } from "react";
import { flushSync } from "react-dom";
import { Menu } from "lucide-react";
import { agentPath, api } from "./api";
import {
  getDesktopServerConfig,
  installDesktopUpdate,
  isDesktopApp,
  type ServerUpdateStatus,
} from "./desktop";
import { latestConversation } from "./agentRuns";
import { pickConversation } from "./conversations";
import type { Agent, ConversationSummary, PetJob, Resource, Status } from "./types";
import Chat from "./Chat";
import DirectChat from "./DirectChat";
import GroupChat from "./GroupChat";
import Pet, { agentPet } from "./Pet";
import { agentHue } from "./pet";
import PetEditor from "./PetEditor";
import PersonaEditor from "./PersonaEditor";
import RenameAgent from "./RenameAgent";
import DeleteAgent from "./DeleteAgent";
import Plugins from "./Plugins";
import Sidebar, { type Page } from "./Sidebar";
import SpaceLibrary from "./SpaceLibrary";
import ServerSettings, { runLocationLabel } from "./ServerSettings";
import SettingsPage from "./SettingsPage";
import WorkThreadView from "./WorkThreadView";
import { listWorkThreads, organizeWorkThread, parseWorkHash, type WorkThread } from "./work";
import { PetContext, usePetRefJob } from "./PetContext";
import ProfileDialog from "./ProfileDialog";
import { useChatUnread, useDirectChats, useGroupChatSummary, useNavPreviews, useProfile } from "./community";
import { saveProfile } from "./profile";

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
  const petSelectionRevision = useRef(0);
  const petReadRevision = useRef(0);
  const pendingPetSelection = useRef(0);
  /** The agent's single persistent chat, remembered with the agent it belongs to. */
  const [selection, setSelection] = useState<{ owner: string; id: string } | null>(null);
  const [chain, setChain] = useState<Resource | null>(null);
  const [page, setPage] = useState<Page>(() => {
    const hashThread = typeof window !== "undefined" ? parseWorkHash(window.location?.hash ?? "") : null;
    return hashThread ? { name: "work-thread", threadId: hashThread } : { name: "chat" };
  });
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
  const [resourceRevision, setResourceRevision] = useState(0);
  const key = agent?.key ?? "";
  const [resourceOwner, setResourceOwner] = useState("");
  const [serverSettingsOpen, setServerSettingsOpen] = useState(false);
  const [desktopHost, setDesktopHost] = useState("");
  const [updateNotice, setUpdateNotice] = useState("");
  const handledDesktopUpdateId = useRef<string | null>(null);
  const [deletingAgent, setDeletingAgent] = useState(false);
  const [workThreads, setWorkThreads] = useState<WorkThread[]>([]);
  const { profile, setProfile, reload: reloadProfile } = useProfile(!!agent);
  const profileRef = useRef(profile);
  profileRef.current = profile;
  const [editingProfile, setEditingProfile] = useState(false);
  const [editingUserPet, setEditingUserPet] = useState(false);
  const userPetJob = usePetRefJob(profile.pet);
  const groupChat = useGroupChatSummary(!!agent, page.name === "group-chat");
  const navPreviews = useNavPreviews(key, page.name);
  const directChats = useDirectChats(!!agent, page.name === "direct" ? page.participantId : null);
  const openDirectId = page.name === "direct" ? page.participantId : null;
  useEffect(() => {
    // A removed or paused agent's private chat closes back to the main chat.
    if (openDirectId && directChats.loaded && !directChats.chats.some((chat) => chat.participant.id === openDirectId)) {
      setPage({ name: "chat" });
    }
  }, [openDirectId, directChats.loaded, directChats.chats]);
  const chatId = selection?.owner === key ? selection.id : null;
  const chatUnread = useChatUnread(
    key,
    chatId,
    chain && chain.id === chatId ? chain.messages.length : null,
    page.name === "chat",
  );
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
  const selectUserPet = useCallback(async (id: string | null) => {
    try {
      setProfile(await saveProfile(profileRef.current, id ? { pet: { agent_key: key, pet_id: id } } : { clear_pet: true }));
    } catch (error) {
      // The write may have committed even when its response never arrived.
      await reloadProfile().catch(() => {});
      throw error;
    }
  }, [key, reloadProfile, setProfile]);

  useEffect(() => {
    if (!desktopMode) return;
    void getDesktopServerConfig()
      .then((cfg) => setDesktopHost(runLocationLabel(cfg)))
      .catch(() => {});
  }, [desktopMode]);

  useEffect(() => {
    if (!desktopMode) return;
    let active = true;
    const pollDesktopUpdates = async () => {
      try {
        const updateStatus = await api<ServerUpdateStatus>("/updates/status");
        if (!active) return;
        const pending = updateStatus?.pending_desktop_update;
        if (pending && pending.request_id !== handledDesktopUpdateId.current) {
          handledDesktopUpdateId.current = pending.request_id;
          setUpdateNotice(`Updating Kadan to ${pending.target_version}…`);
          await api("/updates/desktop/ack", { request_id: pending.request_id }, "POST").catch(() => {});
          try {
            const installed = await installDesktopUpdate();
            if (active && !installed.restarting) {
              setUpdateNotice("");
            }
          } catch (err) {
            if (active) {
              setUpdateNotice("");
              setError((err as Error).message);
            }
          }
        }
      } catch {
        // Server may be restarting during an update; ignore transient poll errors.
      }
    };
    void pollDesktopUpdates();
    const timer = setInterval(() => void pollDesktopUpdates(), 10000);
    return () => {
      active = false;
      clearInterval(timer);
    };
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
        tsuchi: { state: "unavailable", message: "Kadan can’t reach the server right now." },
        hikari: { state: "unavailable", message: "Kadan can’t reach its AI models right now.", model_id: null },
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
        setDeletingAgent(false);
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
  const refreshWorkThreads = useCallback(async () => {
    try {
      setWorkThreads(await listWorkThreads());
    } catch {
      // Ignore transient Work polling errors so chat stays uninterrupted.
    }
  }, []);

  useEffect(() => {
    void refreshAgent();
    void refreshStatus();
    void refreshWorkThreads();
    const statusTimer = setInterval(() => void refreshStatus(), 30000);
    const workTimer = setInterval(() => void refreshWorkThreads(), 5000);
    const onHashChange = () => {
      const hashThread = typeof window !== "undefined" ? parseWorkHash(window.location?.hash ?? "") : null;
      if (hashThread) {
        setMenu(false);
        transition(() => setPage({ name: "work-thread", threadId: hashThread }));
      }
    };
    if (typeof window !== "undefined" && typeof window.addEventListener === "function") {
      window.addEventListener("hashchange", onHashChange);
    }
    return () => {
      clearInterval(statusTimer);
      clearInterval(workTimer);
      if (typeof window !== "undefined" && typeof window.removeEventListener === "function") {
        window.removeEventListener("hashchange", onHashChange);
      }
    };
  }, [refreshWorkThreads]);
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
  // Resolve the agent's single persistent thread, creating it if none exists yet.
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
          id = created.id;
        }
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
  }
  function show(next: Page) {
    setMenu(false);
    transition(() => setPage(next));
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
          profile={profile}
          onProfile={() => setEditingProfile(true)}
          groupChat={groupChat.summary}
          groupChatUnread={groupChat.unread}
          chatUnread={chatUnread}
          navPreviews={navPreviews}
          directChats={directChats.chats}
          directUnread={directChats.unread}
          onOpenDirect={(participantId) => {
            if (typeof window !== "undefined" && window.location?.hash?.startsWith("#work/")) {
              window.history?.replaceState?.(null, "", window.location.pathname + window.location.search);
            }
            show({ name: "direct", participantId });
          }}
          page={page}
          status={status}
          loading={loading}
          disabled={disabled}
          closeRef={closeNavigation}
          onClose={() => setMenu(false)}
          onRefresh={() => {
            void refreshAgent();
            void refreshStatus();
            void refreshWorkThreads();
          }}
          onPage={(name) => {
            if (typeof window !== "undefined" && window.location?.hash?.startsWith("#work/")) {
              window.history?.replaceState?.(null, "", window.location.pathname + window.location.search);
            }
            if (
              name === "chat" ||
              name === "group-chat" ||
              name === "space" ||
              name === "plugins" ||
              name === "settings"
            ) {
              show({ name });
            }
          }}
          workThreads={workThreads}
          onOpenWorkThread={(threadId) => {
            if (typeof window !== "undefined") {
              window.history?.replaceState?.(null, "", `#work/${threadId}`);
            }
            show({ name: "work-thread", threadId });
          }}
          onNewWorkThread={() => {
            if (typeof window !== "undefined" && window.location?.hash?.startsWith("#work/")) {
              window.history?.replaceState?.(null, "", window.location.pathname + window.location.search);
            }
            show({ name: "work-new" });
          }}
          onSettleWorkThread={(threadId) => {
            void organizeWorkThread(threadId, { action: "settle" })
              .then(() => refreshWorkThreads())
              .catch(() => {});
          }}
          onUnsettleWorkThread={(threadId) => {
            void organizeWorkThread(threadId, { action: "unsettle" })
              .then(() => refreshWorkThreads())
              .catch(() => {});
          }}
          onWakeWorkThread={(threadId) => {
            void organizeWorkThread(threadId, { action: "default" })
              .then(() => refreshWorkThreads())
              .catch(() => {});
          }}
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
        {updateNotice && (
          <div className="banner" role="status">
            {updateNotice}
          </div>
        )}
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
            <h2>{error ? "Could not reach Kadan Server" : "Setting up your agent"}</h2>
            <p className="center-hint">
              {error ? "Check your server connection and try again." : "Connecting to your server…"}
            </p>
            <button className="primary" onClick={() => void refreshAgent()}>
              Retry
            </button>
          </div>
        ) : page.name === "direct" ? (
          <DirectChat
            key={page.participantId}
            participantId={page.participantId}
            agentKey={key}
            status={status}
            onSnapshot={directChats.onSnapshot}
          />
        ) : page.name === "group-chat" ? (
          <GroupChat agentKey={key} agentName={agent?.name || key} status={status} profile={profile}
            onSnapshot={groupChat.onSnapshot} onEditProfile={() => setEditingProfile(true)}
            onEditAgentPet={() => setEditingPet(true)} />
        ) : page.name === "work-thread" || page.name === "work-new" ? (
          <WorkThreadView
            key={page.name === "work-thread" ? page.threadId : "work-new"}
            threadId={page.name === "work-thread" ? page.threadId : null}
            onThreadCreated={(threadId) => {
              window.history.replaceState(null, "", `#work/${threadId}`);
              show({ name: "work-thread", threadId });
            }}
            onOpenSettings={() => {
              if (window.location.hash.startsWith("#work/")) {
                window.history.replaceState(null, "", window.location.pathname + window.location.search);
              }
              show({ name: "settings" });
            }}
            onChanged={() => void refreshWorkThreads()}
            onDeleted={() => {
              void refreshWorkThreads();
              if (window.location.hash.startsWith("#work/")) {
                window.history.replaceState(null, "", window.location.pathname + window.location.search);
              }
              show({ name: "work-new" });
            }}
          />
        ) : page.name === "settings" ? (
          <SettingsPage
            desktopMode={desktopMode}
            desktopHost={desktopHost}
            status={status}
            onOpenServerSettings={desktopMode ? () => setServerSettingsOpen(true) : undefined}
            onChanged={() => void refreshWorkThreads()}
          />
        ) : page.name === "plugins" ? (
          <Plugins agentName={agent?.name || key} />
        ) : page.name === "space" ? (
          <SpaceLibrary key={key} agentKey={key} />
        ) : chain && resourceOwner === key && chain.id === selectedId ? (
          <Chat
            key={`${key}:${chain.id}`}
            agentKey={key}
            agentName={agent?.name || key}
            chain={chain}
            status={status}
            entrance="waving"
            onChange={updateChain}
            onBusy={(value) => { if (agentKey.current === key) setBusy(value); }}
            onEditPet={() => setEditingPet(true)}
            onEditPersona={() => setEditingPersona(true)}
            onRename={() => setRenaming(true)}
            onDeleteAgent={() => setDeletingAgent(true)}
            onAgentName={(name) => setAgent((current) => current?.key === key && current.name !== name ? { ...current, name } : current)}
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
      {deletingAgent && agent && <DeleteAgent
        agent={agent}
        onDeleted={() => {
          setDeletingAgent(false);
          setAgent(null);
          setPets({});
          setSelection(null);
          setChain(null);
          setResourceOwner("");
          setPage({ name: "chat" });
          void refreshAgent();
        }}
        onClose={() => {
          setDeletingAgent(false);
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
      {editingProfile && <ProfileDialog
        profile={profile}
        onSaved={setProfile}
        onEditPet={() => setEditingUserPet(true)}
        onReload={reloadProfile}
        onClose={() => {
          setEditingProfile(false);
          restoreFocus();
        }}
      />}
      {editingUserPet && agent && <PetEditor
        agentKey={key}
        agentName={profile.display_name}
        title="Your pet"
        selected={userPetJob}
        onSelect={selectUserPet}
        onRefreshSelection={reloadProfile}
        onClose={() => setEditingUserPet(false)}
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
          setError("");
          void refreshAgent();
          void refreshStatus();
        }}
        onClose={() => { setServerSettingsOpen(false); restoreFocus(); }}
      />}
    </div>
    </PetContext.Provider>
  );
}
