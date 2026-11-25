import { useEffect, useRef, useState } from "react";
import {
  ChevronRight,
  Folder,
  FolderOpen,
  FolderPlus,
  MoreHorizontal,
  Pencil,
  Plus,
  Puzzle,
  RefreshCw,
  SquarePen,
  Trash2,
  X,
} from "lucide-react";
import Pet from "./Pet";
import { useAgentPet } from "./PetContext";
import { listedConversations, relativeTime } from "./conversations";
import type { Agent, ConversationSummary, Space, Status } from "./types";

export type Page = { name: "chat" } | { name: "space"; id: string } | { name: "plugins" } | { name: "agent" };

interface Props {
  agent: Agent | null;
  page: Page;
  conversations: ConversationSummary[];
  spaces: Space[];
  selectedId: string | null;
  status: Status | null;
  loading: boolean;
  disabled: boolean;
  closeRef: React.RefObject<HTMLButtonElement | null>;
  onClose: () => void;
  onRefresh: () => void;
  onNewChat: (spaceId: string | null) => void;
  onOpenChat: (id: string) => void;
  onOpenSpace: (id: string) => void;
  onCreateSpace: () => void;
  onPage: (page: "plugins" | "agent") => void;
  onRenameChat: (id: string, title: string) => Promise<void>;
  onMoveChat: (id: string, spaceId: string | null) => Promise<void>;
  onDeleteChat: (id: string) => void;
}

/** Spaces show a few recent chats in the sidebar; the Space page lists them all. */
const SPACE_PREVIEW = 5;

function ChatRow({ chat, selected, spaces, disabled, now, onOpen, onRename, onMove, onDelete }: {
  chat: ConversationSummary;
  selected: boolean;
  spaces: Space[];
  disabled: boolean;
  now: number;
  onOpen: () => void;
  onRename: (title: string) => Promise<void>;
  onMove: (spaceId: string | null) => Promise<void>;
  onDelete: () => void;
}) {
  const [menu, setMenu] = useState(false);
  const [renaming, setRenaming] = useState(false);
  const [title, setTitle] = useState(chat.title);
  const row = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!menu) return;
    const outside = (event: PointerEvent) => {
      if (!row.current?.contains(event.target as Node)) setMenu(false);
    };
    const escape = (event: KeyboardEvent) => { if (event.key === "Escape") setMenu(false); };
    document.addEventListener("pointerdown", outside);
    document.addEventListener("keydown", escape);
    return () => {
      document.removeEventListener("pointerdown", outside);
      document.removeEventListener("keydown", escape);
    };
  }, [menu]);
  async function submit() {
    const next = title.trim();
    setRenaming(false);
    if (next && next !== chat.title) await onRename(next).catch(() => setTitle(chat.title));
    else setTitle(chat.title);
  }
  if (renaming) {
    return (
      <form className="chat-row is-renaming" onSubmit={(event) => { event.preventDefault(); void submit(); }}>
        <label className="sr-only" htmlFor={`rename-${chat.id}`}>Chat name</label>
        <input id={`rename-${chat.id}`} autoFocus value={title} maxLength={200}
          onChange={(event) => setTitle(event.target.value)}
          onBlur={() => void submit()}
          onKeyDown={(event) => { if (event.key === "Escape") { setTitle(chat.title); setRenaming(false); } }} />
      </form>
    );
  }
  const elsewhere = spaces.filter((space) => space.id !== chat.space_id);
  return (
    <div ref={row} className={`chat-row ${selected ? "selected" : ""} ${menu ? "menu-open" : ""}`}>
      <button className="chat-row-open" aria-current={selected ? "page" : undefined} onClick={onOpen} title={chat.title}>
        <span className="chat-row-title">{chat.title}</span>
        <time dateTime={chat.updated_at}>{relativeTime(chat.updated_at, now)}</time>
      </button>
      <button className="chat-row-more" aria-label={`Options for ${chat.title}`} aria-expanded={menu}
        aria-haspopup="menu" disabled={disabled} onClick={() => setMenu(!menu)}>
        <MoreHorizontal size={15} />
      </button>
      {menu && (
        <div className="menu" role="menu">
          <button role="menuitem" onClick={() => { setMenu(false); setTitle(chat.title); setRenaming(true); }}>
            <Pencil size={14} />Rename
          </button>
          {elsewhere.length > 0 && <p className="menu-label">Move to Space</p>}
          {elsewhere.map((space) => (
            <button role="menuitem" key={space.id} onClick={() => { setMenu(false); void onMove(space.id); }}>
              <Folder size={14} /><span className="menu-text">{space.title}</span>
            </button>
          ))}
          {chat.space_id && (
            <button role="menuitem" onClick={() => { setMenu(false); void onMove(null); }}>
              <FolderOpen size={14} />Remove from Space
            </button>
          )}
          <hr />
          <button role="menuitem" className="danger" onClick={() => { setMenu(false); onDelete(); }}>
            <Trash2 size={14} />Delete
          </button>
        </div>
      )}
    </div>
  );
}

export default function Sidebar(props: Props) {
  const { agent, page, conversations, spaces, selectedId, status, loading, disabled } = props;
  const sheet = useAgentPet(agent?.key ?? "");
  // Spaces open with their active chat; a toggle overrides that until the next visit.
  const [toggled, setToggled] = useState<Record<string, boolean>>({});
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), 60000);
    return () => clearInterval(timer);
  }, []);
  const key = agent?.key ?? "";
  const openSpaceId = page.name === "space" ? page.id
    : page.name === "chat" ? conversations.find((row) => row.id === selectedId)?.space_id ?? null : null;
  const chats = listedConversations(conversations, null, page.name === "chat" ? selectedId : null);
  const rowProps = (chat: ConversationSummary) => ({
    chat,
    selected: page.name === "chat" && chat.id === selectedId,
    spaces,
    disabled,
    now,
    onOpen: () => props.onOpenChat(chat.id),
    onRename: (title: string) => props.onRenameChat(chat.id, title),
    onMove: (spaceId: string | null) => props.onMoveChat(chat.id, spaceId),
    onDelete: () => props.onDeleteChat(chat.id),
  });
  return (
    <>
      <header className="panel-head">
        <span className="brand">
          <span className="logo" aria-hidden="true"><i /><i /><i /><i /></span>
          Kadan
        </span>
        <button className="icon mobile-only" aria-label="Close navigation" ref={props.closeRef} onClick={props.onClose}>
          <X size={18} />
        </button>
      </header>
      <nav className="panel-nav" aria-label="Views">
        <button disabled={!key || disabled} onClick={() => props.onNewChat(null)}>
          <SquarePen size={16} />
          New chat
        </button>
        <button className={page.name === "plugins" ? "selected" : ""} disabled={!key}
          aria-current={page.name === "plugins" ? "page" : undefined} onClick={() => props.onPage("plugins")}>
          <Puzzle size={16} />
          Plugins
        </button>
      </nav>
      <div className="panel-scroll">
        <section className="panel-section" aria-labelledby="spaces-label">
          <header className="panel-section-head">
            <h2 className="panel-label" id="spaces-label">Spaces</h2>
            <button className="icon panel-add" aria-label="New Space" title="New Space" disabled={!key || disabled}
              onClick={props.onCreateSpace}>
              <Plus size={15} />
            </button>
          </header>
          {spaces.length === 0 ? (
            <button className="panel-empty" disabled={!key || disabled} onClick={props.onCreateSpace}>
              <FolderPlus size={16} />
              New Space
            </button>
          ) : spaces.map((space) => {
            const inside = listedConversations(conversations, space.id, page.name === "chat" ? selectedId : null);
            const expanded = toggled[space.id] ?? openSpaceId === space.id;
            const shown = inside.slice(0, SPACE_PREVIEW);
            return (
              <div className="space-group" key={space.id}>
                <div className={`space-row ${page.name === "space" && page.id === space.id ? "selected" : ""}`}>
                  <button className="space-toggle" aria-label={expanded ? `Hide chats in ${space.title}` : `Show chats in ${space.title}`}
                    aria-expanded={expanded} disabled={!inside.length}
                    onClick={() => setToggled((current) => ({ ...current, [space.id]: !expanded }))}>
                    <ChevronRight size={12} className="space-chevron" aria-hidden="true" />
                    {expanded ? <FolderOpen size={16} /> : <Folder size={16} />}
                  </button>
                  <button className="space-open" aria-current={page.name === "space" && page.id === space.id ? "page" : undefined}
                    onClick={() => {
                      setToggled((current) => ({ ...current, [space.id]: true }));
                      props.onOpenSpace(space.id);
                    }}>
                    {space.title}
                  </button>
                </div>
                {expanded && shown.length > 0 && (
                  <div className="space-chats">
                    {shown.map((chat) => <ChatRow key={chat.id} {...rowProps(chat)} />)}
                    {inside.length > shown.length && (
                      <button className="panel-more" onClick={() => props.onOpenSpace(space.id)}>
                        See all {inside.length}
                      </button>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </section>
        <section className="panel-section" aria-labelledby="chats-label">
          <header className="panel-section-head">
            <h2 className="panel-label" id="chats-label">Chats</h2>
          </header>
          {chats.length === 0 ? (
            <p className="panel-hint">{loading ? "Loading chats…" : "Chats you start appear here."}</p>
          ) : chats.map((chat) => <ChatRow key={chat.id} {...rowProps(chat)} />)}
        </section>
      </div>
      <footer className="panel-foot">
        {agent && (
          <button className={`agent-link ${page.name === "agent" ? "selected" : ""}`}
            aria-current={page.name === "agent" ? "page" : undefined} onClick={() => props.onPage("agent")}>
            <span className="agent-link-pet" aria-hidden="true">
              <Pet sheet={sheet} state="idle" size={28} />
            </span>
            <span className="agent-link-name">{agent.name}</span>
            <span className="agent-link-hint">Agent</span>
          </button>
        )}
        <div className="services">
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
          <button className="icon services-refresh" aria-label="Refresh services and agent" title="Refresh"
            disabled={disabled} onClick={props.onRefresh}>
            <RefreshCw size={14} className={loading ? "spin" : ""} />
          </button>
        </div>
      </footer>
    </>
  );
}
