import { useEffect, useRef, useState } from "react";
import {
  LayoutGrid,
  MoreHorizontal,
  Pencil,
  Puzzle,
  RefreshCw,
  Settings,
  SquarePen,
  Trash2,
  X,
} from "lucide-react";
import { Link } from "react-router";
import Pet from "./Pet";
import { useAgentPet } from "./PetContext";
import { listedConversations, relativeTime } from "./conversations";
import type { Agent, ConversationSummary, Status } from "./types";

export type Page = { name: "chat" } | { name: "space" } | { name: "plugins" } | { name: "agent" };

interface Props {
  agent: Agent | null;
  page: Page;
  conversations: ConversationSummary[];
  selectedId: string | null;
  status: Status | null;
  loading: boolean;
  disabled: boolean;
  closeRef: React.RefObject<HTMLButtonElement | null>;
  onClose: () => void;
  onRefresh: () => void;
  onNewChat: () => void;
  onOpenChat: (id: string) => void;
  onPage: (page: "space" | "plugins" | "agent") => void;
  onRenameChat: (id: string, title: string) => Promise<void>;
  onDeleteChat: (id: string) => void;
  desktopHost?: string;
  onServerSettings?: () => void;
}

function ChatRow({ chat, selected, disabled, now, onOpen, onRename, onDelete }: {
  chat: ConversationSummary;
  selected: boolean;
  disabled: boolean;
  now: number;
  onOpen: () => void;
  onRename: (title: string) => Promise<void>;
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
  const { agent, page, conversations, selectedId, status, loading, disabled } = props;
  const sheet = useAgentPet(agent?.key ?? "");
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), 60000);
    return () => clearInterval(timer);
  }, []);
  const key = agent?.key ?? "";
  const chats = listedConversations(conversations, page.name === "chat" ? selectedId : null);
  const rowProps = (chat: ConversationSummary) => ({
    chat,
    selected: page.name === "chat" && chat.id === selectedId,
    disabled,
    now,
    onOpen: () => props.onOpenChat(chat.id),
    onRename: (title: string) => props.onRenameChat(chat.id, title),
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
        <button disabled={!key || disabled} onClick={props.onNewChat}>
          <SquarePen size={16} />
          New chat
        </button>
        <button className={page.name === "space" ? "selected" : ""} disabled={!key}
          aria-current={page.name === "space" ? "page" : undefined} onClick={() => props.onPage("space")}>
          <LayoutGrid size={16} />
          Space
        </button>
        <button className={page.name === "plugins" ? "selected" : ""} disabled={!key}
          aria-current={page.name === "plugins" ? "page" : undefined} onClick={() => props.onPage("plugins")}>
          <Puzzle size={16} />
          Plugins
        </button>
      </nav>
      <div className="panel-scroll">
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
        {props.onServerSettings && (
          <button
            className="server-settings-trigger"
            onClick={props.onServerSettings}
            title="Change Kadan Server host"
          >
            <Settings size={15} />
            <span className="server-settings-nav">
              <span>Server settings</span>
              {props.desktopHost && (
                <small>{props.desktopHost.replace(/^https?:\/\//, "")}</small>
              )}
            </span>
          </button>
        )}
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
          <Link to="/tsuchi" title={status?.tsuchi.message ?? "Checking"}>
            <span className={`dot ${status?.tsuchi.state === "ready" ? "ready" : status ? "down" : ""}`} />
            Tsuchi
            <span className="sr-only">
              {!status ? "checking" : status.tsuchi.state === "ready" ? "connected" : "unavailable"}
            </span>
          </Link>
          <Link to="/hikari/settings" title={status?.hikari.message ?? "Checking"}>
            <span
              className={`dot ${status?.hikari.state === "ready" ? "ready" : status?.hikari.state === "unavailable" ? "down" : status ? "warn" : ""}`}
            />
            Hikari
            <span className="sr-only">
              {!status ? "checking" : status.hikari.state === "ready" ? "ready" : "needs attention"}
            </span>
          </Link>
          <button className="icon services-refresh" aria-label="Refresh services and agent" title="Refresh"
            disabled={disabled} onClick={props.onRefresh}>
            <RefreshCw size={14} className={loading ? "spin" : ""} />
          </button>
        </div>
      </footer>
    </>
  );
}
