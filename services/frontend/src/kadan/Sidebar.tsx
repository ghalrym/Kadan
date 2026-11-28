import { useEffect, useState } from "react";
import {
  AlarmClockOff,
  Archive,
  ArchiveRestore,
  ChevronDown,
  LayoutGrid,
  Loader2,
  Puzzle,
  RefreshCw,
  Search,
  Settings,
  SquarePen,
  X,
} from "lucide-react";
import type { GroupChatSummary } from "./community";
import { isParticipantActive } from "./groupChat";
import Pet from "./Pet";
import { ParticipantAvatar, UserAvatar } from "./PetAvatar";
import type { UserProfile } from "./profile";
import { useAgentPet } from "./PetContext";
import ProviderIcon from "./ProviderIcon";
import type { Agent, Status } from "./types";
import {
  groupThreadsByBucket,
  isThreadActive,
  listWorkProjects,
  matchesThreadQuery,
  pendingRequestsFor,
  projectHue,
  projectInitials,
  relativeTime,
  snoozeRemaining,
  workStatusLabel,
  type WorkProject,
  type WorkThread,
} from "./work";

export type Page =
  | { name: "chat" }
  | { name: "group-chat" }
  | { name: "space" }
  | { name: "plugins" }
  | { name: "settings" }
  | { name: "work-new" }
  | { name: "work-thread"; threadId: string };

type Shelf = "working" | "snoozed" | "settled";
type Shelves = Record<Shelf, boolean>;

const SHELVES_KEY = "kadan:work:shelves";
const SETTLED_PAGE = 25;
const DEFAULT_SHELVES: Shelves = { working: true, snoozed: false, settled: false };

function readShelves(): Shelves {
  try {
    const raw = typeof localStorage !== "undefined" ? localStorage.getItem(SHELVES_KEY) : null;
    return raw ? { ...DEFAULT_SHELVES, ...JSON.parse(raw) } : DEFAULT_SHELVES;
  } catch {
    return DEFAULT_SHELVES;
  }
}

function writeShelves(shelves: Shelves): void {
  try {
    localStorage.setItem(SHELVES_KEY, JSON.stringify(shelves));
  } catch {
    // Shelf state is a convenience; ignore storage errors.
  }
}

function byUpdatedDesc(a: WorkThread, b: WorkThread): number {
  return (b.updated_at ?? "").localeCompare(a.updated_at ?? "");
}

function bySnoozeAsc(a: WorkThread, b: WorkThread): number {
  return (a.snoozed_until ?? "").localeCompare(b.snoozed_until ?? "");
}

/** Avatars shown on the Group Chat row before the rest are counted. */
const STACK = 3;

interface Props {
  agent: Agent | null;
  profile: UserProfile;
  onProfile: () => void;
  groupChat: GroupChatSummary | null;
  groupChatUnread: boolean;
  chatUnread: boolean;
  page: Page;
  status: Status | null;
  loading: boolean;
  disabled: boolean;
  closeRef: React.RefObject<HTMLButtonElement | null>;
  onClose: () => void;
  onRefresh: () => void;
  onPage: (page: Page["name"]) => void;
  desktopHost?: string;
  onServerSettings?: () => void;
  workThreads?: WorkThread[];
  onOpenWorkThread?: (threadId: string) => void;
  onNewWorkThread?: () => void;
  onSettleWorkThread?: (threadId: string) => void;
  onUnsettleWorkThread?: (threadId: string) => void;
  onWakeWorkThread?: (threadId: string) => void;
}

export default function Sidebar(props: Props) {
  const { agent, page, status, loading, disabled } = props;
  const sheet = useAgentPet(agent?.key ?? "");
  const key = agent?.key ?? "";
  // Until Group Chat answers, the agent is the only other member worth showing.
  const groupMembers = props.groupChat
    ? props.groupChat.participants.filter(isParticipantActive)
    : key ? [{ kind: "kadan_agent" as const, agent_key: key, handle: key, pet: null }] : [];
  const [query, setQuery] = useState("");
  const [shelves, setShelves] = useState<Shelves>(() => readShelves());
  const [settledLimit, setSettledLimit] = useState(SETTLED_PAGE);
  const [projects, setProjects] = useState<WorkProject[]>([]);
  const workThreads = props.workThreads ?? [];

  // Project names and badges are shown on each thread row; reload when a thread
  // points at a project we have not seen yet.
  const unknownProject = workThreads.some((t) => !projects.some((p) => p.id === t.project_id));
  useEffect(() => {
    if (!key || !unknownProject) return;
    let cancelled = false;
    void listWorkProjects()
      .then((list) => {
        if (!cancelled) setProjects(list);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [key, unknownProject, workThreads.length]);

  const projectById = new Map(projects.map((p) => [p.id, p]));
  const searching = query.trim().length > 0;
  const visible = workThreads.filter((t) =>
    matchesThreadQuery(t, query, projectById.get(t.project_id)?.title ?? ""),
  );
  const buckets = groupThreadsByBucket(visible);
  buckets.default.sort(byUpdatedDesc);
  buckets.working.sort(byUpdatedDesc);
  buckets.snoozed.sort(bySnoozeAsc);
  buckets.settled.sort(byUpdatedDesc);
  const isOpen = (shelf: Shelf) => shelves[shelf] || (searching && buckets[shelf].length > 0);

  function toggleShelf(shelf: Shelf) {
    setShelves((prev) => {
      const next = { ...prev, [shelf]: !prev[shelf] };
      writeShelves(next);
      return next;
    });
  }

  const serverDot = !status
    ? ""
    : status.tsuchi.state !== "ready" || status.hikari.state === "unavailable"
      ? "down"
      : status.hikari.state !== "ready"
        ? "warn"
        : "ready";
  const serverStatusText = !status
    ? "Checking server…"
    : status.tsuchi.state !== "ready"
      ? "Server offline"
      : status.hikari.state === "ready"
        ? "Server connected"
        : status.hikari.state === "model_required"
          ? "Model setup needed"
          : "Server attention";
  const serverTooltip = !status
    ? "Checking Kadan Server"
    : status.tsuchi.state !== "ready"
      ? status.tsuchi.message
      : status.hikari.state !== "ready"
        ? status.hikari.message
        : "Change Kadan Server host";

  const selectedWorkId = page.name === "work-thread" ? page.threadId : null;

  function renderStatus(thread: WorkThread) {
    if (pendingRequestsFor(thread).length > 0) {
      return (
        <span className="work-row-status is-warn">
          <span className="work-row-dot" aria-hidden="true" />
          Needs input
        </span>
      );
    }
    if (isThreadActive(thread.run_status)) {
      return (
        <span className="work-row-status is-active">
          <Loader2 size={11} className="spin" aria-hidden="true" />
          {workStatusLabel(thread.run_status)}
        </span>
      );
    }
    if (thread.bucket === "snoozed") {
      const left = snoozeRemaining(thread.snoozed_until);
      return <span className="work-row-status">{left ? `Snoozed, ${left} left` : "Waking soon"}</span>;
    }
    if (thread.run_status === "failed") {
      return <span className="work-row-status is-error">Failed</span>;
    }
    return <span className="work-row-status">{workStatusLabel(thread.run_status)}</span>;
  }

  function renderThreadRow(thread: WorkThread) {
    const selected = selectedWorkId === thread.id;
    const project = projectById.get(thread.project_id);
    const projectTitle = project?.title ?? "Project";
    let action: React.ReactNode = null;
    if (thread.bucket === "snoozed") {
      action = (
        <button
          type="button"
          className="work-row-action"
          aria-label={`Wake ${thread.title}`}
          title="Wake now"
          onClick={(e) => {
            e.stopPropagation();
            props.onWakeWorkThread?.(thread.id);
          }}
        >
          <AlarmClockOff size={13} />
        </button>
      );
    } else if (thread.bucket === "settled") {
      action = (
        <button
          type="button"
          className="work-row-action"
          aria-label={`Unsettle ${thread.title}`}
          title="Un-settle"
          onClick={(e) => {
            e.stopPropagation();
            props.onUnsettleWorkThread?.(thread.id);
          }}
        >
          <ArchiveRestore size={13} />
        </button>
      );
    } else if (props.onSettleWorkThread && thread.bucket === "default") {
      action = (
        <button
          type="button"
          className="work-row-action"
          aria-label={`Settle ${thread.title}`}
          title="Settle"
          onClick={(e) => {
            e.stopPropagation();
            props.onSettleWorkThread?.(thread.id);
          }}
        >
          <Archive size={13} />
        </button>
      );
    }
    return (
      <div key={thread.id} className={`chat-row work-thread-row ${selected ? "selected" : ""}`}>
        <button
          className="chat-row-open"
          aria-current={selected ? "page" : undefined}
          onClick={() => props.onOpenWorkThread?.(thread.id)}
        >
          <span className="work-row-top">
            <span
              className="work-project-badge"
              style={{ "--project-h": projectHue(thread.project_id) } as React.CSSProperties}
              aria-hidden="true"
            >
              {projectInitials(projectTitle)}
            </span>
            <span className="work-row-project">{projectTitle}</span>
            <time className="work-row-time" dateTime={thread.updated_at}>
              {relativeTime(thread.updated_at)}
            </time>
          </span>
          <span className="work-row-title">{thread.title}</span>
          <span className="work-row-meta">
            {renderStatus(thread)}
            <ProviderIcon harness={thread.harness} size={14} className="work-row-provider" />
          </span>
        </button>
        {action}
      </div>
    );
  }

  function renderShelf(shelf: Shelf, label: string, threads: WorkThread[]) {
    const open = isOpen(shelf);
    const shown = shelf === "settled" ? threads.slice(0, settledLimit) : threads;
    return (
      <div className="work-shelf" data-bucket={shelf}>
        <button
          type="button"
          className="work-shelf-toggle"
          aria-expanded={open}
          onClick={() => toggleShelf(shelf)}
        >
          <span className="work-shelf-label">{label}</span>
          <span className="work-shelf-count">{threads.length}</span>
          <ChevronDown size={13} className="work-shelf-chevron" aria-hidden="true" />
        </button>
        {open && (
          <div className="work-shelf-list">
            {shown.map((thread) => renderThreadRow(thread))}
            {shelf === "settled" && threads.length > shown.length && (
              <button
                type="button"
                className="work-show-more"
                onClick={() => setSettledLimit((n) => n + SETTLED_PAGE)}
              >
                Show more
              </button>
            )}
          </div>
        )}
      </div>
    );
  }

  return (
    <>
      <div className="panel-head">
        <span className="brand">
          <span className="logo" aria-hidden="true"><i /><i /><i /><i /></span>
          Kadan
        </span>
        <button className="icon mobile-only" aria-label="Close navigation" ref={props.closeRef} onClick={props.onClose}>
          <X size={18} />
        </button>
      </div>
      <header className="panel-head">
        <label className="sidebar-search">
          <Search size={15} aria-hidden="true" />
          <input
            type="search"
            aria-label="Search threads"
            placeholder="Search threads"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
        </label>
        <button
          className={`icon sidebar-head-btn ${page.name === "work-new" ? "selected" : ""}`}
          aria-label="New work thread"
          title="New thread"
          disabled={!key}
          onClick={() => props.onNewWorkThread?.()}
        >
          <SquarePen size={17} />
        </button>
      </header>
      <nav className="panel-nav" aria-label="Views">
        {agent && (
          <button className={`agent-link ${page.name === "chat" ? "selected" : ""}`} disabled={!key}
            aria-current={page.name === "chat" ? "page" : undefined} onClick={() => props.onPage("chat")}>
            <span className="agent-link-pet" aria-hidden="true">
              <Pet sheet={sheet} state="idle" size={28} />
            </span>
            <span className="agent-link-name">{agent.name}</span>
            {props.chatUnread && <span className="unread-dot" role="status" aria-label="New messages" />}
          </button>
        )}
        <button className={`group-chat-link ${page.name === "group-chat" ? "selected" : ""}`} disabled={!key}
          aria-current={page.name === "group-chat" ? "page" : undefined} onClick={() => props.onPage("group-chat")}>
          <span className="member-stack" aria-hidden="true">
            <UserAvatar profile={props.profile} size={22} />
            {groupMembers.slice(0, STACK - 1).map((member) => (
              <ParticipantAvatar key={member.handle} participant={member} size={22} />
            ))}
          </span>
          <span className="group-chat-link-name">Group Chat</span>
          {groupMembers.length > STACK - 1 && (
            <span className="member-count" title={`${groupMembers.length + 1} members`}>
              +{groupMembers.length - (STACK - 1)}
            </span>
          )}
          {props.groupChatUnread && <span className="unread-dot" role="status" aria-label="New messages" />}
        </button>
        <button className={page.name === "space" ? "selected" : ""} disabled={!key}
          aria-current={page.name === "space" ? "page" : undefined} onClick={() => props.onPage("space")}>
          <LayoutGrid size={15} />
          Space
        </button>
        <button className={page.name === "plugins" ? "selected" : ""} disabled={!key}
          aria-current={page.name === "plugins" ? "page" : undefined} onClick={() => props.onPage("plugins")}>
          <Puzzle size={15} />
          Plugins
        </button>
      </nav>

      <section className="panel-scroll work-sidebar" aria-label="Work threads">
        <div className="work-active" data-bucket="default">
          {buckets.default.length === 0 ? (
            <p className="panel-hint work-empty-hint">
              {searching ? "No threads found" : "No active threads"}
            </p>
          ) : (
            buckets.default.map((thread) => renderThreadRow(thread))
          )}
        </div>
        <div className="work-shelves">
          {buckets.working.length > 0 && renderShelf("working", "Working", buckets.working)}
          {buckets.snoozed.length > 0 && renderShelf("snoozed", "Snoozed", buckets.snoozed)}
          {renderShelf("settled", "Settled", buckets.settled)}
        </div>
      </section>

      <footer className="panel-foot">
        <button className="profile-link" onClick={props.onProfile} title="Your name and pet">
          <UserAvatar profile={props.profile} size={30} className="profile-link-pet" />
          <span className="profile-link-name">{props.profile.display_name}</span>
          <span className="agent-link-hint">You</span>
        </button>
        <div className="sidebar-server-row">
          <button
            className={`server-settings-trigger ${page.name === "settings" ? "selected" : ""}`}
            aria-current={page.name === "settings" ? "page" : undefined}
            onClick={() => props.onPage("settings")}
            title={serverTooltip}
          >
            <Settings size={15} />
            <span className="server-settings-nav">
              <span className="server-settings-label">
                <span>Settings</span>
                <span className={`dot ${serverDot}`} aria-hidden="true" />
              </span>
              {props.desktopHost ? (
                <small>{props.desktopHost.replace(/^https?:\/\//, "")}</small>
              ) : (
                <small>{serverStatusText}</small>
              )}
            </span>
          </button>
          <button className="icon services-refresh" aria-label="Refresh server status and agent" title="Refresh"
            disabled={disabled} onClick={props.onRefresh}>
            <RefreshCw size={14} className={loading ? "spin" : ""} />
          </button>
        </div>
      </footer>
    </>
  );
}
