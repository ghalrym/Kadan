import { useState } from "react";
import {
  ArchiveRestore,
  ChevronDown,
  ChevronRight,
  Clock,
  LayoutGrid,
  MessageSquare,
  Plus,
  Puzzle,
  RefreshCw,
  Search,
  Settings,
  SlidersHorizontal,
  X,
} from "lucide-react";
import Pet from "./Pet";
import { useAgentPet } from "./PetContext";
import type { Agent, Status } from "./types";
import {
  formatSnoozeLabel,
  groupThreadsByBucket,
  pendingRequestsFor,
  type WorkThread,
} from "./work";

export type Page =
  | { name: "chat" }
  | { name: "space" }
  | { name: "plugins" }
  | { name: "agent" }
  | { name: "work-thread"; threadId: string }
  | { name: "work-setup"; tab?: "new" | "projects" | "harnesses" };

interface Props {
  agent: Agent | null;
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
  onOpenWorkSetup?: (tab: "new" | "projects" | "harnesses") => void;
  onUnsettleWorkThread?: (threadId: string) => void;
  onWakeWorkThread?: (threadId: string) => void;
}

export default function Sidebar(props: Props) {
  const { agent, page, status, loading, disabled } = props;
  const sheet = useAgentPet(agent?.key ?? "");
  const key = agent?.key ?? "";
  const [settledOpen, setSettledOpen] = useState(false);
  const [settledQuery, setSettledQuery] = useState("");
  const workThreads = props.workThreads ?? [];
  const buckets = groupThreadsByBucket(workThreads, settledQuery);
  const settledTotal = workThreads.filter((t) => t.bucket === "settled").length;

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

  function renderThreadRow(thread: WorkThread, extra?: React.ReactNode) {
    const selected = selectedWorkId === thread.id;
    const pendingCount = pendingRequestsFor(thread).length;
    return (
      <div key={thread.id} className={`chat-row work-thread-row ${selected ? "selected" : ""}`}>
        <button
          className="chat-row-open"
          aria-current={selected ? "page" : undefined}
          onClick={() => props.onOpenWorkThread?.(thread.id)}
        >
          <span className="work-row-main">
            <span
              className={`work-row-dot status-${thread.run_status}`}
              aria-hidden="true"
            />
            <span className="chat-row-title">{thread.title}</span>
          </span>
          {pendingCount > 0 ? (
            <span className="work-row-badge warn">Needs input</span>
          ) : thread.bucket === "snoozed" && thread.snoozed_until ? (
            <span className="work-row-time">{formatSnoozeLabel(thread.snoozed_until)}</span>
          ) : null}
        </button>
        {extra}
      </div>
    );
  }

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
        <button className={page.name === "chat" ? "selected" : ""} disabled={!key}
          aria-current={page.name === "chat" ? "page" : undefined} onClick={() => props.onPage("chat")}>
          <MessageSquare size={16} />
          Chat
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

      <section className="panel-scroll work-sidebar" aria-label="Work">
        <div className="panel-section-head work-sidebar-head">
          <span className="panel-label">WORK</span>
          <div className="work-sidebar-actions">
            <button
              className="icon work-mini-btn"
              aria-label="New work thread"
              title="New work thread"
              disabled={!key}
              onClick={() => props.onOpenWorkSetup?.("new")}
            >
              <Plus size={14} />
            </button>
            <button
              className="icon work-mini-btn"
              aria-label="Work projects and harnesses"
              title="Projects & Harnesses"
              disabled={!key}
              onClick={() => props.onOpenWorkSetup?.("projects")}
            >
              <SlidersHorizontal size={13} />
            </button>
          </div>
        </div>

        <div className="panel-section work-bucket" data-bucket="default">
          <div className="work-bucket-head">
            <span className="work-bucket-label">Default</span>
            {buckets.default.length > 0 && (
              <span className="work-bucket-count">{buckets.default.length}</span>
            )}
          </div>
          {buckets.default.length === 0 ? (
            <p className="panel-hint work-empty-hint">No threads waiting</p>
          ) : (
            buckets.default.map((thread) => renderThreadRow(thread))
          )}
        </div>

        <div className="panel-section work-bucket" data-bucket="working">
          <div className="work-bucket-head">
            <span className="work-bucket-label">Working</span>
            {buckets.working.length > 0 && (
              <span className="work-bucket-count">{buckets.working.length}</span>
            )}
          </div>
          {buckets.working.length === 0 ? (
            <p className="panel-hint work-empty-hint">Nothing running</p>
          ) : (
            buckets.working.map((thread) => renderThreadRow(thread))
          )}
        </div>

        <div className="panel-section work-bucket" data-bucket="snoozed">
          <div className="work-bucket-head">
            <span className="work-bucket-label">Snoozed</span>
            {buckets.snoozed.length > 0 && (
              <span className="work-bucket-count">{buckets.snoozed.length}</span>
            )}
          </div>
          {buckets.snoozed.length > 0 &&
            buckets.snoozed.map((thread) =>
              renderThreadRow(
                thread,
                <button
                  className="chat-row-more"
                  aria-label={`Wake ${thread.title}`}
                  title="Wake now"
                  onClick={(e) => {
                    e.stopPropagation();
                    props.onWakeWorkThread?.(thread.id);
                  }}
                >
                  <Clock size={12} />
                </button>,
              ),
            )}
        </div>

        <div className="panel-section work-bucket" data-bucket="settled">
          <button
            className="work-bucket-toggle"
            aria-expanded={settledOpen}
            onClick={() => setSettledOpen((open) => !open)}
          >
            {settledOpen ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
            <span className="work-bucket-label">Settled</span>
            {settledTotal > 0 && <span className="work-bucket-count">{settledTotal}</span>}
          </button>
          {settledOpen && (
            <>
              {settledTotal > 0 && (
                <div className="work-settled-search">
                  <Search size={12} />
                  <input
                    type="search"
                    aria-label="Search settled threads"
                    placeholder="Filter settled…"
                    value={settledQuery}
                    onChange={(e) => setSettledQuery(e.target.value)}
                  />
                </div>
              )}
              {buckets.settled.length === 0 ? (
                <p className="panel-hint work-empty-hint">
                  {settledTotal > 0 ? "No matches" : "No settled threads"}
                </p>
              ) : (
                buckets.settled.map((thread) =>
                  renderThreadRow(
                    thread,
                    <button
                      className="chat-row-more"
                      aria-label={`Unsettle ${thread.title}`}
                      title="Move back to Default"
                      onClick={(e) => {
                        e.stopPropagation();
                        props.onUnsettleWorkThread?.(thread.id);
                      }}
                    >
                      <ArchiveRestore size={12} />
                    </button>,
                  ),
                )
              )}
            </>
          )}
        </div>
      </section>

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
        <div className="sidebar-server-row">
          {props.onServerSettings ? (
            <button
              className="server-settings-trigger"
              onClick={props.onServerSettings}
              title={serverTooltip}
            >
              <Settings size={15} />
              <span className="server-settings-nav">
                <span className="server-settings-label">
                  <span>Server settings</span>
                  <span className={`dot ${serverDot}`} aria-hidden="true" />
                </span>
                {props.desktopHost && (
                  <small>{props.desktopHost.replace(/^https?:\/\//, "")}</small>
                )}
              </span>
            </button>
          ) : (
            <span className="server-status-label" title={serverTooltip}>
              <span className={`dot ${serverDot}`} aria-hidden="true" />
              {serverStatusText}
            </span>
          )}
          <button className="icon services-refresh" aria-label="Refresh server status and agent" title="Refresh"
            disabled={disabled} onClick={props.onRefresh}>
            <RefreshCw size={14} className={loading ? "spin" : ""} />
          </button>
        </div>
      </footer>
    </>
  );
}
