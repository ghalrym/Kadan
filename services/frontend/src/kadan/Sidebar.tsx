import {
  LayoutGrid,
  MessageSquare,
  Puzzle,
  RefreshCw,
  Settings,
  X,
} from "lucide-react";
import Pet from "./Pet";
import { useAgentPet } from "./PetContext";
import type { Agent, Status } from "./types";

export type Page = { name: "chat" } | { name: "space" } | { name: "plugins" } | { name: "agent" };

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
}

export default function Sidebar(props: Props) {
  const { agent, page, status, loading, disabled } = props;
  const sheet = useAgentPet(agent?.key ?? "");
  const key = agent?.key ?? "";
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
      <div className="panel-spacer" />
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
