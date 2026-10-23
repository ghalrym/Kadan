import { useEffect, useRef, useState } from "react";
import {
  Activity,
  ChartNoAxesCombined,
  Folder,
  Images,
  Layers3,
  Menu,
  MessagesSquare,
  NotebookText,
  PanelsTopLeft,
  Paperclip,
  Plus,
  RefreshCw,
  Search,
  Trash2,
  UserRound,
  X,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { agentPath, request, useLoad } from "./api";
import { AgentForm } from "./AgentForm";
import { AssetEditor } from "./AssetEditor";
import { Dialog, ErrorNotice, Loading } from "./components";
import { AssetsPage, ResourcesPage, TrashPage } from "./LibraryPages";
import { ActivityPage, OverviewPage } from "./OverviewPages";
import { ResourceEditor } from "./ResourceEditor";
import type { Agent, Asset, Kind, Resource, View } from "./types";

const navigation: { view: View; label: string; icon: LucideIcon }[] = [
  { view: "overview", label: "Overview", icon: ChartNoAxesCombined },
  { view: "persona", label: "Personas", icon: UserRound },
  { view: "memory", label: "Memories", icon: NotebookText },
  { view: "space", label: "Spaces", icon: Folder },
  { view: "deck", label: "Slide decks", icon: PanelsTopLeft },
  { view: "images", label: "Images", icon: Images },
  { view: "attachments", label: "Attachments", icon: Paperclip },
  { view: "conversation", label: "Conversations", icon: MessagesSquare },
  { view: "activity", label: "Activity", icon: Activity },
  { view: "trash", label: "Trash", icon: Trash2 },
];
type Editor =
  | { type: "resource"; value: Resource | Kind }
  | { type: "asset"; value: Asset | null; imagesOnly: boolean };
const initialParameters = new URLSearchParams(window.location.search);

export function App() {
  const [agentsVersion, setAgentsVersion] = useState(0);
  const [agentKey, setAgentKey] = useState(
    initialParameters.get("agent") ||
      localStorage.getItem("tsuchi-namespace") ||
      "",
  );
  const [newAgent, setNewAgent] = useState(false);
  const agents = useLoad(
    () => request<Agent[]>("/api/agents"),
    [agentsVersion],
  );
  const onCreated = (agent: Agent) => {
    setAgentKey(agent.key);
    setAgentsVersion((value) => value + 1);
    setNewAgent(false);
  };
  const activeAgent =
    agents.data?.find((agent) => agent.key === agentKey) || agents.data?.[0];
  if (!agents.data?.length)
    return (
      <main className="onboarding">
        <div className="brand">
          <Layers3 size={27} />
          Tsuchi
        </div>
        {agents.loading ? (
          <Loading />
        ) : agents.error ? (
          <>
            <ErrorNotice message={agents.error} />
            <button onClick={() => setAgentsVersion((value) => value + 1)}>
              <RefreshCw size={16} />
              Retry
            </button>
          </>
        ) : (
          <>
            <h1>Create namespace</h1>
            <AgentForm onCreated={onCreated} />
          </>
        )}
      </main>
    );
  return (
    <>
      {activeAgent && (
        <NamespaceApp
          key={activeAgent.key}
          agent={activeAgent}
          agents={agents.data}
          onNewAgent={() => setNewAgent(true)}
          onSwitch={(key) => {
            setAgentKey(key);
            localStorage.setItem("tsuchi-namespace", key);
            window.history.replaceState(null, "", "/");
          }}
        />
      )}
      {newAgent && (
        <Dialog title="New namespace" onClose={() => setNewAgent(false)}>
          <AgentForm onCreated={onCreated} />
        </Dialog>
      )}
    </>
  );
}
function NamespaceApp({
  agent,
  agents,
  onSwitch,
  onNewAgent,
}: {
  agent: Agent;
  agents: Agent[];
  onSwitch: (key: string) => void;
  onNewAgent: () => void;
}) {
  const [view, setView] = useState<View>("overview");
  const [refresh, setRefresh] = useState(0);
  const [editor, setEditor] = useState<Editor | null>(null);
  const editorIntent = useRef(0);
  const [query, setQuery] = useState("");
  const [search, setSearch] = useState("");
  const [menuOpen, setMenuOpen] = useState(false);
  const [error, setError] = useState("");
  const health = useLoad(
    () => request<{ status: string; version: string }>("/api/health"),
    [refresh],
  );
  useEffect(() => {
    if (view !== "overview" && view !== "activity") return;
    const timer = window.setInterval(
      () => setRefresh((value) => value + 1),
      30000,
    );
    return () => window.clearInterval(timer);
  }, [view]);
  useEffect(() => {
    const intent = ++editorIntent.current;
    const id = initialParameters.get("resource");
    if (id && initialParameters.get("agent") === agent.key)
      request<Resource>(
        `${agentPath(agent.key)}/resources/${encodeURIComponent(id)}`,
      )
        .then((resource) => {
          if (intent === editorIntent.current)
            setEditor({ type: "resource", value: resource });
        })
        .catch((error) => {
          if (intent === editorIntent.current)
            setError(
              error instanceof Error ? error.message : "Unable to open resource.",
            );
        });
    return () => {
      editorIntent.current++;
    };
  }, [agent.key]);
  const navigate = (next: View) => {
    editorIntent.current++;
    setView(next);
    setMenuOpen(false);
  };
  const openResource = async (resource: Resource) => {
    const intent = ++editorIntent.current;
    setError("");
    try {
      const value = await request<Resource>(
        `${agentPath(agent.key)}/resources/${resource.id}`,
      );
      if (intent === editorIntent.current)
        setEditor({ type: "resource", value });
    } catch (error) {
      if (intent !== editorIntent.current) return;
      setError(
        error instanceof Error ? error.message : "Unable to open resource.",
      );
    }
  };
  const openAsset = async (asset: Asset) => {
    const intent = ++editorIntent.current;
    setError("");
    try {
      const value = await request<Asset>(
        `${agentPath(agent.key)}/assets/${asset.id}`,
      );
      if (intent === editorIntent.current)
        setEditor({ type: "asset", value, imagesOnly: asset.is_image });
    } catch (error) {
      if (intent !== editorIntent.current) return;
      setError(
        error instanceof Error ? error.message : "Unable to open asset.",
      );
    }
  };
  const createResource = (kind: Kind) => {
    editorIntent.current++;
    setEditor({ type: "resource", value: kind });
  };
  const closeEditor = () => {
    editorIntent.current++;
    setEditor(null);
  };
  const content = () => {
    if (view === "overview")
      return (
        <OverviewPage
          agentKey={agent.key}
          refresh={refresh}
          navigate={navigate}
        />
      );
    if (view === "activity")
      return <ActivityPage agentKey={agent.key} refresh={refresh} />;
    if (view === "images" || view === "attachments")
      return (
        <AssetsPage
          agentKey={agent.key}
          images={view === "images"}
          refresh={refresh}
          onOpen={openAsset}
          onNew={() => {
            editorIntent.current++;
            setEditor({
              type: "asset",
              value: null,
              imagesOnly: view === "images",
            });
          }}
        />
      );
    if (view === "trash")
      return (
        <TrashPage
          agentKey={agent.key}
          refresh={refresh}
          onResource={openResource}
          onAsset={openAsset}
        />
      );
    return (
      <ResourcesPage
        agentKey={agent.key}
        kind={view === "search" ? undefined : view}
        search={view === "search" ? search : undefined}
        refresh={refresh}
        onOpen={openResource}
        onNew={createResource}
      />
    );
  };
  return (
    <div className="app-shell">
      <a href="#main" className="skip-link">
        Skip to content
      </a>
      {menuOpen && (
        <button
          className="sidebar-scrim"
          aria-label="Close navigation"
          onClick={() => setMenuOpen(false)}
        />
      )}
      <aside className={`sidebar ${menuOpen ? "sidebar-open" : ""}`}>
        <div className="brand">
          <Layers3 size={26} />
          Tsuchi
          <button
            className="icon-button mobile-only"
            aria-label="Close navigation"
            onClick={() => setMenuOpen(false)}
          >
            <X size={20} />
          </button>
        </div>
        <div className="namespace-control">
          <label>
            Namespace
            <select
              value={agent.key}
              onChange={(event) => onSwitch(event.target.value)}
            >
              {agents.map((item) => (
                <option key={item.key} value={item.key}>
                  {item.name}
                </option>
              ))}
            </select>
          </label>
          <button
            className="icon-button"
            aria-label="Create namespace"
            onClick={onNewAgent}
          >
            <Plus size={18} />
          </button>
        </div>
        <nav aria-label="Main navigation">
          {navigation.map((item) => (
            <button
              key={item.view}
              className={view === item.view ? "nav-item active" : "nav-item"}
              aria-current={view === item.view ? "page" : undefined}
              onClick={() => navigate(item.view)}
            >
              <item.icon size={19} />
              {item.label}
            </button>
          ))}
        </nav>
        <div className="sidebar-status">
          <span className={`status-dot ${health.error ? "unavailable" : ""}`} />
          <span>
            {health.error
              ? "Service unavailable"
              : health.data
                ? "Local storage"
                : "Connecting…"}
          </span>
          {health.data && (
            <span className="version">v{health.data.version}</span>
          )}
        </div>
      </aside>
      <div className="workspace">
        <header className="topbar">
          <button
            className="icon-button mobile-only"
            aria-label="Open navigation"
            onClick={() => setMenuOpen(true)}
          >
            <Menu size={21} />
          </button>
          <form
            className="global-search"
            onSubmit={(event) => {
              event.preventDefault();
              if (query.trim()) {
                if (view === "search" && query.trim() === search)
                  setRefresh((value) => value + 1);
                setSearch(query.trim());
                navigate("search");
              }
            }}
          >
            <Search size={18} />
            <input
              aria-label="Search all resources"
              placeholder="Search all resources"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
            />
            <button type="submit" aria-label="Search">
              <ArrowSearch />
            </button>
          </form>
          <span className="namespace-key mono">{agent.key}</span>
          <button
            className="icon-button"
            aria-label="Refresh data"
            onClick={() => setRefresh((value) => value + 1)}
          >
            <RefreshCw size={18} />
          </button>
        </header>
        <main id="main" className="main-content">
          <ErrorNotice message={error} />
          <div key={`${view}-${search}`}>{content()}</div>
        </main>
      </div>
      {editor?.type === "resource" && (
        <ResourceEditor
          agentKey={agent.key}
          initial={editor.value}
          onClose={closeEditor}
          onChanged={() => setRefresh((value) => value + 1)}
        />
      )}
      {editor?.type === "asset" && (
        <AssetEditor
          agentKey={agent.key}
          initial={editor.value}
          imagesOnly={editor.imagesOnly}
          onClose={closeEditor}
          onChanged={() => setRefresh((value) => value + 1)}
        />
      )}
    </div>
  );
}
function ArrowSearch() {
  return <span aria-hidden="true">↵</span>;
}
