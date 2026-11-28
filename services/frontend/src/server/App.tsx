import { useEffect, useState, type CSSProperties, type ReactNode } from "react";
import { Route, Routes, useLocation, useNavigate } from "react-router";
import { Activity, Cpu, ExternalLink, Home, Settings } from "lucide-react";
import PetAvatar from "../kadan/PetAvatar";
import { PetContext, useAgentPet } from "../kadan/PetContext";
import { agentHue } from "../kadan/pet";
import type { PetJob } from "../kadan/types";
import { getJson, kadanPath, type Agent } from "./api";
import ActivityPage from "./ActivityPage";
import AgentPage from "./AgentPage";
import HomePage from "./HomePage";
import ModelsPage from "./ModelsPage";
import SettingsPage from "./SettingsPage";

/** Inside the Kadan Server window there is nowhere to go back to, so links to the client are hidden. */
const embedded = typeof window !== "undefined" && window.self !== window.top;

function NavItem({ to, current, icon, children }: { to: string; current: boolean; icon: ReactNode; children: ReactNode }) {
  const navigate = useNavigate();
  return (
    <button className={`srv-nav-link ${current ? "selected" : ""}`} aria-current={current ? "page" : undefined}
      onClick={() => navigate(to)}>
      {icon}
      <span>{children}</span>
    </button>
  );
}

function AgentNavItem({ agent, current }: { agent: Agent; current: boolean }) {
  const navigate = useNavigate();
  return (
    <button className={`srv-nav-link srv-nav-agent ${current ? "selected" : ""}`} aria-current={current ? "page" : undefined}
      onClick={() => navigate("/server/agent")}>
      <PetAvatar sheet={useAgentPet(agent.key)} seed={agent.key} size={26} />
      <span>{agent.name}</span>
    </button>
  );
}

export default function ServerApp() {
  const { pathname } = useLocation();
  const [agent, setAgent] = useState<Agent | null>(null);
  const [agentError, setAgentError] = useState("");
  const [pet, setPet] = useState<PetJob | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    getJson<Agent[]>("/kadan/api/agents", controller.signal)
      .then(([first]) => {
        setAgent(first ?? null);
        if (first) return getJson<PetJob | null>(`${kadanPath(first.key)}/pet`, controller.signal).then(setPet);
      })
      .catch((error) => {
        if (!controller.signal.aborted) setAgentError((error as Error).message);
      });
    return () => controller.abort();
  }, []);

  const section = pathname.replace(/^\/server\/?/, "").split("/")[0];
  const hue = agent ? agentHue(agent.key) : 252;

  return (
    <PetContext.Provider value={agent ? { [agent.key]: pet } : {}}>
      <div className="app srv-app" style={{ "--agent-h": hue } as CSSProperties}>
        <aside className="sidebar srv-sidebar" aria-label="Kadan Server">
          <div className="panel-head">
            <span className="brand">
              <span className="logo" aria-hidden="true"><i /><i /><i /><i /></span>
              Kadan Server
            </span>
          </div>
          <nav className="srv-nav" aria-label="Sections">
            <NavItem to="/server" current={section === ""} icon={<Home size={16} />}>Home</NavItem>
            <NavItem to="/server/models" current={section === "models"} icon={<Cpu size={16} />}>AI models</NavItem>
            {agent && <AgentNavItem agent={agent} current={section === "agent"} />}
            <NavItem to="/server/activity" current={section === "activity"} icon={<Activity size={16} />}>Activity</NavItem>
            <NavItem to="/server/settings" current={section === "settings"} icon={<Settings size={16} />}>Settings</NavItem>
          </nav>
          {!embedded && (
            <a className="srv-nav-link srv-open-kadan" href="/">
              <ExternalLink size={16} />
              <span>Open Kadan</span>
            </a>
          )}
        </aside>
        <main id="main" className="srv-main" tabIndex={-1}>
          {agentError && !agent && <div className="banner" role="alert">{agentError}</div>}
          <Routes>
            <Route index element={<HomePage agent={agent} />} />
            <Route path="models" element={<ModelsPage />} />
            <Route path="agent/*" element={agent ? <AgentPage agent={agent} /> : <p className="srv-loading">Loading your agent…</p>} />
            <Route path="activity" element={<ActivityPage />} />
            <Route path="settings" element={<SettingsPage />} />
            <Route path="*" element={<HomePage agent={agent} />} />
          </Routes>
        </main>
      </div>
    </PetContext.Provider>
  );
}
