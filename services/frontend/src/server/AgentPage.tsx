import { Route, Routes, useLocation, useNavigate } from "react-router";
import { Download } from "lucide-react";
import PetAvatar from "../kadan/PetAvatar";
import { useAgentPet } from "../kadan/PetContext";
import { storagePath, type Agent } from "./api";
import AgentHistory from "./AgentHistory";
import AgentMemories from "./AgentMemories";
import AgentPersona from "./AgentPersona";
import AgentTrash from "./AgentTrash";

const TABS = [
  ["", "Memories"],
  ["persona", "Persona"],
  ["history", "Chat history"],
  ["trash", "Trash"],
  ["backup", "Backup"],
] as const;

function Backup({ agent }: { agent: Agent }) {
  return (
    <section className="srv-card">
      <header className="srv-card-head">
        <h2>Download a backup</h2>
        <p>
          Saves a copy of everything {agent.name} has stored: memories, persona, chat history, pages, slides and files.
          Keep it somewhere safe. Downloading it doesn’t change anything in Kadan.
        </p>
      </header>
      <a className="primary srv-link-button" href={`${storagePath(agent.key)}/export`} download>
        <Download size={15} />
        Download backup
      </a>
    </section>
  );
}

export default function AgentPage({ agent }: { agent: Agent }) {
  const navigate = useNavigate();
  const { pathname } = useLocation();
  const current = pathname.replace(/^\/server\/agent\/?/, "").split("/")[0];
  return (
    <div className="srv-page">
      <header className="srv-agent-head">
        <PetAvatar sheet={useAgentPet(agent.key)} seed={agent.key} size={64} state="waving" />
        <div>
          <h1>{agent.name}</h1>
          <p>Everything {agent.name} remembers and has saved.</p>
        </div>
      </header>
      <nav className="srv-tabs" aria-label={`${agent.name}’s data`}>
        {TABS.map(([path, label]) => (
          <button key={path} className={current === path ? "selected" : ""} aria-current={current === path ? "page" : undefined}
            onClick={() => navigate(path ? `/server/agent/${path}` : "/server/agent")}>
            {label}
          </button>
        ))}
      </nav>
      <Routes>
        <Route index element={<AgentMemories agent={agent} />} />
        <Route path="persona" element={<AgentPersona agent={agent} />} />
        <Route path="history" element={<AgentHistory agent={agent} />} />
        <Route path="trash" element={<AgentTrash agent={agent} />} />
        <Route path="backup" element={<Backup agent={agent} />} />
      </Routes>
    </div>
  );
}
