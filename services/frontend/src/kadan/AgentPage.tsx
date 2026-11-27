import { useEffect, useState } from "react";
import { ArrowLeft, Brain, ChevronRight, Palette, Pencil, RefreshCw, Search, Server, Trash2, UserRound } from "lucide-react";
import { agentPath, api } from "./api";
import Pet, { useMood } from "./Pet";
import { useAgentPet } from "./PetContext";
import MemoryEditor from "./MemoryEditor";
import type { Agent, Resource } from "./types";

interface Props {
  agent: Agent;
  disabled: boolean;
  onEditPet: () => void;
  onEditPersona: () => void;
  onRename: () => void;
  onRunner: () => void;
  onDelete: () => void;
}

function Memories({ agentKey, agentName }: { agentKey: string; agentName: string }) {
  const [query, setQuery] = useState("");
  const [rows, setRows] = useState<Resource[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [attempt, setAttempt] = useState(0);
  const [selected, setSelected] = useState<Resource | null>(null);
  useEffect(() => {
    let current = true;
    setLoading(true);
    setError("");
    const timer = setTimeout(() => {
      api<Resource[]>(`${agentPath(agentKey)}/memories?limit=200&q=${encodeURIComponent(query)}`)
        .then((values) => { if (current) setRows(values); })
        .catch((e) => { if (current) setError((e as Error).message); })
        .finally(() => { if (current) setLoading(false); });
    }, 200);
    return () => {
      current = false;
      clearTimeout(timer);
    };
  }, [agentKey, query, attempt]);
  if (selected) {
    return (
      <div className="memory-detail">
        <button className="page-back" onClick={() => setSelected(null)}>
          <ArrowLeft size={16} />
          All memories
        </button>
        <MemoryEditor key={`${agentKey}:${selected.id}`} agentKey={agentKey} memory={selected}
          onSaved={(value) => {
            setSelected(value);
            setRows((items) => items.map((item) => item.id === value.id ? value : item));
          }} />
      </div>
    );
  }
  return (
    <>
      <div className="memory-tools">
        <label className="search-field">
          <Search size={16} />
          <span className="sr-only">Search memories</span>
          <input type="search" placeholder="Search memories" value={query} onChange={(event) => setQuery(event.target.value)} />
        </label>
        <button className="icon" aria-label="Refresh memories" title="Refresh" disabled={loading}
          onClick={() => setAttempt((n) => n + 1)}>
          <RefreshCw size={15} className={loading ? "spin" : ""} />
        </button>
      </div>
      {error ? (
        <div role="alert" className="error feedback">
          {error}
          <button onClick={() => setAttempt((n) => n + 1)}>Retry</button>
        </div>
      ) : loading ? (
        <p className="page-empty" role="status">Loading memories…</p>
      ) : rows.length === 0 ? (
        <p className="page-empty">
          {query ? "No matching memories." : `${agentName} hasn’t saved any memories yet.`}
        </p>
      ) : (
        <ul className="memory-list">
          {rows.map((memory) => (
            <li key={memory.id}>
              <button onClick={() => setSelected(memory)}>
                <Brain size={16} aria-hidden="true" />
                <span className="memory-text">
                  <strong>{memory.title}</strong>
                  {memory.markdown.trim() && <span>{memory.markdown.replace(/\s+/g, " ").slice(0, 140)}</span>}
                </span>
                <ChevronRight size={14} aria-hidden="true" />
              </button>
            </li>
          ))}
        </ul>
      )}
    </>
  );
}

/** Everything about the agent itself: its look, name, memories and where its work runs. */
export default function AgentPage({ agent, disabled, onEditPet, onEditPersona, onRename, onRunner, onDelete }: Props) {
  const sheet = useAgentPet(agent.key);
  const [mood, react] = useMood("idle");
  return (
    <section className="page agent-page" aria-labelledby="agent-title">
      <div className="page-inner narrow">
        <header className="agent-hero">
          <button className="agent-hero-pet" aria-label={`Edit ${agent.name}’s pet`}
            onPointerEnter={() => react("waving")} onClick={() => { react("jumping"); onEditPet(); }}>
            <Pet sheet={sheet} state={mood} size={128} look grounded />
          </button>
          <div className="agent-hero-text">
            <h1 id="agent-title">{agent.name}</h1>
            <code>{agent.key}</code>
          </div>
          <div className="agent-hero-actions">
            <button onClick={onEditPet}><Palette size={15} />Edit pet</button>
            <button disabled={disabled} onClick={onEditPersona}><UserRound size={15} />Persona</button>
            <button disabled={disabled} onClick={onRename}><Pencil size={15} />Rename</button>
          </div>
        </header>
        <section className="page-card" aria-labelledby="memories-title">
          <header>
            <h2 id="memories-title">Memories</h2>
            <p>Things {agent.name} chose to remember. Open one to correct it.</p>
          </header>
          <Memories agentKey={agent.key} agentName={agent.name} />
        </section>
        <section className="page-card" aria-labelledby="runner-title">
          <header>
            <h2 id="runner-title">Runner</h2>
            <p>Choose which runner works on {agent.name}’s replies.</p>
          </header>
          <div>
            <button disabled={disabled} onClick={onRunner}><Server size={15} />Manage runner</button>
          </div>
        </section>
        <section className="page-card danger-card" aria-labelledby="delete-agent-section-title">
          <header>
            <h2 id="delete-agent-section-title">Delete agent</h2>
            <p>
              Permanently delete {agent.name}, all saved memories, chat history, persona, Space items,
              and pets to start fresh with a new agent.
            </p>
          </header>
          <div>
            <button className="danger-button" disabled={disabled} onClick={onDelete}>
              <Trash2 size={15} />
              Delete agent
            </button>
          </div>
        </section>
      </div>
    </section>
  );
}
