import { useNavigate } from "react-router";
import { ChevronRight } from "lucide-react";
import type { MetricsResponse } from "../hikari/api/generated/types.gen";
import PetAvatar from "../kadan/PetAvatar";
import { useAgentPet } from "../kadan/PetContext";
import { overallStatus, STATE_LABEL, STATE_TONE, toAbility } from "./abilities";
import { useServerAddress } from "./address";
import { formatBytes, storagePath, usePoll, type Agent, type Status } from "./api";
import { CopyField, Meter, PageHeader, StatusPill } from "./common";
import { useModels } from "./useModels";

interface Stats {
  resources: Record<string, number>;
  storage_bytes: number;
  database_bytes: number;
  trashed_resources: number;
  trashed_assets: number;
}

/** Plain names for memory meters; graphics cards are numbered from one for people. */
export function meterLabel(label: string): string {
  const gpu = /GPU (\d+)/i.exec(label);
  if (gpu) return `Graphics card ${Number(gpu[1]) + 1} memory`;
  if (/ram|system/i.test(label)) return "Computer memory";
  return label.replace(/ used$/i, "");
}

function AgentCard({ agent }: { agent: Agent }) {
  const navigate = useNavigate();
  const stats = usePoll<Stats>(`${storagePath(agent.key)}/stats`, 30000);
  const memories = stats.data?.resources.memory ?? 0;
  return (
    <button className="srv-card srv-agent-card" onClick={() => navigate("/server/agent")}>
      <PetAvatar sheet={useAgentPet(agent.key)} seed={agent.key} size={56} state="waving" />
      <span className="srv-agent-card-text">
        <strong>{agent.name}</strong>
        <span>
          {stats.data
            ? `${memories} ${memories === 1 ? "memory" : "memories"} · ${formatBytes(stats.data.storage_bytes + stats.data.database_bytes)} saved`
            : "Your agent’s memories, persona and history"}
        </span>
      </span>
      <ChevronRight size={18} aria-hidden="true" />
    </button>
  );
}

export default function HomePage({ agent }: { agent: Agent | null }) {
  const navigate = useNavigate();
  const address = useServerAddress();
  const status = usePoll<Status>("/kadan/api/status", 5000);
  const metrics = usePoll<MetricsResponse>("/hikari/v1/metrics", 5000);
  const models = useModels();
  const abilities = models.rows?.map(toAbility) ?? [];
  const storageReady = status.data ? status.data.tsuchi.state === "ready" : !status.error;
  const overall = models.rows ? overallStatus(abilities, storageReady) : null;

  return (
    <div className="srv-page">
      <PageHeader title="Home" />

      <section className={`srv-hero ${overall?.tone ?? "busy"}`} aria-live="polite">
        <span className="srv-hero-dot" aria-hidden="true" />
        <div>
          <h2>{overall?.title ?? (models.error ? "Kadan Server isn’t answering" : "Checking Kadan Server…")}</h2>
          <p>{overall?.detail ?? models.error}</p>
        </div>
      </section>

      <section className="srv-card">
        <header className="srv-card-head">
          <h2>Connect Kadan</h2>
          <p>To use this Kadan from another computer, open Kadan there and paste this address under <strong>Settings → Kadan Server → Change</strong>.</p>
        </header>
        {address && <CopyField value={address} label="server address" />}
      </section>

      <section className="srv-card">
        <header className="srv-card-head srv-card-head-row">
          <div>
            <h2>What your agent can do</h2>
            <p>Each ability uses an AI model on this computer.</p>
          </div>
          <button className="button" onClick={() => navigate("/server/models")}>Manage AI models</button>
        </header>
        {models.error && !models.rows && <p className="error" role="alert">{models.error}</p>}
        <ul className="srv-ability-grid">
          {abilities.map((ability) => (
            <li key={ability.id} className="srv-ability">
              <div className="srv-ability-top">
                <strong>{ability.name}</strong>
                <StatusPill tone={STATE_TONE[ability.state]}>
                  {ability.state === "downloading" ? `${STATE_LABEL.downloading} ${ability.progress}%` : STATE_LABEL[ability.state]}
                </StatusPill>
              </div>
              <p>{ability.description}</p>
              {(ability.state === "needs_download" || ability.state === "failed") && (
                <button className="button srv-ability-action" onClick={() => navigate("/server/models")}>
                  {ability.state === "failed" ? "Try again" : "Download"}
                </button>
              )}
            </li>
          ))}
        </ul>
      </section>

      <div className="srv-two">
        {agent && <AgentCard agent={agent} />}
        <section className="srv-card">
          <header className="srv-card-head">
            <h2>Memory in use</h2>
            <p>Includes everything running on this computer, not only Kadan.</p>
          </header>
          {metrics.data?.resources.length ? (
            metrics.data.resources.map((meter) => (
              <Meter key={meter.label} label={meterLabel(meter.label)} used={meter.used} total={meter.total} unit="GB" />
            ))
          ) : (
            <p className="field-hint">{metrics.error || "Checking memory…"}</p>
          )}
        </section>
      </div>
    </div>
  );
}
