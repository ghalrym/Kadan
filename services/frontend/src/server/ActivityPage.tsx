import { useState } from "react";
import type { RequestRecord, RequestsResponse } from "../hikari/api/generated/types.gen";
import { usePoll } from "./api";
import { PageHeader, StatusPill } from "./common";

/** What each kind of request did, in plain words. */
export const REQUEST_LABEL: Record<RequestRecord["type"], string> = {
  LLM: "Chat reply",
  Image: "Image",
  TTS: "Reading aloud",
  STT: "Dictation",
  Decision: "Decision",
};

export function requestOutcome(request: Pick<RequestRecord, "status">): { tone: "ok" | "warn" | "error"; label: string } {
  if (request.status >= 500) return { tone: "error", label: "Failed" };
  if (request.status >= 400) return { tone: "warn", label: "Couldn’t run" };
  return { tone: "ok", label: "Done" };
}

export function duration(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)} ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)} s`;
  return `${Math.floor(ms / 60000)} min ${Math.round((ms % 60000) / 1000)} s`;
}

function Details({ request }: { request: RequestRecord }) {
  const facts: [string, string][] = [];
  if (request.model) facts.push(["Model", request.model]);
  const firstWord = request.stream_ttft_ms ?? request.generation_ttft_ms;
  if (firstWord != null) facts.push(["First words after", duration(firstWord)]);
  if (request.decode_tokens_per_second != null) facts.push(["Speed", `${request.decode_tokens_per_second.toFixed(1)} word pieces per second`]);
  if (request.output_tokens != null) facts.push(["Length", `${request.output_tokens} word pieces`]);
  facts.push(["Request ID", request.id]);
  return (
    <div className="srv-request-details">
      <dl>
        {facts.map(([term, value]) => (
          <div key={term}><dt>{term}</dt><dd>{value}</dd></div>
        ))}
      </dl>
      {request.prompt && <><h3>Asked</h3><pre>{request.prompt}</pre></>}
      {request.output && <><h3>Answered</h3><pre>{request.output}</pre></>}
    </div>
  );
}

const PAGE = 25;

export default function ActivityPage() {
  const [type, setType] = useState<RequestRecord["type"] | "">("");
  const [offset, setOffset] = useState(0);
  const [open, setOpen] = useState<string | null>(null);
  const query = new URLSearchParams({ offset: String(offset), limit: String(PAGE) });
  if (type) query.set("type", type);
  const { data, error } = usePoll<RequestsResponse>(`/hikari/v1/requests?${query}`, 3000);

  return (
    <div className="srv-page">
      <PageHeader title="Activity" description="What the AI models on this computer have worked on recently. Kadan keeps a short history while it runs." />
      <div className="srv-filters" role="group" aria-label="Show">
        {(["", "LLM", "Image", "STT", "TTS", "Decision"] as const).map((value) => (
          <button key={value} className={type === value ? "selected" : ""} aria-pressed={type === value}
            onClick={() => { setType(value); setOffset(0); setOpen(null); }}>
            {value ? REQUEST_LABEL[value] : "Everything"}
          </button>
        ))}
      </div>
      {error && <p className="error" role="alert">{error}</p>}
      {!data && !error && <p className="srv-loading">Loading activity…</p>}
      {data && data.requests.length === 0 && <p className="srv-empty">Nothing yet. Activity shows up here as your agent works.</p>}
      {data && data.requests.length > 0 && (
        <ul className="srv-activity">
          {data.requests.map((request) => {
            const outcome = requestOutcome(request);
            const expanded = open === request.id;
            return (
              <li key={request.id}>
                <button className="srv-activity-row" aria-expanded={expanded} onClick={() => setOpen(expanded ? null : request.id)}>
                  <time dateTime={request.time}>{new Date(request.time).toLocaleTimeString([], { hour: "numeric", minute: "2-digit", second: "2-digit" })}</time>
                  <strong>{REQUEST_LABEL[request.type]}</strong>
                  <span className="srv-activity-time">{duration(request.latency_ms)}</span>
                  <StatusPill tone={outcome.tone}>{outcome.label}</StatusPill>
                </button>
                {expanded && <Details request={request} />}
              </li>
            );
          })}
        </ul>
      )}
      {data && (offset > 0 || data.total > PAGE) && (
        <div className="srv-pager">
          <button disabled={offset === 0} onClick={() => setOffset((value) => Math.max(0, value - PAGE))}>Newer</button>
          <span>Page {Math.floor(offset / PAGE) + 1} of {Math.max(1, Math.ceil(data.total / PAGE))}</span>
          <button disabled={offset + PAGE >= data.total} onClick={() => setOffset((value) => value + PAGE)}>Older</button>
        </div>
      )}
    </div>
  );
}
