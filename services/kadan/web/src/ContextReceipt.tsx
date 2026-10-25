import { useEffect, useRef, useState } from "react";
import { agentPath, api } from "./api";

interface SelectionProgress {
  stage: "selected" | "tool_result_recorded" | "model_response_recorded";
  truncated: boolean | null;
}
interface SelectedContext {
  id: string;
  kind: "space" | "memory" | "workflow";
  title: string;
  version: number;
}
export interface RunContext {
  run_id: string;
  status: string;
  selected_references: SelectedContext[];
  selected_workflow: SelectedContext | null;
  receipt: { references: SelectionProgress | null; workflow: SelectionProgress | null } | null;
}
function progressLabel(progress: SelectionProgress | null | undefined) {
  if (!progress) return "Receipt unavailable";
  if (progress.stage === "selected") return "Selected; no tool result recorded. Truncation unknown";
  const recorded = progress.stage === "model_response_recorded"
    ? "Included in a model call with a recorded response"
    : "Tool result recorded; model handoff unconfirmed";
  return `${recorded}. ${progress.truncated === null ? "Truncation unknown" : progress.truncated ? "Truncated" : "Full snapshot"}`;
}
export default function ContextReceipt({ agentKey, conversationId, runId }: {
  agentKey: string; conversationId: string; runId: string;
}) {
  const [value, setValue] = useState<RunContext | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [open, setOpen] = useState(false);
  const request = useRef(0);
  const path = `${agentPath(agentKey)}/conversations/${conversationId}/runs/${runId}/context`;
  useEffect(() => {
    setValue(null); setError(""); setLoading(false); setOpen(false);
    return () => { request.current++; };
  }, [path]);
  async function load() {
    const order = ++request.current;
    setOpen(true); setLoading(true); setError("");
    try {
      const next = await api<RunContext>(path);
      if (order === request.current) setValue(next);
    } catch (e) {
      if (order === request.current) setError((e as Error).message);
    } finally {
      if (order === request.current) setLoading(false);
    }
  }
  return <div className="context-receipt">
    <button className="pet-edit-link" aria-expanded={open}
      onClick={() => { if (!value) void load(); else setOpen(!open); }}>Context receipt</button>
    {open && <div className="notice">
      <button className="tag" disabled={loading} onClick={() => void load()}>{loading ? "Loading…" : "Refresh"}</button>
      {error && <p role="alert">{value ? "Showing previous receipt. Refresh failed: " : "Receipt unavailable: "}{error}</p>}
      {value && <>
        <p>Saved references and workflow for this request</p>
        {!value.receipt && <p>Receipt unavailable{value.status === "queued" ? "; request queued or not prepared" : " for this run"}.</p>}
        {!value.selected_references.length && !value.selected_workflow && <p>No references or workflow selected.</p>}
        {value.selected_references.map(item => <p key={item.id}>
          {item.title} · {item.kind} · v{item.version} — {progressLabel(value.receipt?.references)}
        </p>)}
        {value.selected_workflow && <p>{value.selected_workflow.title} · workflow · v{value.selected_workflow.version} — {progressLabel(value.receipt?.workflow)}</p>}
        <p>A recorded model response confirms the recorded tool result entered that call. It does not confirm the model used, understood, or followed it.</p>
      </>}
    </div>}
  </div>;
}
