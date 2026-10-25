import { useEffect, useRef, useState } from "react";
import { agentPath, api } from "./api";
import type { SelectedReference } from "./RequestReferences";
import type { WorkflowSelection } from "./workflows";

interface Advice {
  model_id: string | null;
  context_limit: number | null;
  context_source: "effective" | "configured" | "architecture" | "unknown";
  approximate_character_budget: number;
  task_characters: number;
  instruction_characters: number;
  selected_characters: number;
  tool_schema_characters: number;
  minimum_exceeds_budget: boolean;
  selected_material_risk: boolean;
}
interface Props {
  agentKey: string;
  conversationId: string;
  version: number;
  task: string;
  persona: string;
  references: SelectedReference[];
  workflow?: WorkflowSelection;
  disabled: boolean;
}

export default function ContextCheck({ agentKey, conversationId, version, task, persona, references, workflow, disabled }: Props) {
  const body = {
    content: task.trim(), expected_version: version, persona_id: persona || null,
    references: references.map(({ resource_id, expected_version }) => ({ resource_id, expected_version })),
    workflow_id: workflow?.id ?? null, workflow_version: workflow?.version ?? null,
  };
  const signature = JSON.stringify([agentKey, conversationId, body]);
  const current = useRef(signature);
  current.current = signature;
  const attempt = useRef(0);
  const [result, setResult] = useState<{ signature: string; advice?: Advice; error?: string } | null>(null);
  const [checking, setChecking] = useState<string | null>(null);
  useEffect(() => {
    ++attempt.current;
    setResult(null);
    setChecking(null);
  }, [signature]);
  async function check() {
    const request = ++attempt.current;
    setChecking(signature);
    setResult(null);
    try {
      const advice = await api<Advice>(`${agentPath(agentKey)}/conversations/${conversationId}/context-advice`, body);
      if (request === attempt.current && current.current === signature) setResult({ signature, advice });
    } catch (error) {
      if (request === attempt.current && current.current === signature) setResult({ signature, error: (error as Error).message });
    } finally { if (request === attempt.current) setChecking(null); }
  }
  const visible = result?.signature === signature ? result : null;
  const advice = visible?.advice;
  return <div className="context-check">
    <button type="button" className="pet-edit-link" disabled={disabled || !body.content || checking === signature} onClick={() => void check()}>
      {checking === signature ? "Checking context…" : workflow ? "Check workflow context" : "Check context"}
    </button>
    {visible?.error && <p className="notice" role="status">Context check unavailable: {visible.error} Your draft is unchanged; Send still validates the request.</p>}
    {advice && <details open>
      <summary>{advice.minimum_exceeds_budget ? "Task, instructions and tools exceed the approximate budget" : advice.selected_material_risk ? "Selected material may exceed the approximate budget" : "Context measurements"}</summary>
      <p>At last check: {advice.model_id || "model not reported"} · {advice.context_limit === null ? "Model window unknown" : `${advice.context_limit.toLocaleString()} token window (${advice.context_source})`}
        {advice.context_source === "configured" || advice.context_source === "architecture" ? "; advisory before loading" : ""}.
        {` Approximate character budget: ${advice.approximate_character_budget.toLocaleString()}${advice.context_source === "unknown" ? " (fallback)" : ""}.`}</p>
      <p>{`Task: ${advice.task_characters.toLocaleString()} characters · Shared instructions and persona: ${advice.instruction_characters.toLocaleString()} · Selected text: ${advice.selected_characters.toLocaleString()} · Request tool schemas: ${advice.tool_schema_characters.toLocaleString()}.`}</p>
      <p>{workflow ? "Includes this saved workflow. " : "Checks the ordinary draft and references; workflow runs have a separate check. "}
        Uses three characters per token with a reply reserve, not a tokenizer. Counts are a lower bound: history, tool results, selection notices, and message wrapping add context. These measurements do not guarantee the request fits. No message is submitted or changed.</p>
    </details>}
  </div>;
}
