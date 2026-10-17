import { ChevronDown, Wrench } from "lucide-react";
import type { ToolExecution } from "./types";
import { toolStatus } from "./toolActivity";

export default function ToolActivity({
  execution,
  active,
}: {
  execution: ToolExecution | null;
  active: boolean;
}) {
  if (!execution)
    return <p className="tool-unavailable">Tool activity could not be read.</p>;
  const unknown = execution.status === "started" && !active;
  return (
    <details className={`tool-activity ${execution.status}`}>
      <summary>
        <Wrench size={14} aria-hidden="true" />
        <strong>{execution.name}</strong>
        <span role="status">{toolStatus(execution, active)}</span>
        <ChevronDown size={14} className="tool-chevron" aria-hidden="true" />
      </summary>
      {unknown && (
        <p className="field-hint">
          No result was saved. The tool may have made changes; check before
          retrying.
        </p>
      )}
      <dl>
        <dt>Arguments</dt>
        <dd>
          <pre>{JSON.stringify(execution.arguments, null, 2)}</pre>
        </dd>
        {execution.result != null && (
          <>
            <dt>Result</dt>
            <dd>
              <pre>{JSON.stringify(execution.result, null, 2)}</pre>
            </dd>
          </>
        )}
      </dl>
    </details>
  );
}
