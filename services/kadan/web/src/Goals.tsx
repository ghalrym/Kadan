import { useEffect, useRef, useState } from "react";
import { api, agentPath } from "./api";

export interface AssistantTask {
  id: string;
  agent_key: string;
  conversation_id: string;
  origin_run_id: string;
  current_run_id: string | null;
  status: "active" | "waiting" | "paused" | "completed" | "failed" | "cancelled";
  goal: string;
  progress: string;
  next_step: string;
  blocker: string | null;
  wait_kind: "user" | "tool" | "timer" | null;
  wake_at: string | null;
  version: number;
  created_at: string;
  updated_at: string;
}

type Control = "pause" | "resume" | "cancel";
interface Props { agentKey: string; conversationId: string }

export default function Goals({ agentKey, conversationId }: Props) {
  const [tasks, setTasks] = useState<AssistantTask[]>([]);
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState("");
  const [actionError, setActionError] = useState("");
  const [pending, setPending] = useState("");
  const [refresh, setRefresh] = useState(0);
  const request = useRef({ closed: false, revision: 0, pending: false });
  useEffect(() => {
    const state = { closed: false, revision: 0, pending: false };
    request.current = state;
    let timer: ReturnType<typeof setTimeout>;
    async function poll() {
      const revision = state.revision;
      try {
        const values = await api<AssistantTask[]>(`${agentPath(agentKey)}/conversations/${conversationId}/tasks`);
        if (!state.closed && revision === state.revision) {
          if (values.some(task => task.agent_key !== agentKey || task.conversation_id !== conversationId)) {
            throw new Error("Couldn’t read goals for this conversation.");
          }
          setTasks(values);
          setLoaded(true);
          setError("");
        }
      } catch (failure) {
        if (!state.closed && revision === state.revision) setError((failure as Error).message);
      } finally {
        if (!state.closed) timer = setTimeout(poll, 2000);
      }
    }
    void poll();
    return () => { state.closed = true; clearTimeout(timer); };
  }, [agentKey, conversationId, refresh]);

  async function control(task: AssistantTask, action: Control) {
    const state = request.current;
    if (state.pending || error || !loaded) return;
    state.pending = true;
    state.revision += 1;
    setPending(task.id);
    setActionError("");
    try {
      const updated = await api<AssistantTask>(`${agentPath(agentKey)}/tasks/${task.id}/${action}`, { expected_version: task.version });
      if (updated.id !== task.id || updated.agent_key !== agentKey || updated.conversation_id !== conversationId) {
        throw new Error("Couldn’t confirm the goal update.");
      }
      if (!state.closed) setTasks(values => values.map(item => item.id === updated.id ? updated : item));
    } catch (failure) {
      if (!state.closed) {
        setActionError(`${(failure as Error).message} Checking the saved state before another change.`);
        setError("Goal update needs confirmation.");
      }
    } finally {
      state.pending = false;
      if (!state.closed) {
        setPending("");
        setRefresh(value => value + 1);
      }
    }
  }

  return <GoalList tasks={tasks} loaded={loaded} error={error} actionError={actionError}
    pending={pending} onControl={(task, action) => void control(task, action)}
    onRefresh={() => setRefresh(value => value + 1)} />;
}

interface ListProps {
  tasks: AssistantTask[];
  loaded: boolean;
  error: string;
  actionError: string;
  pending: string;
  onControl: (task: AssistantTask, action: Control) => void;
  onRefresh: () => void;
}

export function GoalList({ tasks, loaded, error, actionError, pending, onControl, onRefresh }: ListProps) {
  const active = tasks.filter(task => ["active", "waiting", "paused"].includes(task.status));
  const finished = tasks.filter(task => !active.includes(task));
  const rows = (items: AssistantTask[]) => <ul className="goal-list">{items.map(task => <li key={task.id}>
    <strong>{task.goal}</strong>
    <span className="activity-state">{taskState(task)}</span>
    {task.progress && <p>{task.progress}</p>}
    {task.next_step && <p><b>Next:</b> {task.next_step}</p>}
    {task.blocker && <p className="goal-blocker"><b>Blocker:</b> {task.blocker}</p>}
    {task.wait_kind === "timer" && task.wake_at && <p>Check again {new Date(task.wake_at).toLocaleString()}.</p>}
    {task.status === "waiting" && task.wait_kind === "user" && <p>Reply in chat to continue.</p>}
    {["active", "waiting", "paused", "failed"].includes(task.status) && <div className="goal-controls">
      <button disabled={!!pending || !!error} onClick={() => onControl(task, ["paused", "failed"].includes(task.status) ? "resume" : "pause")}>
        {pending === task.id ? "Saving…" : ["paused", "failed"].includes(task.status) ? "Resume" : "Pause"}
      </button>
      {task.status !== "failed" && <button disabled={!!pending || !!error} onClick={() => onControl(task, "cancel")}>Cancel goal</button>}
    </div>}
  </li>)}</ul>;
  return <section className="agent-card-goals" aria-labelledby="agent-card-goals">
    <header><h3 id="agent-card-goals">Goals</h3></header>
    {error && <p className="activity-error">{loaded ? "Couldn’t refresh goals. Showing the last update." : "Couldn’t load goals."} <button disabled={!!pending} onClick={onRefresh}>Refresh</button></p>}
    {actionError && <p role="alert" className="activity-error">{actionError}</p>}
    {!loaded ? !error && <p className="activity-empty">Loading goals…</p> : tasks.length === 0 ? <p className="activity-empty">No saved goals yet.</p> : <>
      <div className="goals-content" role="region" aria-label="Saved goals" tabIndex={0}>
        {rows(active)}
        {finished.length > 0 && <details><summary>Finished and failed ({finished.length})</summary>{rows(finished)}</details>}
      </div>
      <p className="goal-note">Pause or cancel stops future goal work. Actions already started may continue; check Computers or Coding.</p>
    </>}
  </section>;
}

function taskState(task: AssistantTask) {
  if (task.status === "waiting") {
    return task.wait_kind === "user" ? "Waiting for you" : task.wait_kind === "tool" ? "Waiting for work to finish" : "Waiting until the next check";
  }
  return { active: "Active", paused: "Paused", completed: "Finished", failed: "Failed", cancelled: "Cancelled" }[task.status];
}
