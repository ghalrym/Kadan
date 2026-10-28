import { useEffect, useRef, useState } from "react";
import { agentPath, api, ApiError } from "./api";
import type { Agent, Runner } from "./types";

interface Props {
  agent: Agent;
  onClose: () => void;
}

function available(runner: Runner, now: number) {
  return new Date(runner.lease_expires_at).getTime() > now;
}

export default function RunnerAssignment({ agent, onClose }: Props) {
  const dialog = useRef<HTMLDialogElement>(null);
  const active = useRef(true);
  const pending = useRef<"read" | "write" | null>(null);
  const [runners, setRunners] = useState<Runner[] | null>(null);
  const [target, setTarget] = useState("");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [needsRefresh, setNeedsRefresh] = useState(true);
  const [checkedAt, setCheckedAt] = useState(0);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const owner = runners?.find((runner) => runner.agent_keys.includes(agent.key));
  const destination = runners?.find((runner) => runner.runner_id === target);
  const busy = loading || saving;
  const canAssign = !!destination && destination !== owner && available(destination, Date.now())
    && destination.agent_keys.length < destination.capacity;

  useEffect(() => {
    active.current = true;
    dialog.current?.showModal();
    void refresh();
    return () => { active.current = false; };
  }, []);

  function close() {
    if (pending.current === "write") return;
    active.current = false;
    onClose();
  }

  async function readRunners() {
    const rows = await api<Runner[]>("/runners");
    if (!active.current) return;
    setRunners(rows);
    setCheckedAt(Date.now());
    setNeedsRefresh(false);
  }

  async function refresh() {
    if (pending.current) return;
    pending.current = "read";
    setLoading(true);
    setError("");
    setNotice("");
    try {
      await readRunners();
    } catch (e) {
      if (active.current) {
        setNeedsRefresh(true);
        setError(`Could not refresh assignments: ${(e as Error).message}`);
      }
    } finally {
      pending.current = null;
      if (active.current) setLoading(false);
    }
  }

  async function assign(runnerId: string | null) {
    if (pending.current || needsRefresh || !runners || runnerId === "" || runnerId === (owner?.runner_id ?? null)) return;
    if (runnerId !== null && (!destination || destination.runner_id !== runnerId
      || !available(destination, Date.now()) || destination.agent_keys.length >= destination.capacity)) {
      setNeedsRefresh(true);
      setError("Choose a runner with a current heartbeat and a free slot. Refresh assignments before trying again.");
      return;
    }
    pending.current = "write";
    setSaving(true);
    setNeedsRefresh(true);
    setError("");
    setNotice("");
    let saved = false;
    try {
      await api<void>(`${agentPath(agent.key)}/runner`, {
        runner_id: runnerId, expected_runner_id: owner?.runner_id ?? null,
      });
      if (!active.current) return;
      saved = true;
      setNotice(runnerId === null ? "Assignment released. An available runner may assign this agent again." : "Assignment saved.");
      await readRunners();
    } catch (e) {
      if (active.current) {
        const message = (e as Error).message;
        setError(saved ? `Could not refresh the saved assignment: ${message} Refresh assignments before trying again.`
          : e instanceof ApiError && e.status === 409 ? `${message} Refresh assignments before trying again.`
            : `Assignment change unconfirmed; it may have been saved. ${message} Refresh assignments before trying again.`);
      }
    } finally {
      pending.current = null;
      if (active.current) setSaving(false);
    }
  }

  return (
    <dialog ref={dialog} className="runner-assignment" aria-labelledby="runner-assignment-title"
      onCancel={(event) => { event.preventDefault(); close(); }}>
      <form onSubmit={(event) => { event.preventDefault(); void assign(target); }}>
        <h2 id="runner-assignment-title">Runner assignment</h2>
        <dl>
          <dt>Agent</dt><dd>{agent.name}</dd>
          <dt>{needsRefresh && runners ? "Last known assignment" : "Assigned runner"}</dt>
          <dd>{runners === null ? (loading ? "Loading…" : "Unavailable") : owner ? owner.runner_id : "Unassigned"}</dd>
          {owner && <>
            <dt>Heartbeat at last refresh</dt><dd>{available(owner, checkedAt) ? "Active" : "Expired"}</dd>
            <dt>Assigned agents</dt><dd>{owner.agent_keys.length} / {owner.capacity}</dd>
          </>}
        </dl>
        {checkedAt > 0 && <p className="field-hint">Checked {new Date(checkedAt).toLocaleTimeString()}. Refresh for current availability.</p>}
        {runners?.length === 0 && <p>No runners registered.</p>}
        {!!runners?.length && <label>
          Assign to
          <select value={target} disabled={busy || needsRefresh} onChange={(event) => setTarget(event.target.value)}>
            <option value="">Choose a runner</option>
            {runners.map((runner) => {
              const online = available(runner, checkedAt);
              const full = runner.agent_keys.length >= runner.capacity;
              return <option key={runner.runner_id} value={runner.runner_id}
                disabled={runner === owner || !online || full}>
                {runner.runner_id} · {runner.agent_keys.length}/{runner.capacity}{runner === owner ? " · current" : ""}{!online ? " · heartbeat expired" : full ? " · full" : ""}
              </option>;
            })}
          </select>
        </label>}
        {owner && <>
          <button type="button" disabled={busy || needsRefresh} onClick={() => void assign(null)}>Release assignment</button>
          <p className="field-hint">Release allows automatic assignment again. To stop a response, use Stop in chat.</p>
        </>}
        {notice && <p role="status">{notice}</p>}
        {error && <p className="error" role="alert">{error}</p>}
        <button type="button" disabled={busy} onClick={() => void refresh()}>{loading ? "Refreshing…" : "Refresh assignments"}</button>
        <div className="form-actions">
          <button type="button" disabled={saving} onClick={close}>Close</button>
          <button className="primary" disabled={busy || needsRefresh || !canAssign}>{saving ? "Waiting…" : "Assign"}</button>
        </div>
      </form>
    </dialog>
  );
}
