import { useEffect, useState } from "react";
import { Search } from "lucide-react";
import type { Resource } from "../kadan/types";
import { formatWhen, getJson, kadanPath, sendJson, storagePath, useAction, type Agent } from "./api";

function MemoryEditor({ agent, memory, onSaved, onForgotten, onClose }: {
  agent: Agent;
  memory: Resource;
  onSaved: (memory: Resource) => void;
  onForgotten: () => void;
  onClose: () => void;
}) {
  const [title, setTitle] = useState(memory.title);
  const [markdown, setMarkdown] = useState(memory.markdown);
  const [base, setBase] = useState(memory);
  const [confirmForget, setConfirmForget] = useState(false);
  const [stale, setStale] = useState(false);
  const action = useAction();
  const changed = title.trim() !== base.title || markdown !== base.markdown;

  async function reload() {
    const fresh = await action.run(() => getJson<Resource>(`${kadanPath(agent.key)}/memories/${memory.id}`));
    if (!fresh) return;
    setBase(fresh);
    setStale(false);
  }

  return (
    <section className="srv-card srv-editor" aria-label={`Memory: ${memory.title}`}>
      <label>
        Title
        <input value={title} maxLength={500} disabled={action.busy} onChange={(event) => setTitle(event.target.value)} />
      </label>
      <label>
        What {agent.name} remembers
        <textarea rows={10} value={markdown} disabled={action.busy} onChange={(event) => setMarkdown(event.target.value)} />
      </label>
      <p className="field-hint">Changes take effect on {agent.name}’s next reply. Last saved {formatWhen(base.updated_at)}.</p>
      {action.error && (
        <p className="error" role="alert">
          {action.error}
          {stale && " Your text is kept here."}
        </p>
      )}
      {stale && <button disabled={action.busy} onClick={() => void reload()}>Load the saved version</button>}
      <div className="form-actions srv-editor-actions">
        {confirmForget ? (
          <>
            <span className="field-hint">Move this memory to trash? You can restore it later.</span>
            <button disabled={action.busy} onClick={() => setConfirmForget(false)}>Keep it</button>
            <button className="danger-button" disabled={action.busy} onClick={() => void action.run(async () => {
              await sendJson(`${storagePath(agent.key)}/resources/${memory.id}?expected_version=${base.version}`, "DELETE");
              onForgotten();
            })}>Forget</button>
          </>
        ) : (
          <>
            <button className="danger-button" disabled={action.busy} onClick={() => setConfirmForget(true)}>Forget…</button>
            <span className="srv-spacer" />
            <button disabled={action.busy} onClick={onClose}>Close</button>
            <button className="primary" disabled={action.busy || stale || !changed || !title.trim()} onClick={() => void action.run(async () => {
              try {
                const saved = await sendJson<Resource>(`${kadanPath(agent.key)}/memories/${memory.id}`, "PUT", {
                  title: title.trim(), markdown, expected_version: base.version,
                });
                setBase(saved);
                onSaved(saved);
              } catch (error) {
                setStale(true);
                throw error;
              }
            })}>
              {action.busy ? "Saving…" : "Save"}
            </button>
          </>
        )}
      </div>
    </section>
  );
}

export default function AgentMemories({ agent }: { agent: Agent }) {
  const [query, setQuery] = useState("");
  const [memories, setMemories] = useState<Resource[] | null>(null);
  const [error, setError] = useState("");
  const [open, setOpen] = useState<Resource | null>(null);
  const [notice, setNotice] = useState("");
  const [refresh, setRefresh] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    const timer = setTimeout(() => {
      getJson<Resource[]>(`${kadanPath(agent.key)}/memories?limit=200&q=${encodeURIComponent(query.trim())}`, controller.signal)
        .then((rows) => { setMemories(rows); setError(""); })
        .catch((failure) => !controller.signal.aborted && setError((failure as Error).message));
    }, 200);
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, [agent.key, query, refresh]);

  return (
    <div className="srv-stack">
      <p className="field-hint">
        {agent.name} decides what to remember while you chat. Here you can read those memories, fix anything that’s wrong
        or have {agent.name} forget something.
      </p>
      <label className="srv-search">
        <Search size={15} aria-hidden="true" />
        <input type="search" placeholder="Search memories" aria-label="Search memories" value={query}
          onChange={(event) => setQuery(event.target.value)} />
      </label>
      {notice && <div className="banner srv-banner" role="status"><span>{notice}</span><button onClick={() => setNotice("")}>Dismiss</button></div>}
      {error && <p className="error" role="alert">{error}</p>}
      {open && (
        <MemoryEditor key={open.id} agent={agent} memory={open}
          onSaved={(saved) => {
            setNotice(`Saved “${saved.title}”.`);
            setMemories((rows) => rows?.map((row) => (row.id === saved.id ? saved : row)) ?? rows);
          }}
          onForgotten={() => {
            setNotice(`${agent.name} forgot “${open.title}”. It’s in Trash if you change your mind.`);
            setOpen(null);
            setRefresh((value) => value + 1);
          }}
          onClose={() => setOpen(null)} />
      )}
      {!memories && !error && <p className="srv-loading">Loading memories…</p>}
      {memories && memories.length === 0 && (
        <p className="srv-empty">{query ? "No memories match that search." : `${agent.name} hasn’t saved any memories yet.`}</p>
      )}
      {memories && memories.length > 0 && (
        <ul className="srv-list">
          {memories.map((memory) => (
            <li key={memory.id}>
              <button className={open?.id === memory.id ? "selected" : ""} onClick={() => setOpen(memory)}>
                <strong>{memory.title}</strong>
                <span>{memory.markdown.replace(/\s+/g, " ").slice(0, 160)}</span>
                <small>{formatWhen(memory.updated_at)}</small>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
