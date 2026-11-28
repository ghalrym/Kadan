import { useEffect, useState } from "react";
import Markdown from "../kadan/Markdown";
import type { Resource } from "../kadan/types";
import { formatWhen, getJson, kadanPath, sendJson, storagePath, useAction, type Agent } from "./api";

interface Version {
  version: number;
  created_at: string;
  snapshot: Resource;
}

export default function AgentPersona({ agent }: { agent: Agent }) {
  const [persona, setPersona] = useState<Resource | null | undefined>(undefined);
  const [markdown, setMarkdown] = useState("");
  const [versions, setVersions] = useState<Version[]>([]);
  const [viewing, setViewing] = useState<Version | null>(null);
  const [loadError, setLoadError] = useState("");
  const [notice, setNotice] = useState("");
  const [refresh, setRefresh] = useState(0);
  const action = useAction();

  useEffect(() => {
    const controller = new AbortController();
    getJson<Resource | null>(`${kadanPath(agent.key)}/persona`, controller.signal)
      .then(async (value) => {
        setPersona(value);
        setMarkdown(value?.markdown ?? "");
        setLoadError("");
        if (value) {
          const rows = await getJson<Version[]>(`${storagePath(agent.key)}/resources/${value.id}/versions`, controller.signal);
          setVersions([...rows].sort((a, b) => b.version - a.version));
        }
      })
      .catch((failure) => !controller.signal.aborted && setLoadError((failure as Error).message));
    return () => controller.abort();
  }, [agent.key, refresh]);

  const changed = markdown !== (persona?.markdown ?? "");

  if (persona === undefined && !loadError) return <p className="srv-loading">Loading persona…</p>;

  return (
    <div className="srv-stack">
      <p className="field-hint">
        The persona describes who {agent.name} is and how {agent.name} should talk and work with you. {agent.name} can
        also update it while you chat.
      </p>
      {loadError && <p className="error" role="alert">{loadError}</p>}
      {notice && <div className="banner srv-banner" role="status"><span>{notice}</span><button onClick={() => setNotice("")}>Dismiss</button></div>}
      <section className="srv-card srv-editor">
        <label>
          Persona
          <textarea rows={14} value={markdown} disabled={action.busy} onChange={(event) => setMarkdown(event.target.value)} />
        </label>
        {persona && <p className="field-hint">Last saved {formatWhen(persona.updated_at)}. Changes apply to {agent.name}’s next reply.</p>}
        {action.error && <p className="error" role="alert">{action.error} Your text is kept here; reload to see the saved persona.</p>}
        <div className="form-actions srv-editor-actions">
          {action.error && <button disabled={action.busy} onClick={() => { action.setError(""); setRefresh((value) => value + 1); }}>Reload</button>}
          <span className="srv-spacer" />
          <button disabled={action.busy || !changed} onClick={() => setMarkdown(persona?.markdown ?? "")}>Undo changes</button>
          <button className="primary" disabled={action.busy || !changed || !markdown.trim()} onClick={() => void action.run(async () => {
            const saved = await sendJson<Resource>(`${kadanPath(agent.key)}/persona`, "PUT", {
              markdown, expected_version: persona?.version ?? null,
            });
            setNotice("Persona saved.");
            setRefresh((value) => value + 1);
            return saved;
          })}>
            {action.busy ? "Saving…" : "Save persona"}
          </button>
        </div>
      </section>

      {versions.length > 1 && (
        <section className="srv-card">
          <header className="srv-card-head">
            <h2>Earlier versions</h2>
            <p>Every save is kept. Restoring an earlier version saves it as the newest one.</p>
          </header>
          <ul className="srv-versions">
            {versions.map((version, index) => (
              <li key={version.version}>
                <div className="srv-version-row">
                  <span>
                    <strong>{index === 0 ? "Current version" : `Version ${version.version}`}</strong>
                    <small>{formatWhen(version.created_at)}</small>
                  </span>
                  <button className="button" onClick={() => setViewing(viewing?.version === version.version ? null : version)}>
                    {viewing?.version === version.version ? "Hide" : "View"}
                  </button>
                  {index > 0 && persona && (
                    <button className="button" disabled={action.busy} onClick={() => void action.run(async () => {
                      await sendJson(`${storagePath(agent.key)}/resources/${persona.id}/versions/${version.version}/restore`, "POST", {
                        expected_version: persona.version,
                      });
                      setViewing(null);
                      setNotice(`Restored version ${version.version}.`);
                      setRefresh((value) => value + 1);
                    })}>Restore</button>
                  )}
                </div>
                {viewing?.version === version.version && (
                  <div className="srv-version-preview"><Markdown>{version.snapshot.markdown}</Markdown></div>
                )}
              </li>
            ))}
          </ul>
        </section>
      )}
    </div>
  );
}
