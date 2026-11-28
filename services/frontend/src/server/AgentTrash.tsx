import { useEffect, useState } from "react";
import type { Resource } from "../kadan/types";
import { formatBytes, formatWhen, getJson, sendJson, storagePath, useAction, type Agent } from "./api";

interface StoredFile {
  id: string;
  title: string;
  filename: string;
  size_bytes: number;
  is_image: boolean;
  version: number;
  deleted_at: string | null;
}

type Trashed =
  | { type: "resource"; id: string; title: string; label: string; when: string; version: number }
  | { type: "file"; id: string; title: string; label: string; when: string; version: number };

const KIND_LABEL: Record<Resource["kind"], string> = {
  memory: "Memory",
  persona: "Persona",
  space: "Page",
  deck: "Slides",
  conversation: "Chat",
  folder: "Folder",
};

async function loadTrash(key: string, signal: AbortSignal): Promise<Trashed[]> {
  const resources: Resource[] = [];
  for (let offset = 0; ; offset += 200) {
    const page = await getJson<Resource[]>(`${storagePath(key)}/resources?trash=true&limit=200&offset=${offset}`, signal);
    resources.push(...page);
    if (page.length < 200) break;
  }
  const files = await getJson<StoredFile[]>(`${storagePath(key)}/assets?trash=true`, signal);
  return [
    ...resources.map((item): Trashed => ({
      type: "resource", id: item.id, title: item.title, label: KIND_LABEL[item.kind] ?? "Item",
      when: item.deleted_at ?? item.updated_at, version: item.version,
    })),
    ...files.map((file): Trashed => ({
      type: "file", id: file.id, title: file.title || file.filename,
      label: `${file.is_image ? "Image" : "File"} · ${formatBytes(file.size_bytes)}`,
      when: file.deleted_at ?? "", version: file.version,
    })),
  ].sort((a, b) => b.when.localeCompare(a.when));
}

export default function AgentTrash({ agent }: { agent: Agent }) {
  const [items, setItems] = useState<Trashed[] | null>(null);
  const [loadError, setLoadError] = useState("");
  const [notice, setNotice] = useState("");
  const [refresh, setRefresh] = useState(0);
  const action = useAction();

  useEffect(() => {
    const controller = new AbortController();
    loadTrash(agent.key, controller.signal)
      .then((rows) => { setItems(rows); setLoadError(""); })
      .catch((failure) => !controller.signal.aborted && setLoadError((failure as Error).message));
    return () => controller.abort();
  }, [agent.key, refresh]);

  return (
    <div className="srv-stack">
      <p className="field-hint">
        Things you or {agent.name} removed stay here, with their files, until you restore them. {agent.name} can’t see
        items in trash.
      </p>
      {loadError && <p className="error" role="alert">{loadError}</p>}
      {action.error && <p className="error" role="alert">{action.error}</p>}
      {notice && <div className="banner srv-banner" role="status"><span>{notice}</span><button onClick={() => setNotice("")}>Dismiss</button></div>}
      {!items && !loadError && <p className="srv-loading">Loading trash…</p>}
      {items && items.length === 0 && <p className="srv-empty">Trash is empty.</p>}
      {items && items.length > 0 && (
        <ul className="srv-list">
          {items.map((item) => (
            <li key={`${item.type}:${item.id}`} className="srv-trash-row">
              <div>
                <strong>{item.title}</strong>
                <span>{item.label}{item.when ? ` · removed ${formatWhen(item.when)}` : ""}</span>
              </div>
              <button className="button" disabled={action.busy} onClick={() => void action.run(async () => {
                const base = storagePath(agent.key);
                await sendJson(
                  item.type === "file" ? `${base}/assets/${item.id}/restore` : `${base}/resources/${item.id}/restore`,
                  "POST",
                  { expected_version: item.version },
                );
                setNotice(`Restored “${item.title}”.`);
                setRefresh((value) => value + 1);
              })}>Restore</button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
