import { useEffect, useState } from "react";
import {
  ArrowLeft,
  ArrowUpRight,
  Brain,
  ChevronRight,
  Download,
  File,
  Folder,
  RefreshCw,
  Search,
  X,
} from "lucide-react";
import { agentPath, allResources, api, ApiError } from "./api";
import type { Asset, Resource } from "./types";
import Markdown from "./Markdown";
import { documentOutline, documentTrail, workspaceDocuments } from "./workspaceDocuments";
import MemoryEditor from "./MemoryEditor";

export default function Workspace({ agentKey, browserUrl }: { agentKey: string; browserUrl: string | null }) {
  const [tab, setTab] = useState<"memory" | "space" | "files">("space");
  const [query, setQuery] = useState("");
  const [resources, setResources] = useState<Resource[]>([]);
  const [documents, setDocuments] = useState<Resource[]>([]);
  const [assets, setAssets] = useState<Asset[]>([]);
  const [selected, setSelected] = useState<Resource | Asset | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [refresh, setRefresh] = useState(0);
  const [children, setChildren] = useState<Resource[]>([]);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState("");
  const [detailRefresh, setDetailRefresh] = useState(0);
  const [missingId, setMissingId] = useState<string | null>(null);
  const selectedKind = selected && "kind" in selected ? selected.kind : "file";
  const missing = !!selected && missingId === selected.id;
  useEffect(() => {
    setSelected(null);
  }, [agentKey, tab]);
  useEffect(() => {
    let current = true;
    setLoading(true);
    setError("");
    const timer = setTimeout(() => {
      const request =
        tab === "files"
          ? api<Asset[]>(
              `${agentPath(agentKey)}/assets?q=${encodeURIComponent(query)}`,
            ).then((rows) => {
              if (current) setAssets(rows);
            })
          : tab === "space"
            ? allResources(agentKey, undefined, query).then((rows) => {
                if (current) {
                  const matches = workspaceDocuments(rows);
                  setResources(matches);
                }
              })
            : allResources(agentKey, tab, query).then((rows) => {
                if (current) setResources(rows);
              });
      request
        .catch((e) => {
          if (current) setError(e.message);
        })
        .finally(() => {
          if (current) setLoading(false);
        });
    }, 200);
    return () => {
      current = false;
      clearTimeout(timer);
    };
  }, [agentKey, tab, query, refresh]);
  useEffect(() => {
    let current = true;
    setChildren([]);
    setDetailError("");
    if (!selected) {
      setDetailLoading(false);
      return;
    }
    const selectedId = selected.id;
    setDetailLoading(true);
    async function load() {
      try {
        let saved: Resource | Asset | undefined;
        if (selectedKind === "space" || selectedKind === "deck") {
          const rows = await allResources(agentKey);
          if (!current) return;
          const documents = workspaceDocuments(rows).filter(row => !row.deleted_at);
          setDocuments(documents);
          saved = documents.find(row => row.id === selectedId && row.kind === selectedKind);
          setChildren(documents.filter((row) => row.parent_id === selectedId));
        } else {
          saved = await api<Resource | Asset>(`${agentPath(agentKey)}/${selectedKind === "file" ? "assets" : "resources"}/${encodeURIComponent(selectedId)}`);
          if (!current) return;
          if (("deleted_at" in saved && saved.deleted_at)
            || (selectedKind === "file" ? !("filename" in saved) : !("kind" in saved) || saved.kind !== selectedKind)) saved = undefined;
        }
        setMissingId(saved ? null : selectedId);
        if (saved) setSelected(saved);
      } catch (e) {
        if (!current) return;
        if (e instanceof ApiError && e.status === 404) setMissingId(selectedId);
        else setDetailError((e as Error).message);
      } finally {
        if (current) setDetailLoading(false);
      }
    }
    void load();
    return () => {
      current = false;
    };
  }, [selected?.id, selectedKind, agentKey, detailRefresh, refresh]);
  const rows = tab === "files" ? assets : resources;
  const outline = tab === "space" ? documentOutline(resources) : [];
  const trail = selected && !missing && "kind" in selected && tab === "space"
    ? documentTrail(selected, documents) : [];
  const parent = trail.length > 1 ? trail[trail.length - 2] : null;
  const rowLabel = tab === "space" ? "document" : "record";
  function openRecord(record: Resource | Asset) {
    setSelected(record);
    setDetailRefresh((n) => n + 1);
  }
  return (
    <section
      className={`workspace ${selected ? "has-detail" : ""}`}
      aria-label="Workspace"
    >
      <div className="workspace-toolbar">
        <div className="tabs" role="group" aria-label="Workspace type">
          {(["space", "memory", "files"] as const).map((kind) => (
            <button
              aria-label={kind === "space" ? "Documents" : kind === "memory" ? "Memories" : "Files"}
              aria-pressed={tab === kind}
              className={tab === kind ? "active" : ""}
              key={kind}
              onClick={() => setTab(kind)}
            >
              {kind === "memory" ? (
                <Brain size={16} />
              ) : kind === "space" ? (
                <Folder size={16} />
              ) : (
                <File size={16} />
              )}
              <span>
                {kind === "memory"
                  ? "Memories"
                  : kind === "space"
                    ? "Documents"
                    : "Files"}
              </span>
            </button>
          ))}
        </div>
        <button className="text-link" disabled={loading || detailLoading}
          onClick={() => setRefresh((n) => n + 1)}>
          <RefreshCw size={15} />
          Refresh
        </button>
        {browserUrl && <a
          className="text-link"
          href={browserUrl}
          target="_blank"
          rel="noreferrer"
        >
          Open Tsuchi
          <ArrowUpRight size={15} />
        </a>}
      </div>
      <div className="workspace-content">
        <div className="resource-list">
          <label className="search-field">
            <Search size={16} />
            <span className="sr-only">Search workspace</span>
            <input
              type="search"
              placeholder={`Search ${tab === "memory" ? "memories" : tab === "space" ? "documents" : "files"}…`}
              value={query}
              onChange={(event) => setQuery(event.target.value)}
            />
          </label>
          {error && (
            <div role="alert" className="error feedback">
              {error}
              <button onClick={() => setRefresh((n) => n + 1)}>Retry</button>
            </div>
          )}
          <div className="resource-rows">
            {loading ? (
              <p className="quiet" role="status">
                Searching workspace…
              </p>
            ) : error ? null : rows.length === 0 ? (
              <p className="quiet">
                {query
                  ? "No matching records."
                  : `No ${tab === "memory" ? "memories" : tab === "space" ? "documents" : "files"} yet.`}
              </p>
            ) : tab === "space" ? (
              <nav aria-label={query ? "Document search results" : "Documents"}>
                {outline.map(({ document, depth }) => (
                  <button
                    className={`resource-row document-row ${selected?.id === document.id ? "selected" : ""}`}
                    key={document.id}
                    style={{ paddingInlineStart: `${8 + Math.min(depth, 6) * 16}px` }}
                    aria-current={selected?.id === document.id ? "page" : undefined}
                    aria-label={documentTrail(document, query ? documents : resources).map(row => row.title).join(" / ")}
                    onClick={() => openRecord(document)}
                    title={document.title}
                  >
                    <File size={16} />
                    <span><strong>{document.title}</strong></span>
                  </button>
                ))}
              </nav>
            ) : (
              rows.map((item) => (
                <button
                  className={`resource-row ${selected?.id === item.id ? "selected" : ""}`}
                  key={item.id}
                  onClick={() => openRecord(item)}
                >
                  <span className="resource-icon">
                    {tab === "files" ? (
                      <File size={18} />
                    ) : (
                      <Brain size={18} />
                    )}
                  </span>
                  <span>
                    <strong>{item.title}</strong>
                    <span className="quiet">
                      {"filename" in item
                        ? item.filename
                        : new Date(item.updated_at).toLocaleDateString(
                            undefined,
                            { month: "short", day: "numeric" },
                          )}
                    </span>
                  </span>
                </button>
              ))
            )}
          </div>
          <div className="list-count">
            {!loading && !error &&
              `${rows.length} ${rowLabel}${rows.length === 1 ? "" : "s"}`}
          </div>
        </div>
        <div className="resource-detail">
          {!selected ? (
            <div className="detail-empty">
              <Folder size={24} />
              <p>{tab === "space" ? "Select a document" : "Select a record"}</p>
            </div>
          ) : (
            <>
              {trail.length > 0 && (
                <nav className="document-breadcrumbs" aria-label="Document path">
                  <button onClick={() => setSelected(null)}>Documents</button>
                  {trail.map((document, index) => (
                    <span key={document.id}>
                      <ChevronRight size={14} aria-hidden="true" />
                      {index === trail.length - 1 ? (
                        <span aria-current="page">{document.title}</span>
                      ) : (
                        <button onClick={() => openRecord(document)}>{document.title}</button>
                      )}
                    </span>
                  ))}
                </nav>
              )}
              {parent && (
                <button className="document-parent" onClick={() => openRecord(parent)}>
                  <ArrowLeft size={16} />
                  <span>Back to {parent.title}</span>
                </button>
              )}
              <div className="detail-header">
                <button
                  className="icon mobile-only"
                  aria-label="Back to records"
                  onClick={() => setSelected(null)}
                >
                  <ArrowLeft size={18} />
                </button>
                <h2>{selected.title}</h2>
                <button
                  className="icon desktop-only"
                  aria-label="Close record"
                  onClick={() => setSelected(null)}
                >
                  <X size={18} />
                </button>
              </div>
              {detailLoading && <p className="quiet" role="status">Refreshing record…</p>}
              {detailError && <div role="alert" className="error feedback">
                {detailError} {missing ? "The record’s availability could not be refreshed." : "Showing the last loaded record."}
                <button onClick={() => setDetailRefresh((n) => n + 1)}>Retry</button>
              </div>}
              {missing ? <p className="quiet" role="status">This record is no longer available in this view.</p> : <>
              {"filename" in selected ? (
                <>
                  <dl className="metadata">
                    <dt>Filename</dt>
                    <dd>{selected.filename}</dd>
                    <dt>Type</dt>
                    <dd>{selected.media_type}</dd>
                    <dt>Size</dt>
                    <dd>
                      {new Intl.NumberFormat().format(selected.size_bytes)}{" "}
                      bytes
                    </dd>
                  </dl>
                  <a
                    className="button"
                    href={`/api${agentPath(agentKey)}/assets/${selected.id}/content?download=true`}
                  >
                    <Download size={16} />
                    Download file
                  </a>
                </>
              ) : (
                <>
                  {selected.kind === "memory" ? <MemoryEditor key={`${agentKey}:${selected.id}`} agentKey={agentKey} memory={selected}
                    onSaved={value => {
                      setSelected(value);
                      setResources(rows => rows.map(row => row.id === value.id ? value : row));
                    }} /> : <Markdown>
                    {selected.markdown || "No text in this record."}
                  </Markdown>}
                  {(selected.kind === "space" || selected.kind === "deck") && (
                    <div className="space-children">
                      {detailLoading || detailError ? null : children.length ? (
                        children.map((child) => (
                          <button
                            className="child-row"
                            key={child.id}
                            onClick={() => openRecord(child)}
                          >
                            {child.kind === "space" ? (
                              <Folder size={16} />
                            ) : (
                              <File size={16} />
                            )}
                            <span>{child.title}</span>
                            <ChevronRight size={14} aria-hidden="true" />
                          </button>
                        ))
                      ) : (
                        <p className="quiet">No subdocuments.</p>
                      )}
                    </div>
                  )}
                  {selected.kind === "conversation" &&
                    selected.messages.map((message) => (
                      <article className="message" key={message.id}>
                        <strong>{message.role}</strong>
                        <Markdown>{message.content}</Markdown>
                      </article>
                    ))}
                  {selected.kind === "deck" && (
                    <p className="field-hint">Slide decks open in Tsuchi.</p>
                  )}
                </>
              )}
              {selected.tags.length > 0 && (
                <div className="tags">
                  {selected.tags.map((tag) => (
                    <span key={tag}>{tag}</span>
                  ))}
                </div>
              )}
              </>}
            </>
          )}
        </div>
      </div>
    </section>
  );
}
