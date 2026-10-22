import { useEffect, useState } from "react";
import {
  ArrowLeft,
  ArrowUpRight,
  Brain,
  ChevronRight,
  Download,
  File,
  Folder,
  Search,
  X,
} from "lucide-react";
import { agentPath, allResources, api } from "./api";
import type { Asset, Resource } from "./types";
import Markdown from "./Markdown";
import { documentOutline, documentTrail, workspaceDocuments } from "./workspaceDocuments";

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
  const [childrenLoading, setChildrenLoading] = useState(false);
  const [childrenError, setChildrenError] = useState("");
  const [childrenRefresh, setChildrenRefresh] = useState(0);
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
                  if (!query) setDocuments(matches);
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
    setChildrenError("");
    if (!selected || !("kind" in selected) || (selected.kind !== "space" && selected.kind !== "deck")) {
      setChildrenLoading(false);
      return;
    }
    setChildrenLoading(true);
    allResources(agentKey)
      .then((rows) => {
        if (current) {
          const documents = workspaceDocuments(rows);
          setDocuments(documents);
          setChildren(documents.filter((row) => row.parent_id === selected.id));
        }
      })
      .catch((e) => {
        if (current) setChildrenError(e.message);
      })
      .finally(() => {
        if (current) setChildrenLoading(false);
      });
    return () => {
      current = false;
    };
  }, [selected, agentKey, childrenRefresh]);
  const rows = tab === "files" ? assets : resources;
  const outline = tab === "space" ? documentOutline(resources) : [];
  const trail = selected && "kind" in selected && tab === "space"
    ? documentTrail(selected, documents) : [];
  const parent = trail.length > 1 ? trail[trail.length - 2] : null;
  const rowLabel = tab === "space" ? "document" : "record";
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
                    aria-label={documentTrail(document, documents).map(row => row.title).join(" / ")}
                    onClick={() => setSelected(document)}
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
                  onClick={() => setSelected(item)}
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
                        <button onClick={() => setSelected(document)}>{document.title}</button>
                      )}
                    </span>
                  ))}
                </nav>
              )}
              {parent && (
                <button className="document-parent" onClick={() => setSelected(parent)}>
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
                  <Markdown>
                    {selected.markdown || "No text in this record."}
                  </Markdown>
                  {(selected.kind === "space" || selected.kind === "deck") && (
                    <div className="space-children">
                      {childrenLoading ? (
                        <p className="quiet">Loading documents…</p>
                      ) : childrenError ? (
                        <div role="alert" className="error feedback">
                          {childrenError}
                          <button onClick={() => setChildrenRefresh((n) => n + 1)}>
                            Retry
                          </button>
                        </div>
                      ) : children.length ? (
                        children.map((child) => (
                          <button
                            className="child-row"
                            key={child.id}
                            onClick={() => setSelected(child)}
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
            </>
          )}
        </div>
      </div>
    </section>
  );
}
