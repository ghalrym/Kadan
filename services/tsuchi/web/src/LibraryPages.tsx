import { useEffect, useState } from "react";
import {
  ArrowLeft,
  ArrowRight,
  ChevronRight,
  File,
  FileText,
  Folder,
  Plus,
  Search,
  Upload,
} from "lucide-react";
import {
  agentPath,
  allResources,
  assetUrl,
  formatBytes,
  formatDate,
  request,
  useLoad,
} from "./api";
import { Empty, ErrorNotice, Loading } from "./components";
import { kindLabels } from "./types";
import type { Asset, Kind, Resource } from "./types";

function ResourceRows({
  resources,
  onOpen,
  showKind = false,
}: {
  resources: Resource[];
  onOpen: (resource: Resource) => void;
  showKind?: boolean;
}) {
  return (
    <div className="resource-list">
      {resources.map((resource) => (
        <button
          className="resource-row"
          key={resource.id}
          onClick={() => onOpen(resource)}
        >
          <FileText size={20} />
          <span className="row-content">
            <span className="row-title">{resource.title}</span>
            <span className="row-metadata">
              {showKind && <span>{kindLabels[resource.kind]}</span>}
              <span>{formatDate(resource.updated_at)}</span>
              {resource.tags.map((tag) => (
                <span className="tag" key={tag}>
                  {tag}
                </span>
              ))}
            </span>
          </span>
          <span className="version">v{resource.version}</span>
          <ChevronRight size={18} />
        </button>
      ))}
    </div>
  );
}
function SpaceTree({
  resources,
  onOpen,
}: {
  resources: Resource[];
  onOpen: (resource: Resource) => void;
}) {
  const spaces = new Set(
    resources
      .filter((resource) => resource.kind === "space")
      .map((resource) => resource.id),
  );
  const children = new Map<string | null, Resource[]>();
  for (const resource of resources) {
    if (resource.kind !== "space" && !resource.parent_id) continue;
    const parent =
      resource.parent_id && spaces.has(resource.parent_id)
        ? resource.parent_id
        : null;
    children.set(parent, [...(children.get(parent) || []), resource]);
  }
  const branch = (
    parent: string | null,
    visited = new Set<string>(),
  ): React.ReactNode => (
    <ul className="space-tree">
      {(children.get(parent) || [])
        .filter((resource) => !visited.has(resource.id))
        .map((resource) => (
          <li key={resource.id}>
            <button className="resource-row" onClick={() => onOpen(resource)}>
              {resource.kind === "space" ? (
                <Folder size={20} />
              ) : (
                <FileText size={20} />
              )}
              <span className="row-content">
                <span className="row-title">{resource.title}</span>
                <span className="row-metadata">
                  {resource.kind === "space"
                    ? `${children.get(resource.id)?.length || 0} children`
                    : kindLabels[resource.kind]}
                </span>
              </span>
              <ChevronRight size={18} />
            </button>
            {children.has(resource.id) &&
              branch(resource.id, new Set([...visited, resource.id]))}
          </li>
        ))}
    </ul>
  );
  return branch(null);
}
export function ResourcesPage({
  agentKey,
  kind,
  search,
  refresh,
  onOpen,
  onNew,
}: {
  agentKey: string;
  kind?: Kind;
  search?: string;
  refresh: number;
  onOpen: (resource: Resource) => void;
  onNew: (kind: Kind) => void;
}) {
  const [query, setQuery] = useState("");
  const [offset, setOffset] = useState(0);
  const [filter, setFilter] = useState("");
  useEffect(() => {
    const timer = window.setTimeout(() => {
      setFilter(query);
      setOffset(0);
    }, 250);
    return () => window.clearTimeout(timer);
  }, [query]);
  const q = search ?? filter;
  const tree = kind === "space" && !q;
  const { data, loading, error } = useLoad(
    () =>
      tree
        ? allResources(agentKey)
        : request<Resource[]>(
            `${agentPath(agentKey)}/resources?limit=100&offset=${offset}${kind ? `&kind=${kind}` : ""}&q=${encodeURIComponent(q)}`,
          ),
    [agentKey, kind, q, offset, refresh, tree],
  );
  const isEmpty = tree
    ? !data?.some((resource) => resource.kind === "space")
    : data?.length === 0;
  return (
    <>
      <div className="page-heading">
        <h1>{kind ? kindLabels[kind] : "Search"}</h1>
        {kind && (
          <button className="primary" onClick={() => onNew(kind)}>
            <Plus size={18} />
            New {kind === "space" ? "Space" : kind === "deck" ? "deck" : kind}
          </button>
        )}
      </div>
      {search === undefined ? (
        <label className="search-field">
          <Search size={18} />
          <input
            aria-label={`Search ${kind ? kindLabels[kind].toLowerCase() : "resources"}`}
            placeholder="Search"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
          />
        </label>
      ) : (
        <div className="search-summary">Results for “{search}”</div>
      )}
      <ErrorNotice message={error} />
      {loading ? (
        <Loading />
      ) : isEmpty ? (
        <Empty
          action={
            kind ? (
              <button onClick={() => onNew(kind)}>
                <Plus size={16} />
                Create{" "}
                {kind === "space" ? "Space" : kind === "deck" ? "deck" : kind}
              </button>
            ) : undefined
          }
        >
          {q
            ? "No matching resources."
            : `No ${kind ? kindLabels[kind].toLowerCase() : "resources"} yet.`}
        </Empty>
      ) : (
        data &&
        (tree ? (
          <SpaceTree resources={data} onOpen={onOpen} />
        ) : (
          <ResourceRows resources={data} onOpen={onOpen} showKind={!kind} />
        ))
      )}
      {!tree && !loading && (offset > 0 || data?.length === 100) && (
        <div className="pagination">
          <button
            disabled={offset === 0}
            onClick={() => setOffset((value) => Math.max(0, value - 100))}
          >
            <ArrowLeft size={16} />
            Previous
          </button>
          <span>Page {offset / 100 + 1}</span>
          <button
            disabled={!data || data.length < 100}
            onClick={() => setOffset((value) => value + 100)}
          >
            Next
            <ArrowRight size={16} />
          </button>
        </div>
      )}
    </>
  );
}
export function AssetsPage({
  agentKey,
  images,
  refresh,
  onOpen,
  onNew,
}: {
  agentKey: string;
  images: boolean;
  refresh: number;
  onOpen: (asset: Asset) => void;
  onNew: () => void;
}) {
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState("");
  useEffect(() => {
    const timer = window.setTimeout(() => setFilter(query), 250);
    return () => window.clearTimeout(timer);
  }, [query]);
  const { data, loading, error } = useLoad(
    () =>
      request<Asset[]>(
        `${agentPath(agentKey)}/assets?images=${images}&q=${encodeURIComponent(filter)}`,
      ),
    [agentKey, images, filter, refresh],
  );
  return (
    <>
      <div className="page-heading">
        <h1>{images ? "Images" : "Attachments"}</h1>
        <button className="primary" onClick={onNew}>
          <Upload size={18} />
          Upload
        </button>
      </div>
      <label className="search-field">
        <Search size={18} />
        <input
          aria-label="Search assets"
          placeholder="Search"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
        />
      </label>
      <ErrorNotice message={error} />
      {loading ? (
        <Loading />
      ) : data?.length === 0 ? (
        <Empty
          action={
            <button onClick={onNew}>
              <Upload size={16} />
              Upload {images ? "image" : "attachment"}
            </button>
          }
        >
          {query
            ? "No matching files."
            : `No ${images ? "images" : "attachments"} yet.`}
        </Empty>
      ) : (
        <div className={images ? "image-grid" : "resource-list"}>
          {data?.map((asset) => (
            <button
              key={asset.id}
              className={images ? "image-card" : "resource-row"}
              onClick={() => onOpen(asset)}
            >
              {images ? (
                <img src={assetUrl(agentKey, asset.id)} alt="" loading="lazy" />
              ) : (
                <File size={20} />
              )}
              <span className="row-content">
                <span className="row-title">{asset.title}</span>
                <span className="row-metadata">
                  {formatBytes(asset.size_bytes)} ·{" "}
                  {images ? formatDate(asset.created_at) : asset.filename}
                </span>
              </span>
              {!images && <ChevronRight size={18} />}
            </button>
          ))}
        </div>
      )}
    </>
  );
}
export function TrashPage({
  agentKey,
  refresh,
  onResource,
  onAsset,
}: {
  agentKey: string;
  refresh: number;
  onResource: (resource: Resource) => void;
  onAsset: (asset: Asset) => void;
}) {
  const { data, loading, error } = useLoad(async () => {
    const resources: Resource[] = [];
    for (let offset = 0; ; offset += 200) {
      const page = await request<Resource[]>(
        `${agentPath(agentKey)}/resources?trash=true&limit=200&offset=${offset}`,
      );
      resources.push(...page);
      if (page.length < 200) break;
    }
    const assets = await request<Asset[]>(
      `${agentPath(agentKey)}/assets?trash=true`,
    );
    return { resources, assets };
  }, [agentKey, refresh]);
  return (
    <>
      <div className="page-heading">
        <h1>Trash</h1>
      </div>
      <ErrorNotice message={error} />
      {loading ? (
        <Loading />
      ) : (
        data &&
        (data.resources.length + data.assets.length === 0 ? (
          <Empty>Trash is empty.</Empty>
        ) : (
          <>
            <ResourceRows
              resources={data.resources}
              onOpen={onResource}
              showKind
            />
            <div className="resource-list">
              {data.assets.map((asset) => (
                <button
                  className="resource-row"
                  key={asset.id}
                  onClick={() => onAsset(asset)}
                >
                  <File size={20} />
                  <span className="row-content">
                    <span className="row-title">{asset.title}</span>
                    <span className="row-metadata">
                      {asset.filename} · {formatBytes(asset.size_bytes)}
                    </span>
                  </span>
                  <ChevronRight size={18} />
                </button>
              ))}
            </div>
          </>
        ))
      )}
    </>
  );
}
