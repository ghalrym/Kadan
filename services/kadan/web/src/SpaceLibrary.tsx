import { useEffect, useRef, useState } from "react";
import {
  ChevronDown,
  ChevronRight,
  Download,
  ExternalLink,
  FileText,
  Folder,
  FolderInput,
  FolderPlus,
  LayoutGrid,
  List,
  MoreHorizontal,
  Pencil,
  Pin,
  PinOff,
  Plus,
  RefreshCw,
  RotateCcw,
  Search,
  Trash2,
  Upload,
  User,
} from "lucide-react";
import { agentPath, api } from "./api";
import { randomUuid } from "./uuid";
import {
  SPACE_TABS,
  childrenOf,
  folderTrail,
  itemMeta,
  moveTargets,
  recentItems,
  sortItems,
  visibleItems,
  type SpaceTab,
} from "./spaceItems";
import SpaceItemView from "./SpaceItemView";
import { ItemIcon, contentUrl } from "./SpaceIcon";
import ConfirmDialog from "./ConfirmDialog";
import type { SpaceItem } from "./types";

interface Props {
  agentKey: string;
}

type Section = "items" | "pinned" | "trash";

const LAYOUT_KEY = "kadan.space.layout";

function readLayout(): "list" | "grid" {
  try {
    return localStorage.getItem(LAYOUT_KEY) === "grid" ? "grid" : "list";
  } catch {
    return "list";
  }
}

/** A small dialog asking for one name, used to create folders and rename items. */
function NameDialog({ title, label, initial, confirm, onSave, onClose }: {
  title: string;
  label: string;
  initial: string;
  confirm: string;
  onSave: (name: string) => Promise<void>;
  onClose: () => void;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  const [name, setName] = useState(initial);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  useEffect(() => { dialog.current?.showModal(); }, []);
  async function save(event: React.FormEvent) {
    event.preventDefault();
    if (!name.trim() || busy) return;
    setBusy(true);
    setError("");
    try {
      await onSave(name.trim());
      onClose();
    } catch (e) {
      setError((e as Error).message);
      setBusy(false);
    }
  }
  return (
    <dialog ref={dialog} className="confirm-dialog" aria-labelledby="name-dialog-title"
      onCancel={(event) => { event.preventDefault(); if (!busy) onClose(); }}>
      <form onSubmit={save}>
        <h2 id="name-dialog-title">{title}</h2>
        <label>
          {label}
          <input autoFocus required maxLength={500} value={name} disabled={busy}
            onFocus={(event) => event.currentTarget.select()}
            onChange={(event) => setName(event.target.value)} />
        </label>
        {error && <p className="error" role="alert">{error}</p>}
        <div className="form-actions">
          <button type="button" disabled={busy} onClick={onClose}>Cancel</button>
          <button className="primary" disabled={busy || !name.trim()}>{busy ? "Saving…" : confirm}</button>
        </div>
      </form>
    </dialog>
  );
}

/** Close a popover on outside click or Escape. */
function useDismiss(open: boolean, ref: React.RefObject<HTMLElement | null>, close: () => void) {
  useEffect(() => {
    if (!open) return;
    const outside = (event: PointerEvent) => { if (!ref.current?.contains(event.target as Node)) close(); };
    const escape = (event: KeyboardEvent) => { if (event.key === "Escape") close(); };
    document.addEventListener("pointerdown", outside);
    document.addEventListener("keydown", escape);
    return () => {
      document.removeEventListener("pointerdown", outside);
      document.removeEventListener("keydown", escape);
    };
  }, [open]);
}

function ItemMenu({ agentKey, item, items, trash, onOpen, onRename, onPin, onMove, onTrash, onRestore }: {
  agentKey: string;
  item: SpaceItem;
  items: SpaceItem[];
  trash: boolean;
  onOpen: () => void;
  onRename: () => void;
  onPin: () => void;
  onMove: (parentId: string | null) => void;
  onTrash: () => void;
  onRestore: () => void;
}) {
  const [open, setOpen] = useState(false);
  const [moving, setMoving] = useState(false);
  const box = useRef<HTMLDivElement>(null);
  useDismiss(open, box, () => { setOpen(false); setMoving(false); });
  const targets = moving ? moveTargets(items, item).filter((folder) => folder.id !== item.parent_id) : [];
  const run = (action: () => void) => () => { setOpen(false); setMoving(false); action(); };
  return (
    <div className="item-menu" ref={box}>
      <button className="icon item-more" aria-label={`Options for ${item.title}`} aria-haspopup="menu" aria-expanded={open}
        onClick={(event) => { event.stopPropagation(); setOpen(!open); setMoving(false); }}>
        <MoreHorizontal size={16} />
      </button>
      {open && (
        <div className="menu" role="menu" onClick={(event) => event.stopPropagation()}>
          {trash ? (
            <button role="menuitem" onClick={run(onRestore)}><RotateCcw size={14} />Restore</button>
          ) : moving ? (
            <>
              <p className="menu-label">Move to</p>
              {item.parent_id && (
                <button role="menuitem" onClick={run(() => onMove(null))}><User size={14} />Your items</button>
              )}
              {targets.map((folder) => (
                <button role="menuitem" key={folder.id} onClick={run(() => onMove(folder.id))}>
                  <Folder size={14} /><span className="menu-text">{folder.title}</span>
                </button>
              ))}
              {!item.parent_id && targets.length === 0 && <p className="menu-label">Create a folder first.</p>}
            </>
          ) : (
            <>
              <button role="menuitem" onClick={run(onOpen)}><ExternalLink size={14} />Open</button>
              <button role="menuitem" onClick={run(onRename)}><Pencil size={14} />Rename</button>
              <button role="menuitem" onClick={run(onPin)}>
                {item.pinned ? <PinOff size={14} /> : <Pin size={14} />}{item.pinned ? "Unpin" : "Pin"}
              </button>
              <button role="menuitem" onClick={() => setMoving(true)}><FolderInput size={14} />Move to…</button>
              {(item.type === "image" || item.type === "file") && (
                <a role="menuitem" className="menu-link" href={contentUrl(agentKey, item.id, true)} onClick={() => setOpen(false)}>
                  <Download size={14} />Download
                </a>
              )}
              <hr />
              <button role="menuitem" className="danger" onClick={run(onTrash)}><Trash2 size={14} />Move to trash</button>
            </>
          )}
        </div>
      )}
    </div>
  );
}

export default function SpaceLibrary({ agentKey }: Props) {
  const base = `${agentPath(agentKey)}/space`;
  const [items, setItems] = useState<SpaceItem[]>([]);
  const [trashed, setTrashed] = useState<SpaceItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [revision, setRevision] = useState(0);
  const [section, setSection] = useState<Section>("items");
  const [folderId, setFolderId] = useState<string | null>(null);
  const [tab, setTab] = useState<SpaceTab>("all");
  const [query, setQuery] = useState("");
  const [layout, setLayout] = useState<"list" | "grid">(readLayout);
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [openItem, setOpenItem] = useState<SpaceItem | null>(null);
  const [editOnOpen, setEditOnOpen] = useState(false);
  const [naming, setNaming] = useState<{ kind: "folder" } | { kind: "rename"; item: SpaceItem } | null>(null);
  const [confirm, setConfirm] = useState<SpaceItem | null>(null);
  const [createOpen, setCreateOpen] = useState(false);
  const [uploading, setUploading] = useState(0);
  const createBox = useRef<HTMLDivElement>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  useDismiss(createOpen, createBox, () => setCreateOpen(false));

  useEffect(() => {
    let current = true;
    setLoading(true);
    Promise.all([api<SpaceItem[]>(`${base}/items`), api<SpaceItem[]>(`${base}/trash`)])
      .then(([active, trash]) => {
        if (!current) return;
        setItems(active);
        setTrashed(trash);
        setError("");
      })
      .catch((e) => { if (current) setError(`Could not load Space: ${(e as Error).message}`); })
      .finally(() => { if (current) setLoading(false); });
    return () => { current = false; };
  }, [base, revision]);

  // A folder that disappears (trashed elsewhere) drops back to the top level.
  useEffect(() => {
    if (folderId && !loading && !items.some((item) => item.id === folderId)) setFolderId(null);
  }, [items, folderId, loading]);

  function changeLayout(next: "list" | "grid") {
    setLayout(next);
    try { localStorage.setItem(LAYOUT_KEY, next); } catch { /* per-browser preference only */ }
  }
  function replace(item: SpaceItem) {
    setItems((rows) => rows.some((row) => row.id === item.id)
      ? rows.map((row) => row.id === item.id ? item : row) : [item, ...rows]);
    setOpenItem((current) => current?.id === item.id ? item : current);
  }
  async function act(action: () => Promise<void>) {
    setError("");
    setNotice("");
    try {
      await action();
    } catch (e) {
      setError((e as Error).message);
    }
  }
  const update = (item: SpaceItem, change: Record<string, unknown>) =>
    api<SpaceItem>(`${base}/items/${encodeURIComponent(item.id)}`, change, "PUT").then(replace);

  function open(item: SpaceItem) {
    if (item.type === "folder") {
      setSection("items");
      setTab("all");
      setQuery("");
      setFolderId(item.id);
      setOpenItem(null);
      return;
    }
    setEditOnOpen(false);
    setOpenItem(item);
  }
  async function createPage() {
    setCreateOpen(false);
    await act(async () => {
      const page = await api<SpaceItem>(`${base}/pages`, { title: "Untitled page", markdown: "", parent_id: folderId }, "POST");
      replace(page);
      setEditOnOpen(true);
      setOpenItem(page);
    });
  }
  async function upload(files: FileList | null) {
    if (!files?.length) return;
    const list = [...files];
    setUploading((n) => n + list.length);
    await act(async () => {
      for (const file of list) {
        try {
          const body = new FormData();
          body.append("asset_id", randomUuid());
          body.append("file", file);
          if (folderId) body.append("folder_id", folderId);
          await api<{ id: string }>(`${agentPath(agentKey)}/assets`, body, "POST");
        } finally {
          setUploading((n) => n - 1);
        }
      }
      setNotice(list.length === 1 ? `Uploaded ${list[0].name}.` : `Uploaded ${list.length} files.`);
      setRevision((n) => n + 1);
    });
  }

  const trail = folderTrail(items, folderId);
  const folder = trail.at(-1) ?? null;
  const inTrash = section === "trash";
  const effectiveTab: SpaceTab = section === "pinned" ? "pinned" : tab;
  const rows = inTrash
    ? sortItems(trashed.filter((item) => !query.trim() || item.title.toLowerCase().includes(query.trim().toLowerCase())))
    : visibleItems(items, effectiveTab, query, folderId);
  const browsing = !inTrash && effectiveTab === "all" && !query.trim();
  const pinnedCount = items.filter((item) => item.pinned).length;
  const heading = inTrash ? "Trash" : section === "pinned" ? "Pinned" : folder?.title ?? "Your items";
  const now = new Date();

  function renderRow(item: SpaceItem, depth = 0): React.ReactNode {
    const kids = browsing && item.type === "page" ? childrenOf(items, item.id) : [];
    const isOpen = expanded.has(item.id);
    return (
      <div key={item.id} className="item-group">
        <div className={`item-row ${item.deleted_at ? "is-trashed" : ""}`} style={{ paddingLeft: depth * 24 }}>
          <span className="item-expander">
            {kids.length > 0 && (
              <button className="icon" aria-label={isOpen ? `Collapse ${item.title}` : `Expand ${item.title}`} aria-expanded={isOpen}
                onClick={() => setExpanded((current) => {
                  const next = new Set(current);
                  if (isOpen) next.delete(item.id); else next.add(item.id);
                  return next;
                })}>
                {isOpen ? <ChevronDown size={15} /> : <ChevronRight size={15} />}
              </button>
            )}
          </span>
          <button className="item-open" disabled={inTrash} onClick={() => open(item)}>
            <span className={`item-icon is-${item.type}`}><ItemIcon agentKey={agentKey} item={item} /></span>
            <span className="item-text">
              <strong>{item.title}</strong>
              <span>{itemMeta(item, now)}</span>
            </span>
          </button>
          {item.pinned && !inTrash && <Pin size={14} className="item-pin" aria-label="Pinned" />}
          <ItemMenu agentKey={agentKey} item={item} items={items} trash={inTrash}
            onOpen={() => open(item)}
            onRename={() => setNaming({ kind: "rename", item })}
            onPin={() => void act(() => update(item, { pinned: !item.pinned }))}
            onMove={(parentId) => void act(async () => {
              await update(item, { parent_id: parentId });
              setNotice(`Moved ${item.title}.`);
            })}
            onTrash={() => setConfirm(item)}
            onRestore={() => void act(async () => {
              await api<SpaceItem>(`${base}/items/${encodeURIComponent(item.id)}/restore`, {}, "POST");
              setNotice(`Restored ${item.title}.`);
              setRevision((n) => n + 1);
            })} />
        </div>
        {isOpen && kids.map((child) => renderRow(child, depth + 1))}
      </div>
    );
  }

  return (
    <section className="space-library" aria-label="Space">
      <aside className="space-panel" aria-label="Space sections">
        <header className="space-panel-head">
          <h1>Space</h1>
          <button className="icon" aria-label="Refresh Space" title="Refresh" disabled={loading}
            onClick={() => setRevision((n) => n + 1)}>
            <RefreshCw size={15} className={loading ? "spin" : ""} />
          </button>
        </header>
        <nav className="space-panel-nav">
          <button className={section === "items" && !folderId ? "selected" : ""}
            onClick={() => { setSection("items"); setFolderId(null); setOpenItem(null); }}>
            <User size={16} />Your items
          </button>
          <button className={section === "pinned" ? "selected" : ""}
            onClick={() => { setSection("pinned"); setOpenItem(null); }}>
            <Pin size={16} />Pinned{pinnedCount > 0 && <span className="count">{pinnedCount}</span>}
          </button>
        </nav>
        <div className="space-recents">
          <h2 className="panel-label">Recents</h2>
          {recentItems(items).map((item) => (
            <button key={item.id} onClick={() => open(item)} title={item.title}>
              <span className={`item-icon small is-${item.type}`}><ItemIcon agentKey={agentKey} item={item} size={15} /></span>
              <span className="menu-text">{item.title}</span>
            </button>
          ))}
          {!loading && items.length === 0 && <p className="panel-hint">Nothing here yet.</p>}
        </div>
        <nav className="space-panel-nav space-panel-foot">
          <button className={inTrash ? "selected" : ""} onClick={() => { setSection("trash"); setOpenItem(null); }}>
            <Trash2 size={16} />Trash{trashed.length > 0 && <span className="count">{trashed.length}</span>}
          </button>
        </nav>
      </aside>
      <div className="space-main">
        {openItem ? (
          <SpaceItemView
            key={openItem.id}
            agentKey={agentKey}
            item={openItem}
            items={items}
            startEditing={editOnOpen}
            onBack={() => setOpenItem(null)}
            onOpen={open}
            onSaved={replace}
          />
        ) : (
          <div className="space-inner">
            <header className="space-head">
              <div className="space-title">
                {trail.length > 0 && browsing && (
                  <nav className="space-crumbs" aria-label="Folder path">
                    <button onClick={() => setFolderId(null)}>Your items</button>
                    {trail.slice(0, -1).map((crumb) => (
                      <span key={crumb.id}><ChevronRight size={13} aria-hidden="true" />
                        <button onClick={() => setFolderId(crumb.id)}>{crumb.title}</button></span>
                    ))}
                  </nav>
                )}
                <h2>{heading}</h2>
              </div>
              <label className="search-field space-search">
                <Search size={16} />
                <span className="sr-only">Search Space</span>
                <input type="search" placeholder="Search" value={query} onChange={(event) => setQuery(event.target.value)} />
              </label>
              {!inTrash && (
                <div className="space-create" ref={createBox}>
                  <button className="space-create-button" aria-haspopup="menu" aria-expanded={createOpen}
                    onClick={() => setCreateOpen(!createOpen)}>
                    Create <ChevronDown size={15} />
                  </button>
                  {createOpen && (
                    <div className="menu" role="menu">
                      <button role="menuitem" onClick={() => void createPage()}><FileText size={14} />Page</button>
                      <button role="menuitem" onClick={() => { setCreateOpen(false); setNaming({ kind: "folder" }); }}>
                        <FolderPlus size={14} />Folder
                      </button>
                      <hr />
                      <button role="menuitem" onClick={() => { setCreateOpen(false); fileInput.current?.click(); }}>
                        <Upload size={14} />Upload files
                      </button>
                    </div>
                  )}
                  <input ref={fileInput} type="file" multiple hidden
                    onChange={(event) => { void upload(event.target.files); event.target.value = ""; }} />
                </div>
              )}
            </header>
            {!inTrash && (
              <div className="space-toolbar">
                {section === "items" && (
                  <div className="space-tabs" role="tablist" aria-label="Item type">
                    {SPACE_TABS.filter((entry) => entry.id !== "pinned").map((entry) => (
                      <button key={entry.id} role="tab" aria-selected={tab === entry.id}
                        className={tab === entry.id ? "active" : ""} onClick={() => setTab(entry.id)}>
                        {entry.label}
                      </button>
                    ))}
                  </div>
                )}
                <div className="space-layout" role="group" aria-label="Layout">
                  <button className={`icon ${layout === "grid" ? "active" : ""}`} aria-label="Grid view" aria-pressed={layout === "grid"}
                    onClick={() => changeLayout("grid")}><LayoutGrid size={16} /></button>
                  <button className={`icon ${layout === "list" ? "active" : ""}`} aria-label="List view" aria-pressed={layout === "list"}
                    onClick={() => changeLayout("list")}><List size={16} /></button>
                </div>
              </div>
            )}
            {error && (
              <div className="error feedback" role="alert">
                {error}
                <button onClick={() => setRevision((n) => n + 1)}>Retry</button>
              </div>
            )}
            {notice && <p className="plugin-notice" role="status">{notice}</p>}
            {uploading > 0 && <p className="plugin-notice" role="status">Uploading {uploading} file{uploading === 1 ? "" : "s"}…</p>}
            {inTrash && rows.length > 0 && (
              <p className="field-hint">Items in trash can be restored. A page or file inside a trashed folder comes back once its folder is restored.</p>
            )}
            {loading && !items.length && !trashed.length ? (
              <p className="page-empty" role="status">Loading Space…</p>
            ) : rows.length === 0 ? (
              <div className="space-empty">
                {inTrash ? <Trash2 size={22} /> : <FileText size={22} />}
                <p>{query.trim() ? `Nothing matches “${query.trim()}”.`
                  : inTrash ? "Trash is empty."
                    : section === "pinned" ? "Pin items to keep them here."
                      : folder ? "This folder is empty."
                        : "Pages, slides and files you or your agent save appear here."}</p>
                {!inTrash && section === "items" && !query.trim() && (
                  <button className="primary" onClick={() => void createPage()}><Plus size={15} />New page</button>
                )}
              </div>
            ) : layout === "grid" && !inTrash ? (
              <div className="item-grid">
                {rows.map((item) => (
                  <div key={item.id} className="item-card">
                    <button className="item-card-open" onClick={() => open(item)}>
                      <span className={`item-card-art is-${item.type}`}><ItemIcon agentKey={agentKey} item={item} size={30} /></span>
                      <span className="item-text">
                        <strong>{item.title}</strong>
                        <span>{itemMeta(item, now)}</span>
                      </span>
                    </button>
                    {item.pinned && <Pin size={13} className="item-pin" aria-label="Pinned" />}
                    <ItemMenu agentKey={agentKey} item={item} items={items} trash={false}
                      onOpen={() => open(item)}
                      onRename={() => setNaming({ kind: "rename", item })}
                      onPin={() => void act(() => update(item, { pinned: !item.pinned }))}
                      onMove={(parentId) => void act(() => update(item, { parent_id: parentId }))}
                      onTrash={() => setConfirm(item)}
                      onRestore={() => {}} />
                  </div>
                ))}
              </div>
            ) : (
              <div className="item-list">{rows.map((item) => renderRow(item))}</div>
            )}
          </div>
        )}
      </div>
      {naming && <NameDialog
        title={naming.kind === "folder" ? "New folder" : "Rename"}
        label="Name"
        initial={naming.kind === "folder" ? "" : naming.item.title}
        confirm={naming.kind === "folder" ? "Create" : "Save"}
        onSave={async (name) => {
          if (naming.kind === "folder") {
            replace(await api<SpaceItem>(`${base}/folders`, { title: name, parent_id: folderId }, "POST"));
          } else {
            await update(naming.item, { title: name });
          }
        }}
        onClose={() => setNaming(null)}
      />}
      {confirm && <ConfirmDialog
        title="Move to trash?"
        message={confirm.type === "folder"
          ? `“${confirm.title}” and everything in it will move to Trash, where it can be restored.`
          : `“${confirm.title}” will move to Trash, where it can be restored.`}
        confirm="Move to trash"
        onConfirm={async () => {
          await api<void>(`${base}/items/${encodeURIComponent(confirm.id)}`, undefined, "DELETE");
          if (openItem?.id === confirm.id) setOpenItem(null);
          setRevision((n) => n + 1);
        }}
        onClose={() => setConfirm(null)}
      />}
    </section>
  );
}
