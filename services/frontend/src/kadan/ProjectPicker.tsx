import { useCallback, useRef, useState } from "react";
import { Check, ChevronDown, Folder, FolderPlus, Settings } from "lucide-react";
import { usePopover } from "./popover";
import type { WorkProject } from "./work";

interface Props {
  projects: WorkProject[];
  value: string;
  disabled?: boolean;
  onChange: (projectId: string) => void;
  onAddFolder: () => void;
  onManage?: () => void;
}

export default function ProjectPicker({ projects, value, disabled, onChange, onAddFolder, onManage }: Props) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);
  const close = useCallback(() => setOpen(false), []);
  const placement = usePopover(open, close, rootRef, 280);
  const current = projects.find((p) => p.id === value) ?? null;

  if (projects.length === 0) {
    return (
      <button type="button" className="work-meta-btn" disabled={disabled} onClick={onAddFolder}>
        <FolderPlus size={14} />
        Add a project folder
      </button>
    );
  }

  return (
    <div className="project-picker" ref={rootRef}>
      <button
        type="button"
        className={`work-meta-btn ${open ? "is-open" : ""}`}
        aria-haspopup="menu"
        aria-expanded={open}
        aria-label={`Project: ${current?.title ?? "none"}`}
        title={current?.root_path}
        disabled={disabled}
        onClick={() => setOpen((v) => !v)}
      >
        <Folder size={14} />
        <span className="work-meta-btn-label">{current?.title ?? "Choose project"}</span>
        <ChevronDown size={13} className="work-meta-chevron" />
      </button>
      {open && (
        <div className={`menu project-picker-menu is-${placement}`} role="menu">
          <p className="menu-label">Projects</p>
          {projects.map((p) => (
            <button
              key={p.id}
              type="button"
              role="menuitemradio"
              aria-checked={p.id === value}
              title={p.root_path}
              onClick={() => {
                onChange(p.id);
                setOpen(false);
              }}
            >
              <Folder size={14} />
              <span className="menu-text">{p.title}</span>
              {p.id === value && <Check size={14} className="menu-check" />}
            </button>
          ))}
          <hr />
          <button
            type="button"
            role="menuitem"
            onClick={() => {
              setOpen(false);
              onAddFolder();
            }}
          >
            <FolderPlus size={14} />
            <span className="menu-text">Add folder…</span>
          </button>
          {onManage && (
            <button
              type="button"
              role="menuitem"
              onClick={() => {
                setOpen(false);
                onManage();
              }}
            >
              <Settings size={14} />
              <span className="menu-text">Manage projects</span>
            </button>
          )}
        </div>
      )}
    </div>
  );
}
