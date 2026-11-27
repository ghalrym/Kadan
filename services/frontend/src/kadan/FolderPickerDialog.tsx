import { useCallback, useEffect, useRef, useState } from "react";
import { ChevronUp, Folder, X } from "lucide-react";
import { browseWorkFilesystem, type FilesystemBrowseResponse } from "./work";

interface Props {
  onSelect: (path: string) => void;
  onClose: () => void;
}

function folderBasename(path: string): string {
  const trimmed = path.replace(/[\\/]+$/, "");
  if (!trimmed) return path;
  const idx = Math.max(trimmed.lastIndexOf("/"), trimmed.lastIndexOf("\\"));
  return idx >= 0 ? trimmed.slice(idx + 1) || trimmed : trimmed;
}

export default function FolderPickerDialog({ onSelect, onClose }: Props) {
  const dialog = useRef<HTMLDialogElement>(null);
  const [browser, setBrowser] = useState<FilesystemBrowseResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const load = useCallback(async (path?: string) => {
    setLoading(true);
    try {
      const next = await browseWorkFilesystem(path);
      setBrowser(next);
      setError("");
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    dialog.current?.showModal();
    void load();
  }, [load]);

  const currentPath = browser?.current_path ?? "";
  const label = currentPath ? folderBasename(currentPath) : "folder";

  return (
    <dialog
      ref={dialog}
      className="rename-agent folder-picker-dialog"
      aria-labelledby="folder-picker-title"
      onCancel={(event) => {
        event.preventDefault();
        onClose();
      }}
    >
      <div className="folder-picker-body">
        <div className="folder-picker-head">
          <h2 id="folder-picker-title">Choose folder</h2>
          <button type="button" className="icon" aria-label="Close" onClick={onClose}>
            <X size={16} />
          </button>
        </div>

        <div className="work-folder-bar">
          <span className="work-folder-current" title={currentPath}>
            {currentPath || "Loading…"}
          </span>
          {browser?.parent_path && (
            <button
              type="button"
              className="button"
              disabled={loading}
              onClick={() => void load(browser.parent_path ?? undefined)}
            >
              <ChevronUp size={14} />
              Up
            </button>
          )}
        </div>

        {error && (
          <p className="error" role="alert">
            {error}
          </p>
        )}

        <div className="work-folder-list" role="listbox" aria-label="Folders">
          {(browser?.entries ?? []).length === 0 && !loading ? (
            <p className="panel-hint work-empty-hint">No subfolders here</p>
          ) : (
            (browser?.entries ?? []).map((entry) => (
              <button
                key={entry.path}
                type="button"
                className="work-folder-entry"
                onClick={() => void load(entry.path)}
              >
                <Folder size={14} />
                <span>{entry.name}</span>
              </button>
            ))
          )}
        </div>

        <div className="form-actions">
          <button type="button" onClick={onClose}>
            Cancel
          </button>
          <button
            type="button"
            className="primary"
            disabled={!currentPath || loading}
            onClick={() => {
              if (currentPath) onSelect(currentPath);
            }}
          >
            Select {label}
          </button>
        </div>
      </div>
    </dialog>
  );
}
