import { useState } from "react";
import { Download, RotateCcw, Save, Trash2, Upload } from "lucide-react";
import {
  agentPath,
  assetUrl,
  formatBytes,
  formatDate,
  json,
  request,
} from "./api";
import { Dialog, ErrorNotice, SourcesField, TagsField } from "./components";
import type { Asset, SourceLink } from "./types";

export function AssetEditor({
  agentKey,
  initial,
  imagesOnly,
  onClose,
  onChanged,
}: {
  agentKey: string;
  initial: Asset | null;
  imagesOnly: boolean;
  onClose: () => void;
  onChanged: () => void;
}) {
  const [asset, setAsset] = useState(initial);
  const [file, setFile] = useState<File | null>(null);
  const [title, setTitle] = useState(initial?.title || "");
  const [tags, setTags] = useState<string[]>(initial?.tags || []);
  const [sources, setSources] = useState<SourceLink[]>(initial?.sources || []);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState("");
  const dirty =
    Boolean(file) ||
    title !== (asset?.title || "") ||
    JSON.stringify(tags) !== JSON.stringify(asset?.tags || []) ||
    JSON.stringify(sources) !== JSON.stringify(asset?.sources || []);
  const close = () => {
    if (!busy && (!dirty || window.confirm("Discard unsaved changes?")))
      onClose();
  };
  const mutate = async (
    operation: () => Promise<Asset>,
    closeAfter = false,
  ) => {
    setBusy(true);
    setError("");
    setNotice("");
    try {
      const result = await operation();
      setAsset(result);
      setTitle(result.title);
      setTags(result.tags);
      setSources(result.sources);
      setFile(null);
      onChanged();
      setNotice("Saved.");
      if (closeAfter) onClose();
    } catch (error) {
      setError(
        error instanceof Error ? error.message : "Unable to save asset.",
      );
    } finally {
      setBusy(false);
    }
  };
  return (
    <Dialog
      title={
        asset?.title || (imagesOnly ? "Upload image" : "Upload attachment")
      }
      onClose={close}
    >
      <form
        onSubmit={(event) => {
          event.preventDefault();
          if (asset)
            void mutate(() =>
              request<Asset>(
                `${agentPath(agentKey)}/assets/${asset.id}`,
                json("PUT", {
                  title,
                  tags,
                  sources,
                  expected_version: asset.version,
                }),
              ),
            );
          else if (file) {
            const form = new FormData();
            form.append("file", file);
            form.append("title", title || file.name);
            form.append("tags", JSON.stringify(tags));
            form.append("sources", JSON.stringify(sources));
            void mutate(() =>
              request<Asset>(`${agentPath(agentKey)}/assets`, {
                method: "POST",
                body: form,
              }),
            );
          }
        }}
      >
        <div className="dialog-scroll">
          <ErrorNotice message={error} />
          {asset && (
            <div className="asset-preview">
              {asset.is_image ? (
                <img src={assetUrl(agentKey, asset.id)} alt={asset.title} />
              ) : asset.media_type.startsWith("audio/") ? (
                <audio controls src={assetUrl(agentKey, asset.id)} />
              ) : asset.media_type.startsWith("video/") ? (
                <video controls src={assetUrl(agentKey, asset.id)} />
              ) : null}
            </div>
          )}
          <fieldset disabled={busy || Boolean(asset?.deleted_at)}>
            {!asset && (
              <label>
                File
                <input
                  type="file"
                  required
                  accept={
                    imagesOnly
                      ? "image/png,image/jpeg,image/webp,image/gif"
                      : undefined
                  }
                  onChange={(event) => {
                    const selected = event.target.files?.[0] || null;
                    setFile(selected);
                    if (selected && !title) setTitle(selected.name);
                  }}
                />
              </label>
            )}
            <label>
              Title
              <input
                required
                value={title}
                onChange={(event) => setTitle(event.target.value)}
              />
            </label>
            <TagsField value={tags} onChange={setTags} />
            <SourcesField value={sources} onChange={setSources} />
          </fieldset>
          {asset && (
            <dl className="record-details">
              <div>
                <dt>Filename</dt>
                <dd>{asset.filename}</dd>
              </div>
              <div>
                <dt>Type</dt>
                <dd>{asset.media_type}</dd>
              </div>
              <div>
                <dt>Size</dt>
                <dd>{formatBytes(asset.size_bytes)}</dd>
              </div>
              <div>
                <dt>Version</dt>
                <dd>{asset.version}</dd>
              </div>
              <div>
                <dt>Created</dt>
                <dd>{formatDate(asset.created_at)}</dd>
              </div>
              <div>
                <dt>SHA-256</dt>
                <dd className="mono">{asset.sha256}</dd>
              </div>
            </dl>
          )}
        </div>
        <div className="dialog-footer">
          <span className="footer-status" role="status">
            {notice ||
              (asset?.deleted_at ? "In trash" : dirty ? "Unsaved changes" : "")}
          </span>
          {asset && (
            <a
              className="button icon-button"
              href={assetUrl(agentKey, asset.id, true)}
              aria-label="Download file"
            >
              <Download size={18} />
            </a>
          )}
          {asset && !asset.deleted_at && (
            <button
              type="button"
              className="danger icon-button"
              aria-label="Move to trash"
              disabled={busy}
              onClick={() => {
                if (window.confirm(`Move “${asset.title}” to trash?`))
                  void mutate(
                    () =>
                      request<Asset>(
                        `${agentPath(agentKey)}/assets/${asset.id}?expected_version=${asset.version}`,
                        json("DELETE"),
                      ),
                    true,
                  );
              }}
            >
              <Trash2 size={18} />
            </button>
          )}
          {asset?.deleted_at ? (
            <button
              type="button"
              className="primary"
              disabled={busy}
              onClick={() =>
                void mutate(() =>
                  request<Asset>(
                    `${agentPath(agentKey)}/assets/${asset.id}/restore`,
                    json("POST", { expected_version: asset.version }),
                  ),
                )
              }
            >
              <RotateCcw size={16} />
              Restore
            </button>
          ) : (
            <button className="primary" disabled={busy || (!asset && !file)}>
              {asset ? <Save size={16} /> : <Upload size={16} />}{" "}
              {busy ? "Saving…" : asset ? "Save changes" : "Upload"}
            </button>
          )}
        </div>
      </form>
    </Dialog>
  );
}
