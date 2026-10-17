import { useEffect, useRef, useState } from "react";
import { Check, ImagePlus, RefreshCw, Sparkles, X } from "lucide-react";
import { agentPath, api } from "./api";
import AgentAvatar from "./AgentAvatar";
import StillImage from "./StillImage";
import type { Agent, Asset } from "./types";

export default function AvatarEditor({
  agent,
  open,
  browserUrl,
  onChange,
  onClose,
  onGenerating,
}: {
  agent: Agent;
  open: boolean;
  onGenerating: (generating: boolean) => void;
  browserUrl: string | null;
  onChange: (agent: Agent) => void;
  onClose: () => void;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  const [images, setImages] = useState<Asset[]>([]);
  const [selected, setSelected] = useState<string | null>(
    agent.avatar_asset_id,
  );
  const [mode, setMode] = useState<"images" | "generate">("images");
  const [loading, setLoading] = useState(true);
  const [pending, setPending] = useState<"generate" | "save" | null>(null);
  const [error, setError] = useState("");
  const [loadError, setLoadError] = useState("");
  const [notice, setNotice] = useState("");
  const [prompt, setPrompt] = useState("");
  async function loadImages() {
    setLoading(true);
    setLoadError("");
    try {
      const assets = await api<Asset[]>(`${agentPath(agent.key)}/assets`);
      setImages(
        assets.filter(
          (asset) =>
            asset.is_image &&
            ["image/png", "image/jpeg", "image/webp", "image/gif"].includes(
              asset.media_type,
            ),
        ),
      );
    } catch (e) {
      setLoadError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }
  useEffect(() => {
    const element = dialog.current;
    if (open) {
      element?.showModal();
      void loadImages();
    } else element?.close();
    return () => element?.close();
  }, [open]);
  async function generate(event: React.FormEvent) {
    event.preventDefault();
    if (pending || !prompt.trim()) return;
    setPending("generate");
    onGenerating(true);
    setError("");
    setNotice("");
    try {
      const image = await api<Asset>(
        `${agentPath(agent.key)}/avatar/generations`,
        { prompt: prompt.trim() },
      );
      setImages((rows) => [
        image,
        ...rows.filter((row) => row.id !== image.id),
      ]);
      setSelected(image.id);
      setNotice("Image saved in Tsuchi. Select Save avatar to use it.");
      setMode("images");
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setPending(null);
      onGenerating(false);
    }
  }
  async function save() {
    if (pending) return;
    setPending("save");
    setError("");
    try {
      const updated = await api<Agent>(
        `${agentPath(agent.key)}/avatar`,
        { asset_id: selected },
        "PUT",
      );
      onChange(updated);
      setNotice("");
      onClose();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setPending(null);
    }
  }
  return (
    <dialog
      ref={dialog}
      className="avatar-dialog"
      aria-labelledby="avatar-title"
      onCancel={(event) => {
        event.preventDefault();
        if (pending !== "save") onClose();
      }}
    >
      <div className="dialog-header">
        <h2 id="avatar-title">Avatar · {agent.name}</h2>
        <button
          className="icon"
          aria-label="Close avatar settings"
          disabled={pending === "save"}
          onClick={onClose}
        >
          <X size={18} />
        </button>
      </div>
      <div className="avatar-preview">
        <AgentAvatar
          agentKey={agent.key}
          name={agent.name}
          assetId={selected}
          size="large"
          activity={pending ? "working" : "idle"}
          follow={open}
        />
        <button
          disabled={Boolean(pending) || selected === null}
          onClick={() => {
            setSelected(null);
            setNotice("");
          }}
        >
          Use initials
        </button>
      </div>
      <div className="avatar-modes" role="group" aria-label="Avatar source">
        <button
          aria-pressed={mode === "images"}
          disabled={Boolean(pending)}
          onClick={() => setMode("images")}
        >
          <ImagePlus size={16} />
          Saved images
        </button>
        <button
          aria-pressed={mode === "generate"}
          disabled={Boolean(pending)}
          onClick={() => setMode("generate")}
        >
          <Sparkles size={16} />
          Generate
        </button>
      </div>
      {mode === "images" ? (
        <div className="avatar-library">
          <div className="avatar-library-actions">
            {browserUrl && (
              <a href={browserUrl} target="_blank" rel="noreferrer">
                Upload in Tsuchi
              </a>
            )}
            <button
              disabled={loading || Boolean(pending)}
              onClick={() => void loadImages()}
            >
              <RefreshCw size={14} />
              Refresh images
            </button>
          </div>
          {loading ? (
            <p role="status">Loading images…</p>
          ) : loadError ? (
            <p className="error" role="alert">
              {loadError}
            </p>
          ) : !images.length ? (
            <p className="quiet">
              No saved images. Generate an avatar or upload an image in Tsuchi.
            </p>
          ) : null}
          <div className="avatar-grid" aria-label="Saved images">
            {images.map((image) => (
              <button
                key={image.id}
                disabled={Boolean(pending)}
                aria-label={`Use ${image.title}`}
                aria-pressed={selected === image.id}
                title={image.title}
                onClick={() => {
                  setSelected(image.id);
                  setNotice("");
                }}
              >
                <StillImage
                  src={`/api${agentPath(agent.key)}/assets/${image.id}/content`}
                />
                <span>{image.title}</span>
                {selected === image.id && (
                  <Check
                    size={16}
                    className="avatar-check"
                    aria-hidden="true"
                  />
                )}
              </button>
            ))}
          </div>
        </div>
      ) : (
        <form onSubmit={generate}>
          <label htmlFor="avatar-prompt">Image description</label>
          <textarea
            id="avatar-prompt"
            value={prompt}
            maxLength={7000}
            rows={4}
            required
            disabled={Boolean(pending)}
            onChange={(event) => setPrompt(event.target.value)}
            placeholder="A small moss-green robot, centered portrait, plain warm gray background"
          />
          <button
            className="button"
            disabled={Boolean(pending) || !prompt.trim()}
          >
            <Sparkles size={16} />
            {pending === "generate" ? "Generating…" : "Generate image"}
          </button>
          <p className="field-hint">
            Uses Hikari’s selected image model. Generation may take several
            minutes.
          </p>
        </form>
      )}
      {pending === "generate" && (
        <p role="status">
          Generating in Hikari. You can close this window and return later.
        </p>
      )}
      {notice && (
        <p className="success" role="status">
          {notice}
        </p>
      )}
      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}
      <div className="form-actions">
        <button disabled={pending === "save"} onClick={onClose}>
          Close
        </button>
        <button
          className="primary"
          disabled={Boolean(pending) || selected === agent.avatar_asset_id}
          onClick={() => void save()}
        >
          <Check size={16} />
          {pending === "save" ? "Saving…" : "Save avatar"}
        </button>
      </div>
    </dialog>
  );
}
