import { useCallback, useEffect, useState } from "react";
import {
  ArrowUpRight,
  ChevronUp,
  Folder,
  ImagePlus,
  KeyRound,
  Plus,
  RefreshCw,
  Trash2,
  X,
} from "lucide-react";
import {
  browseWorkFilesystem,
  createWorkProject,
  createWorkThread,
  deleteWorkProject,
  getHarnessSignIn,
  harnessAuthLabel,
  listWorkHarnesses,
  listWorkProjects,
  signOutHarness,
  startHarnessSignIn,
  submitHarnessSignIn,
  type FilesystemBrowseResponse,
  type WorkHarness,
  type WorkProject,
} from "./work";

interface Props {
  initialTab?: "new" | "projects" | "harnesses";
  onThreadCreated: (threadId: string) => void;
  onChanged: () => void;
}

export default function WorkSetup({ initialTab = "new", onThreadCreated, onChanged }: Props) {
  const [tab, setTab] = useState<"new" | "projects" | "harnesses">(initialTab);
  const [projects, setProjects] = useState<WorkProject[]>([]);
  const [harnesses, setHarnesses] = useState<WorkHarness[]>([]);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");

  // New thread form state.
  const [projectId, setProjectId] = useState("");
  const [harnessId, setHarnessId] = useState("");
  const [modelId, setModelId] = useState("");
  const [title, setTitle] = useState("");
  const [prompt, setPrompt] = useState("");
  const [attachedImages, setAttachedImages] = useState<{ name: string; dataUrl: string }[]>([]);

  // Add project folder form state.
  const [folderBrowser, setFolderBrowser] = useState<FilesystemBrowseResponse | null>(null);
  const [selectedFolder, setSelectedFolder] = useState("");
  const [projectTitle, setProjectTitle] = useState("");

  // Harness sign-in code / key inputs keyed by harness ID.
  const [signInInputs, setSignInInputs] = useState<Record<string, string>>({});

  useEffect(() => {
    setTab(initialTab);
  }, [initialTab]);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const [projList, harnessList] = await Promise.all([
        listWorkProjects(),
        listWorkHarnesses(),
      ]);
      setProjects(projList);
      setHarnesses(harnessList);
      setProjectId((prev) => prev || projList[0]?.id || "");
      const readyHarness = harnessList.find((h) => h.auth.status === "ready") ?? harnessList[0];
      if (readyHarness) {
        setHarnessId((prev) => prev || readyHarness.id);
        setModelId((prev) => prev || readyHarness.default_model || "");
      }
      setError("");
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const loadFolder = useCallback(async (path?: string) => {
    try {
      const res = await browseWorkFilesystem(path);
      setFolderBrowser(res);
      setSelectedFolder(res.current_path);
    } catch (err) {
      setError((err as Error).message);
    }
  }, []);

  useEffect(() => {
    if (tab === "projects" && !folderBrowser) {
      void loadFolder();
    }
  }, [tab, folderBrowser, loadFolder]);

  // Poll any harness currently waiting on browser device sign-in.
  useEffect(() => {
    const waiting = harnesses.filter(
      (h) => h.auth.status === "starting" || h.auth.status === "waiting" || h.auth.status === "verifying",
    );
    if (waiting.length === 0) return;
    const timer = setInterval(() => {
      for (const h of waiting) {
        void getHarnessSignIn(h.id)
          .then((nextAuth) => {
            setHarnesses((prev) =>
              prev.map((item) => (item.id === h.id ? { ...item, auth: nextAuth } : item)),
            );
            if (nextAuth.status === "ready") {
              setNotice(`${h.display_name} is now connected.`);
            }
          })
          .catch(() => {});
      }
    }, 2500);
    return () => clearInterval(timer);
  }, [harnesses]);

  const selectedHarness = harnesses.find((h) => h.id === harnessId) ?? null;

  function handleSelectImages(files: FileList | null) {
    if (!files || files.length === 0) return;
    const allowed = Array.from(files).filter(
      (f) => f.type === "image/png" || f.type === "image/jpeg",
    );
    if (allowed.length < files.length) {
      setError("Only PNG and JPEG screenshots can be attached.");
    }
    for (const file of allowed.slice(0, Math.max(0, 8 - attachedImages.length))) {
      const reader = new FileReader();
      reader.onload = () => {
        if (typeof reader.result === "string") {
          setAttachedImages((prev) => [
            ...prev.slice(0, 7),
            { name: file.name, dataUrl: reader.result as string },
          ]);
        }
      };
      reader.readAsDataURL(file);
    }
  }

  async function handleStartThread(event: React.FormEvent) {
    event.preventDefault();
    if (!projectId || !title.trim()) return;
    const chosenHarness = harnessId || harnesses[0]?.id || "inkling";
    setBusy(true);
    setError("");
    try {
      const created = await createWorkThread({
        project_id: projectId,
        title: title.trim(),
        harness: chosenHarness,
        model: modelId || selectedHarness?.default_model || null,
        prompt: prompt.trim() || null,
        images:
          selectedHarness?.supports_images && attachedImages.length > 0
            ? attachedImages.map((img) => img.dataUrl)
            : undefined,
      });
      setAttachedImages([]);
      onChanged();
      onThreadCreated(created.id);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function handleAddProject(event: React.FormEvent) {
    event.preventDefault();
    if (!selectedFolder) return;
    setBusy(true);
    setError("");
    setNotice("");
    try {
      const created = await createWorkProject({
        root_path: selectedFolder,
        title: projectTitle.trim() || null,
      });
      setProjectTitle("");
      setProjectId(created.id);
      setNotice(`Added project "${created.title}".`);
      await refresh();
      onChanged();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function handleDeleteProject(project: WorkProject) {
    setBusy(true);
    setError("");
    try {
      await deleteWorkProject(project.id, true);
      setNotice(`Removed "${project.title}".`);
      await refresh();
      onChanged();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function handleStartSignIn(harness: WorkHarness) {
    setBusy(true);
    setError("");
    setNotice("");
    try {
      const auth = await startHarnessSignIn(harness.id);
      setHarnesses((prev) =>
        prev.map((item) => (item.id === harness.id ? { ...item, auth } : item)),
      );
      if (auth.authorize_url) {
        window.open(auth.authorize_url, "_blank", "noopener,noreferrer");
      }
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function handleSubmitSignIn(harness: WorkHarness) {
    const value = (signInInputs[harness.id] ?? "").trim();
    if (!value) return;
    setBusy(true);
    setError("");
    try {
      const auth = await submitHarnessSignIn(
        harness.id,
        harness.auth.login_method === "oauth_code" ? { code: value } : { credential: value },
      );
      setSignInInputs((prev) => ({ ...prev, [harness.id]: "" }));
      setHarnesses((prev) =>
        prev.map((item) => (item.id === harness.id ? { ...item, auth } : item)),
      );
      if (auth.status === "ready") {
        setNotice(`${harness.display_name} is now connected.`);
      }
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function handleSignOut(harness: WorkHarness) {
    setBusy(true);
    setError("");
    try {
      const auth = await signOutHarness(harness.id);
      setHarnesses((prev) =>
        prev.map((item) => (item.id === harness.id ? { ...item, auth } : item)),
      );
      setNotice(`Signed out of ${harness.display_name}.`);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="work-setup-page">
      <header className="work-setup-head">
        <div>
          <h2>Work</h2>
          <p className="quiet">
            Start coding threads, choose project folders, and connect your coding assistants.
          </p>
        </div>
        <button className="icon" aria-label="Refresh Work setup" onClick={() => void refresh()} disabled={loading}>
          <RefreshCw size={15} className={loading ? "spin" : ""} />
        </button>
      </header>

      <div className="work-tabs" role="tablist" aria-label="Work sections">
        <button
          role="tab"
          aria-selected={tab === "new"}
          className={tab === "new" ? "selected" : ""}
          onClick={() => setTab("new")}
        >
          New thread
        </button>
        <button
          role="tab"
          aria-selected={tab === "projects"}
          className={tab === "projects" ? "selected" : ""}
          onClick={() => setTab("projects")}
        >
          Projects ({projects.length})
        </button>
        <button
          role="tab"
          aria-selected={tab === "harnesses"}
          className={tab === "harnesses" ? "selected" : ""}
          onClick={() => setTab("harnesses")}
        >
          Harnesses ({harnesses.filter((h) => h.auth.status === "ready").length}/{harnesses.length})
        </button>
      </div>

      {error && (
        <div className="banner" role="alert">
          {error}
        </div>
      )}
      {notice && (
        <div className="banner success" role="status">
          {notice}
        </div>
      )}

      {tab === "new" && (
        <section className="work-card">
          {projects.length === 0 ? (
            <div className="work-empty-card">
              <h3>Add a project folder first</h3>
              <p className="quiet">
                Choose a folder on this computer to start your first coding thread.
              </p>
              <button className="primary" onClick={() => setTab("projects")}>
                <Plus size={15} />
                Add a project folder
              </button>
            </div>
          ) : (
            <form className="work-form" onSubmit={handleStartThread}>
              <div className="work-form-row">
                <label>
                  Project folder
                  <select value={projectId} onChange={(e) => setProjectId(e.target.value)} required>
                    {projects.map((p) => (
                      <option key={p.id} value={p.id}>
                        {p.title}
                      </option>
                    ))}
                  </select>
                </label>

                <label>
                  Coding assistant
                  <select
                    value={harnessId}
                    onChange={(e) => {
                      const next = harnesses.find((h) => h.id === e.target.value);
                      setHarnessId(e.target.value);
                      setModelId(next?.default_model ?? "");
                    }}
                  >
                    {harnesses.map((h) => (
                      <option key={h.id} value={h.id}>
                        {h.display_name} — {harnessAuthLabel(h.auth, h.installed)}
                      </option>
                    ))}
                  </select>
                </label>

                {selectedHarness && selectedHarness.models.length > 0 && (
                  <label>
                    Model
                    <select
                      value={modelId || selectedHarness.default_model || ""}
                      onChange={(e) => setModelId(e.target.value)}
                    >
                      {selectedHarness.models.map((m) => (
                        <option key={m.id} value={m.id}>
                          {m.label}
                        </option>
                      ))}
                    </select>
                  </label>
                )}
              </div>

              {selectedHarness &&
                selectedHarness.auth.login_method === "local" &&
                selectedHarness.auth.status !== "ready" && (
                  <p className="work-local-hint" role="status">
                    {selectedHarness.auth.message}
                  </p>
                )}

              <label>
                Task title
                <input
                  type="text"
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  placeholder="For example: Fix sign-in redirect"
                  required
                />
              </label>

              <label>
                What should it work on?
                <textarea
                  rows={4}
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  placeholder="Describe the change, bug fix, or investigation…"
                />
              </label>

              {selectedHarness?.supports_images && (
                <div className="work-image-attach-row">
                  <label className="button work-attach-btn">
                    <ImagePlus size={14} />
                    Attach screenshot
                    <input
                      type="file"
                      accept="image/png,image/jpeg"
                      multiple
                      hidden
                      disabled={busy}
                      onChange={(e) => {
                        handleSelectImages(e.target.files);
                        e.target.value = "";
                      }}
                    />
                  </label>
                  {attachedImages.length > 0 && (
                    <div className="work-image-chips">
                      {attachedImages.map((img, idx) => (
                        <span key={`${img.name}-${idx}`} className="work-image-chip">
                          <span>{img.name}</span>
                          <button
                            type="button"
                            className="icon"
                            aria-label={`Remove ${img.name}`}
                            onClick={() =>
                              setAttachedImages((prev) => prev.filter((_, i) => i !== idx))
                            }
                          >
                            <X size={12} />
                          </button>
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              )}

              <div className="work-form-actions">
                <button type="submit" className="primary" disabled={busy || !title.trim()}>
                  Start work thread
                </button>
              </div>
            </form>
          )}
        </section>
      )}

      {tab === "projects" && (
        <div className="work-stack">
          <section className="work-card">
            <h3>Add a project folder</h3>
            <form className="work-form" onSubmit={handleAddProject}>
              <div className="work-folder-picker">
                <div className="work-folder-bar">
                  <span className="work-folder-current">
                    {folderBrowser?.current_path || "Loading folders…"}
                  </span>
                  {folderBrowser?.parent_path && (
                    <button
                      type="button"
                      className="button"
                      onClick={() => void loadFolder(folderBrowser.parent_path ?? undefined)}
                    >
                      <ChevronUp size={14} />
                      Up
                    </button>
                  )}
                </div>
                <div className="work-folder-list" role="listbox" aria-label="Folders">
                  {(folderBrowser?.entries ?? []).map((entry) => (
                    <button
                      key={entry.path}
                      type="button"
                      className={`work-folder-entry ${selectedFolder === entry.path ? "selected" : ""}`}
                      onClick={() => {
                        setSelectedFolder(entry.path);
                        if (!projectTitle) setProjectTitle(entry.name);
                      }}
                      onDoubleClick={() => void loadFolder(entry.path)}
                    >
                      <Folder size={15} />
                      <span>{entry.name}</span>
                    </button>
                  ))}
                </div>
                <p className="field-hint">
                  Click a folder to select it, or double-click to open it. Selected: {selectedFolder || "None"}
                </p>
              </div>

              <label>
                Project name (optional)
                <input
                  type="text"
                  value={projectTitle}
                  onChange={(e) => setProjectTitle(e.target.value)}
                  placeholder="Uses the folder name if left blank"
                />
              </label>

              <div className="work-form-actions">
                <button type="submit" className="primary" disabled={busy || !selectedFolder}>
                  <Plus size={15} />
                  Add project
                </button>
              </div>
            </form>
          </section>

          {projects.length > 0 && (
            <section className="work-card">
              <h3>Registered projects</h3>
              <ul className="work-project-list">
                {projects.map((project) => (
                  <li key={project.id} className="work-project-row">
                    <div>
                      <strong>{project.title}</strong>
                      <div className="field-hint">{project.root_path}</div>
                    </div>
                    <button
                      className="icon"
                      aria-label={`Remove ${project.title}`}
                      disabled={busy}
                      onClick={() => void handleDeleteProject(project)}
                    >
                      <Trash2 size={15} />
                    </button>
                  </li>
                ))}
              </ul>
            </section>
          )}
        </div>
      )}

      {tab === "harnesses" && (
        <div className="work-harness-grid">
          {harnesses.map((harness) => {
            const isReady = harness.auth.status === "ready";
            const isWaiting =
              harness.auth.status === "starting" ||
              harness.auth.status === "waiting" ||
              harness.auth.status === "verifying";
            const needsInput =
              harness.auth.needs_code ||
              harness.auth.login_method === "api_key" ||
              harness.auth.login_method === "oauth_code";

            return (
              <section key={harness.id} className="work-card work-harness-card">
                <div className="work-harness-head">
                  <div>
                    <h3>{harness.display_name}</h3>
                    <p className="quiet">{harness.description}</p>
                  </div>
                  <span className={`work-pill ${isReady ? "ok" : harness.installed ? "warn" : ""}`}>
                    {harnessAuthLabel(harness.auth, harness.installed)}
                  </span>
                </div>

                <p className="work-harness-message">{harness.auth.message}</p>

                {harness.auth.authorize_url && (
                  <div className="work-auth-box">
                    <a
                      className="button"
                      href={harness.auth.authorize_url}
                      target="_blank"
                      rel="noreferrer"
                    >
                      Open sign-in page
                      <ArrowUpRight size={14} />
                    </a>
                    {harness.auth.user_code && (
                      <div className="work-device-code">
                        <span>One-time code:</span>
                        <code>{harness.auth.user_code}</code>
                      </div>
                    )}
                  </div>
                )}

                {!isReady && harness.installed && needsInput && (isWaiting || harness.auth.login_method === "api_key") && (
                  <div className="work-auth-submit">
                    <input
                      type={harness.auth.login_method === "api_key" ? "password" : "text"}
                      value={signInInputs[harness.id] ?? ""}
                      onChange={(e) =>
                        setSignInInputs((prev) => ({ ...prev, [harness.id]: e.target.value }))
                      }
                      placeholder={
                        harness.auth.login_method === "oauth_code"
                          ? "Paste the sign-in code from your browser"
                          : "Paste your access key"
                      }
                      disabled={busy}
                    />
                    <button
                      className="primary"
                      disabled={busy || !(signInInputs[harness.id] ?? "").trim()}
                      onClick={() => void handleSubmitSignIn(harness)}
                    >
                      <KeyRound size={14} />
                      {harness.auth.login_method === "oauth_code" ? "Finish sign-in" : "Connect"}
                    </button>
                  </div>
                )}

                <div className="work-harness-foot">
                  <span className="field-hint">
                    Models: {harness.models.map((m) => m.label).join(", ")}
                  </span>
                  <div className="work-harness-actions">
                    {harness.auth.login_method === "local" ? (
                      isReady ? (
                        <span className="field-hint">No account or API key needed</span>
                      ) : (
                        <button
                          className="button"
                          disabled={busy || loading}
                          onClick={() => void refresh()}
                        >
                          Check status
                        </button>
                      )
                    ) : isReady ? (
                      <button
                        className="button"
                        disabled={busy}
                        onClick={() => void handleSignOut(harness)}
                      >
                        Sign out
                      </button>
                    ) : (
                      <button
                        className="primary"
                        disabled={busy || !harness.installed}
                        onClick={() => void handleStartSignIn(harness)}
                      >
                        {isWaiting ? "Start over" : `Sign in to ${harness.display_name}`}
                      </button>
                    )}
                  </div>
                </div>
              </section>
            );
          })}
        </div>
      )}
    </div>
  );
}
