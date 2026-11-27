import { useCallback, useEffect, useState } from "react";
import {
  ArrowUpRight,
  Folder,
  KeyRound,
  Plus,
  RefreshCw,
  Server,
  Trash2,
} from "lucide-react";
import FolderPickerDialog from "./FolderPickerDialog";
import { pickDesktopFolder } from "./desktop";
import type { Status } from "./types";
import {
  createWorkProject,
  deleteWorkProject,
  getHarnessSignIn,
  harnessAuthLabel,
  listWorkHarnesses,
  listWorkProjects,
  signOutHarness,
  startHarnessSignIn,
  submitHarnessSignIn,
  type WorkHarness,
  type WorkProject,
} from "./work";

interface Props {
  desktopMode: boolean;
  desktopHost?: string;
  status: Status | null;
  onOpenServerSettings?: () => void;
  onChanged: () => void;
}

export default function SettingsPage({
  desktopMode,
  desktopHost,
  status,
  onOpenServerSettings,
  onChanged,
}: Props) {
  const [harnesses, setHarnesses] = useState<WorkHarness[]>([]);
  const [projects, setProjects] = useState<WorkProject[]>([]);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [expandedSignIn, setExpandedSignIn] = useState<string | null>(null);
  const [signInInputs, setSignInInputs] = useState<Record<string, string>>({});
  const [folderDialogOpen, setFolderDialogOpen] = useState(false);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const [harnessList, projList] = await Promise.all([
        listWorkHarnesses(),
        listWorkProjects(),
      ]);
      setHarnesses(harnessList);
      setProjects(projList);
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

  useEffect(() => {
    const waiting = harnesses.filter(
      (h) =>
        h.auth.status === "starting" ||
        h.auth.status === "waiting" ||
        h.auth.status === "verifying",
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
              setExpandedSignIn((cur) => (cur === h.id ? null : cur));
            }
          })
          .catch(() => {});
      }
    }, 2500);
    return () => clearInterval(timer);
  }, [harnesses]);

  async function addFolderPath(path: string) {
    setBusy(true);
    setError("");
    try {
      await createWorkProject({ root_path: path });
      await refresh();
      onChanged();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function handleAddFolder() {
    setError("");
    if (desktopMode) {
      const picked = await pickDesktopFolder();
      if (picked.supported) {
        if (picked.path) await addFolderPath(picked.path);
        return;
      }
    }
    setFolderDialogOpen(true);
  }

  async function handleDeleteProject(project: WorkProject) {
    setBusy(true);
    setError("");
    try {
      await deleteWorkProject(project.id, true);
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
    setExpandedSignIn(harness.id);
    try {
      const auth = await startHarnessSignIn(harness.id);
      setHarnesses((prev) =>
        prev.map((item) => (item.id === harness.id ? { ...item, auth } : item)),
      );
      if (auth.authorize_url && auth.login_method !== "api_key") {
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
        setExpandedSignIn(null);
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
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  const serverDot = !status
    ? ""
    : status.tsuchi.state !== "ready" || status.hikari.state === "unavailable"
      ? "down"
      : status.hikari.state !== "ready"
        ? "warn"
        : "ready";

  return (
    <section className="page settings-page" aria-labelledby="settings-title">
      <div className="page-inner narrow">
        <header className="settings-head">
          <h1 id="settings-title">Settings</h1>
          <button
            className="icon"
            aria-label="Refresh settings"
            title="Refresh"
            onClick={() => void refresh()}
            disabled={loading}
          >
            <RefreshCw size={15} className={loading ? "spin" : ""} />
          </button>
        </header>

        {error && (
          <div className="banner" role="alert">
            {error}
          </div>
        )}

        <section className="page-card settings-card" aria-labelledby="assistants-title">
          <header className="settings-section-head">
            <h2 id="assistants-title">Coding assistants</h2>
          </header>

          <ul className="settings-list">
            {harnesses.map((harness) => {
              const isReady = harness.auth.status === "ready";
              const isLocal = harness.auth.login_method === "local";
              const isWaiting =
                harness.auth.status === "starting" ||
                harness.auth.status === "waiting" ||
                harness.auth.status === "verifying";
              const showSignInFlow =
                !isReady &&
                !isLocal &&
                harness.installed &&
                (expandedSignIn === harness.id || isWaiting);
              const needsInput =
                harness.auth.needs_code ||
                harness.auth.login_method === "api_key" ||
                harness.auth.login_method === "oauth_code";

              return (
                <li key={harness.id} className="settings-list-item">
                  <div className="settings-row">
                    <div className="settings-row-main">
                      <strong>{harness.display_name}</strong>
                      <span
                        className={`work-pill ${isReady ? "ok" : harness.installed ? "warn" : ""}`}
                      >
                        {harnessAuthLabel(harness.auth, harness.installed)}
                      </span>
                    </div>

                    <div className="settings-row-actions">
                      {isLocal ? (
                        !isReady && (
                          <button
                            className="button"
                            disabled={busy || loading}
                            onClick={() => void refresh()}
                          >
                            Check
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
                          className="button"
                          disabled={busy || !harness.installed}
                          onClick={() => void handleStartSignIn(harness)}
                        >
                          {isWaiting ? "Retry" : "Sign in"}
                        </button>
                      )}
                    </div>
                  </div>

                  {isLocal && !isReady && harness.auth.message && (
                    <p className="field-hint settings-row-note">{harness.auth.message}</p>
                  )}

                  {showSignInFlow && (
                    <div className="settings-signin-inline">
                      {harness.auth.authorize_url && (
                        <div className="work-auth-box">
                          <a
                            className="button"
                            href={harness.auth.authorize_url}
                            target="_blank"
                            rel="noreferrer"
                          >
                            Open sign-in
                            <ArrowUpRight size={14} />
                          </a>
                          {harness.auth.user_code && (
                            <div className="work-device-code">
                              <span>Code:</span>
                              <code>{harness.auth.user_code}</code>
                            </div>
                          )}
                        </div>
                      )}
                      {needsInput && (
                        <div className="work-auth-submit">
                          <input
                            type={harness.auth.login_method === "api_key" ? "password" : "text"}
                            value={signInInputs[harness.id] ?? ""}
                            onChange={(e) =>
                              setSignInInputs((prev) => ({
                                ...prev,
                                [harness.id]: e.target.value,
                              }))
                            }
                            placeholder={
                              harness.auth.login_method === "oauth_code"
                                ? "Paste sign-in code"
                                : "Paste access key"
                            }
                            disabled={busy}
                          />
                          <button
                            className="primary"
                            disabled={busy || !(signInInputs[harness.id] ?? "").trim()}
                            onClick={() => void handleSubmitSignIn(harness)}
                          >
                            <KeyRound size={14} />
                            Connect
                          </button>
                        </div>
                      )}
                    </div>
                  )}
                </li>
              );
            })}
          </ul>
        </section>

        <section className="page-card settings-card" aria-labelledby="projects-title">
          <header className="settings-section-head">
            <h2 id="projects-title">Projects</h2>
            <button className="button" disabled={busy} onClick={() => void handleAddFolder()}>
              <Plus size={14} />
              Add folder
            </button>
          </header>

          {projects.length === 0 ? (
            <p className="field-hint">No project folders added yet.</p>
          ) : (
            <ul className="settings-list">
              {projects.map((project) => (
                <li key={project.id} className="settings-list-item">
                  <div className="settings-row">
                    <div className="settings-project-info">
                      <Folder size={15} />
                      <div className="settings-project-text">
                        <strong>{project.title}</strong>
                        <span className="field-hint">{project.root_path}</span>
                      </div>
                    </div>
                    <button
                      className="icon"
                      aria-label={`Remove ${project.title}`}
                      disabled={busy}
                      onClick={() => void handleDeleteProject(project)}
                    >
                      <Trash2 size={14} />
                    </button>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </section>

        {desktopMode && onOpenServerSettings && (
          <section className="page-card settings-card" aria-labelledby="server-title">
            <header className="settings-section-head">
              <h2 id="server-title">Kadan Server</h2>
            </header>
            <div className="settings-row">
              <div className="settings-row-main">
                <span className={`dot ${serverDot}`} aria-hidden="true" />
                <code>{desktopHost || "Not configured"}</code>
              </div>
              <button className="button" onClick={onOpenServerSettings}>
                <Server size={14} />
                Change server
              </button>
            </div>
          </section>
        )}
      </div>

      {folderDialogOpen && (
        <FolderPickerDialog
          onSelect={(path) => {
            setFolderDialogOpen(false);
            void addFolderPath(path);
          }}
          onClose={() => setFolderDialogOpen(false)}
        />
      )}
    </section>
  );
}
