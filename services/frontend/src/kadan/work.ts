import { api } from "./api";

export type WorkBucket = "default" | "working" | "snoozed" | "settled";
export type WorkRunStatus =
  | "idle"
  | "preparing"
  | "queued"
  | "running"
  | "waiting"
  | "completed"
  | "failed"
  | "interrupted"
  | "cancelled";
export type WorkWorkspaceMode = "root" | "worktree";

export interface WorkProject {
  id: string;
  title: string;
  root_path: string;
  default_branch: string;
  setup_script: string | null;
  version: number;
  created_at: string;
  updated_at: string;
}

export interface WorkPendingRequest {
  id: string;
  kind: "question" | "approval";
  title: string;
  prompt: string;
  options: Record<string, unknown>[];
  tool_name: string | null;
  arguments: Record<string, unknown> | null;
  status: "pending" | "answered" | "cancelled";
  response: Record<string, unknown> | null;
  created_at: string;
  resolved_at: string | null;
}

export interface WorkThread {
  id: string;
  client_request_id: string | null;
  project_id: string;
  title: string;
  harness: string;
  model: string | null;
  model_options: Record<string, unknown>;
  workspace_mode: WorkWorkspaceMode;
  worktree_path: string | null;
  branch: string | null;
  base_ref: string | null;
  run_status: WorkRunStatus;
  bucket: WorkBucket;
  snoozed_until: string | null;
  origin_agent_key: string | null;
  origin_conversation_id: string | null;
  origin_run_id: string | null;
  harness_session_id: string | null;
  pending_requests: WorkPendingRequest[];
  error: string | null;
  version: number;
  created_at: string;
  updated_at: string;
}

export interface WorkRun {
  id: string;
  thread_id: string;
  ordinal: number;
  mode: "send" | "steer" | "queue";
  prompt: string;
  status: WorkRunStatus;
  error: string | null;
  final_message: string | null;
  version: number;
  created_at: string;
  updated_at: string;
}

export interface WorkActivityItem {
  id: string;
  thread_id: string;
  position: number;
  run_id: string | null;
  kind:
    | "user_message"
    | "assistant_message"
    | "reasoning"
    | "tool_call"
    | "tool_result"
    | "file_change"
    | "pending_request"
    | "status"
    | "error";
  text: string;
  data: Record<string, unknown>;
  created_at: string;
}

export interface WorkThreadDetail {
  thread: WorkThread;
  project: WorkProject | null;
  runs: WorkRun[];
  activity: WorkActivityItem[];
}

export interface WorkHarnessModel {
  id: string;
  label: string;
  description: string;
  options: Record<string, string[]>;
}

export interface WorkHarnessAuth {
  status: "ready" | "sign_in_required" | "starting" | "waiting" | "verifying" | "unavailable" | "error";
  login_method: "device_code" | "oauth_code" | "api_key" | "browser" | "local" | "none";
  authorize_url: string | null;
  user_code: string | null;
  needs_code: boolean;
  account_label: string | null;
  message: string;
}

export interface WorkHarness {
  id: string;
  display_name: string;
  description: string;
  installed: boolean;
  auth: WorkHarnessAuth;
  models: WorkHarnessModel[];
  default_model: string | null;
  supports_steer: boolean;
  supports_resume: boolean;
  supports_approvals: boolean;
  supports_questions: boolean;
}

export interface FilesystemBrowseResponse {
  current_path: string;
  parent_path: string | null;
  is_git_repo: boolean;
  default_branch: string | null;
  entries: { name: string; path: string; is_git_repo: boolean }[];
}

export interface WorkBuckets {
  default: WorkThread[];
  working: WorkThread[];
  snoozed: WorkThread[];
  settled: WorkThread[];
}

export function groupThreadsByBucket(threads: WorkThread[], settledQuery = ""): WorkBuckets {
  const buckets: WorkBuckets = { default: [], working: [], snoozed: [], settled: [] };
  const needle = settledQuery.trim().toLowerCase();
  for (const thread of threads) {
    if (thread.bucket === "settled") {
      if (!needle || thread.title.toLowerCase().includes(needle) || (thread.branch ?? "").toLowerCase().includes(needle)) {
        buckets.settled.push(thread);
      }
    } else if (thread.bucket === "working") {
      buckets.working.push(thread);
    } else if (thread.bucket === "snoozed") {
      buckets.snoozed.push(thread);
    } else {
      buckets.default.push(thread);
    }
  }
  return buckets;
}

export function isThreadActive(status: WorkRunStatus): boolean {
  return status === "preparing" || status === "queued" || status === "running" || status === "waiting";
}

export function pendingRequestsFor(thread: WorkThread): WorkPendingRequest[] {
  return thread.pending_requests.filter((req) => req.status === "pending");
}

export function workStatusLabel(status: WorkRunStatus): string {
  switch (status) {
    case "preparing":
      return "Preparing";
    case "queued":
      return "Queued";
    case "running":
      return "Working";
    case "waiting":
      return "Needs input";
    case "completed":
      return "Done";
    case "failed":
      return "Failed";
    case "interrupted":
      return "Stopped";
    case "cancelled":
      return "Cancelled";
    default:
      return "Ready";
  }
}

export function harnessAuthLabel(auth: WorkHarnessAuth, installed: boolean): string {
  if (!installed || auth.status === "unavailable") return "Not installed";
  if (auth.status === "ready") return "Connected";
  if (auth.status === "starting" || auth.status === "waiting" || auth.status === "verifying") {
    return "Signing in…";
  }
  if (auth.status === "error") return "Sign-in problem";
  return "Sign in needed";
}

export function formatSnoozeLabel(iso: string | null): string {
  if (!iso) return "";
  const target = new Date(iso);
  if (Number.isNaN(target.getTime())) return "";
  const diffMs = target.getTime() - Date.now();
  if (diffMs <= 0) return "Waking soon";
  const mins = Math.round(diffMs / 60000);
  if (mins < 60) return `In ${mins}m`;
  const hrs = Math.round(mins / 60);
  if (hrs < 24) return `In ${hrs}h`;
  return target.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

export function parseWorkHash(hash: string): string | null {
  const match = /^#work\/([0-9a-fA-F-]{8,64})$/.exec(hash.trim());
  return match ? match[1] : null;
}

export const listWorkProjects = () => api<WorkProject[]>("/work/projects");
export const createWorkProject = (body: {
  title?: string | null;
  root_path?: string | null;
  clone_url?: string | null;
  default_branch?: string | null;
  setup_script?: string | null;
}) => api<WorkProject>("/work/projects", body, "POST");
export const updateWorkProject = (
  id: string,
  body: { title?: string; default_branch?: string; setup_script?: string | null; expected_version: number },
) => api<WorkProject>(`/work/projects/${encodeURIComponent(id)}`, body, "PUT");
export const deleteWorkProject = (id: string, force = false) =>
  api<void>(`/work/projects/${encodeURIComponent(id)}${force ? "?force=true" : ""}`, undefined, "DELETE");

export const browseWorkFilesystem = (path?: string) =>
  api<FilesystemBrowseResponse>(
    `/work/filesystem/browse${path ? `?path=${encodeURIComponent(path)}` : ""}`,
  );

export const listWorkHarnesses = () => api<WorkHarness[]>("/work/harnesses");
export const startHarnessSignIn = (harnessId: string) =>
  api<WorkHarnessAuth>(`/work/harnesses/${encodeURIComponent(harnessId)}/sign-in`, {}, "POST");
export const getHarnessSignIn = (harnessId: string) =>
  api<WorkHarnessAuth>(`/work/harnesses/${encodeURIComponent(harnessId)}/sign-in`);
export const submitHarnessSignIn = (
  harnessId: string,
  body: { code?: string; credential?: string; account_label?: string },
) => api<WorkHarnessAuth>(`/work/harnesses/${encodeURIComponent(harnessId)}/sign-in/submit`, body, "POST");
export const signOutHarness = (harnessId: string) =>
  api<WorkHarnessAuth>(`/work/harnesses/${encodeURIComponent(harnessId)}/sign-in`, undefined, "DELETE");

export const listWorkThreads = () => api<WorkThread[]>("/work/threads");
export const createWorkThread = (body: {
  project_id: string;
  title: string;
  harness: string;
  model?: string | null;
  workspace_mode: WorkWorkspaceMode;
  branch?: string | null;
  base_ref?: string | null;
  prompt?: string | null;
}) => api<WorkThread>("/work/threads", body, "POST");
export const getWorkThreadDetail = (id: string) =>
  api<WorkThreadDetail>(`/work/threads/${encodeURIComponent(id)}`);
export const updateWorkThread = (
  id: string,
  body: { title?: string; harness?: string; model?: string | null; expected_version: number },
) => api<WorkThread>(`/work/threads/${encodeURIComponent(id)}`, body, "PUT");
export const deleteWorkThread = (id: string) =>
  api<void>(`/work/threads/${encodeURIComponent(id)}`, undefined, "DELETE");
export const organizeWorkThread = (
  id: string,
  body: { action: "default" | "snooze" | "settle" | "unsettle"; snooze_minutes?: number; snoozed_until?: string },
) => api<WorkThread>(`/work/threads/${encodeURIComponent(id)}/organize`, body, "POST");
export const sendWorkMessage = (id: string, body: { prompt: string; mode: "send" | "steer" | "queue" }) =>
  api<WorkRun>(`/work/threads/${encodeURIComponent(id)}/runs`, body, "POST");
export const interruptWorkThread = (id: string) =>
  api<WorkThread>(`/work/threads/${encodeURIComponent(id)}/interrupt`, {}, "POST");
export const respondWorkPendingRequest = (
  threadId: string,
  requestId: string,
  response: Record<string, unknown>,
) =>
  api<WorkThread>(
    `/work/threads/${encodeURIComponent(threadId)}/pending-requests/${encodeURIComponent(requestId)}/respond`,
    { response },
    "POST",
  );
