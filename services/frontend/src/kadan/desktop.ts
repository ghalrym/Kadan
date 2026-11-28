export interface SavedServer {
  id: string;
  label: string;
  host: string;
}

/** "host": Kadan runs its services on this computer. "connect": it uses another computer's. */
export type RunMode = "host" | "connect";

export interface DesktopServerConfig {
  mode?: RunMode | null;
  /** The other computer to use in "connect" mode. */
  active_host: string;
  saved_servers: SavedServer[];
}

export interface ServerTestResult {
  ok: boolean;
  message: string;
  version?: string | null;
  app?: string | null;
  latency_ms?: number | null;
}

declare global {
  interface Window {
    __KADAN_DESKTOP__?: boolean;
  }
}

export function isDesktopApp(): boolean {
  return typeof window !== "undefined" && Boolean(window.__KADAN_DESKTOP__);
}

export async function getDesktopServerConfig(): Promise<DesktopServerConfig> {
  const response = await fetch("/__kadan_desktop__/config", { cache: "no-store" });
  if (!response.ok) {
    throw new Error("Could not read Kadan Desktop server settings.");
  }
  return (await response.json()) as DesktopServerConfig;
}

export async function saveDesktopServerConfig(
  config: DesktopServerConfig,
): Promise<DesktopServerConfig> {
  const response = await fetch("/__kadan_desktop__/config", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config),
  });
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(body?.detail || "Could not save server settings.");
  }
  return (await response.json()) as DesktopServerConfig;
}

export interface DesktopUpdateCheck {
  current_version: string;
  latest_version: string | null;
  update_available: boolean;
  release_tag: string | null;
  release_url: string | null;
  can_self_update: boolean;
  install_kind: string;
  reason: string | null;
}

export interface DesktopInstallResult {
  ok: boolean;
  target_version: string | null;
  restarting: boolean;
  message: string;
}

export interface PendingDesktopUpdate {
  request_id: string;
  target_version: string;
  requested_at: number;
}

export interface ServerUpdateStatus {
  state: string;
  current_version: string;
  staged_version: string | null;
  message: string | null;
  pending_desktop_update: PendingDesktopUpdate | null;
}

export async function getDesktopUpdateStatus(): Promise<DesktopUpdateCheck> {
  const response = await fetch("/__kadan_desktop__/updates", { cache: "no-store" });
  if (!response.ok) {
    throw new Error("Could not check Kadan Desktop updates.");
  }
  return (await response.json()) as DesktopUpdateCheck;
}

export async function installDesktopUpdate(): Promise<DesktopInstallResult> {
  const response = await fetch("/__kadan_desktop__/updates/install", {
    method: "POST",
  });
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(body?.detail || "Could not install Kadan Desktop update.");
  }
  return (await response.json()) as DesktopInstallResult;
}

export interface DesktopFolderPickResult {
  path: string | null;
  cancelled: boolean;
  supported: boolean;
}

export interface DesktopSaveDownloadResult {
  saved: boolean;
  cancelled: boolean;
  path: string | null;
  filename: string | null;
  supported: boolean;
}

export async function pickDesktopFolder(): Promise<DesktopFolderPickResult> {
  if (!isDesktopApp()) {
    return { path: null, cancelled: false, supported: false };
  }
  const response = await fetch("/__kadan_desktop__/pick-folder", { method: "POST" });
  if (!response.ok) {
    return { path: null, cancelled: false, supported: false };
  }
  return (await response.json()) as DesktopFolderPickResult;
}

export async function saveDesktopDownload(
  url: string,
  filename?: string,
): Promise<DesktopSaveDownloadResult> {
  if (!isDesktopApp()) {
    return { saved: false, cancelled: false, path: null, filename: null, supported: false };
  }
  const response = await fetch("/__kadan_desktop__/save-download", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url, filename }),
  });
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(body?.detail || "Could not save file.");
  }
  return (await response.json()) as DesktopSaveDownloadResult;
}

export async function testDesktopServerConnection(
  host: string,
): Promise<ServerTestResult> {
  const response = await fetch("/__kadan_desktop__/test", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ host }),
  });
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    return {
      ok: false,
      message: body?.detail || "Could not reach that Kadan Server.",
    };
  }
  return (await response.json()) as ServerTestResult;
}

/** Kadan's services on this computer, when the app hosts them. */
export interface HostService {
  id: "api" | "runner" | "worker";
  label: string;
  phase: "stopped" | "starting" | "running" | "error";
  message: string;
}

export interface HostSettings {
  bind_all_interfaces: boolean;
  server_port: number;
  worker_token: string;
  data_dir: string;
  auto_start_server: boolean;
  launch_at_login: boolean;
}

export interface HostStatus {
  mode: RunMode;
  status: {
    running: boolean;
    services: HostService[];
    server_local_url: string;
    worker_binary_found: boolean;
    python_runtime_found: boolean;
  };
  settings: HostSettings;
  hardware: {
    hostname: string;
    lan_ip: string | null;
    cpu_cores: number;
    total_ram_mb: number;
    available_ram_mb: number;
    gpus: { name: string; total_vram_mb: number; free_vram_mb: number }[];
  };
}

export interface HostLogEntry {
  timestamp: number;
  service: string;
  line: string;
}

async function hostRequest<T>(path: string, method = "GET", body?: unknown): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`/__kadan_desktop__/host${path}`, {
      method,
      cache: "no-store",
      headers: body === undefined ? undefined : { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch {
    throw new Error("Kadan’s app isn’t responding. Close and reopen Kadan.");
  }
  if (!response.ok) {
    const detail = await response.json().catch(() => null);
    throw new Error(detail?.detail || "Kadan couldn’t do that. Try again.");
  }
  return (await response.json()) as T;
}

export const getHostStatus = () => hostRequest<HostStatus>("");
export const startHost = () => hostRequest<HostStatus["status"]>("/start", "POST");
export const stopHost = () => hostRequest<HostStatus["status"]>("/stop", "POST");
export const restartHostService = (service: HostService["id"]) =>
  hostRequest<HostStatus["status"]>("/restart", "POST", { service });
export const getHostLogs = (service?: string) =>
  hostRequest<HostLogEntry[]>(`/logs${service ? `?service=${encodeURIComponent(service)}` : ""}`);
export const saveHostSettings = (settings: HostSettings, restart: boolean) =>
  hostRequest<HostSettings>("/settings", "PUT", { settings, restart });
