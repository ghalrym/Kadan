export interface SavedServer {
  id: string;
  label: string;
  host: string;
}

export interface DesktopServerConfig {
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
