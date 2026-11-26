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
