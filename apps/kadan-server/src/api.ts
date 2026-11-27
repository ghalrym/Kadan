import { invoke } from "@tauri-apps/api/core";
import { open as openDialog } from "@tauri-apps/plugin-dialog";
import { open as openShell } from "@tauri-apps/plugin-shell";
import type {
  ApplyUpdateResult,
  HardwareSummary,
  LogEntry,
  ServerConfig,
  ServerStatusSnapshot,
  UpdateCheckResult,
  UpdateStatus,
} from "./types";

export const getConfig = () => invoke<ServerConfig>("get_config");

export const updateConfig = (config: ServerConfig, restart: boolean) =>
  invoke<ServerConfig>("update_config", { config, restart });

export const getStatus = () => invoke<ServerStatusSnapshot>("get_status");

export const getHardware = () => invoke<HardwareSummary>("get_hardware");

export const getLogs = (service?: string) =>
  invoke<LogEntry[]>("get_logs", { service: service ?? null });

export const startServer = () => invoke<ServerStatusSnapshot>("start_server");

export const stopServer = () => invoke<ServerStatusSnapshot>("stop_server");

export const restartService = (service: string) =>
  invoke<ServerStatusSnapshot>("restart_service", { service });

export async function pickFolder(defaultPath?: string): Promise<string | null> {
  const result = await openDialog({
    directory: true,
    multiple: false,
    defaultPath,
    title: "Choose folder for Kadan Server data and models",
  });
  return typeof result === "string" ? result : null;
}

export const openExternalUrl = (url: string) => openShell(url);

export const checkUpdates = () => invoke<UpdateCheckResult>("check_updates");

export const getUpdateStatus = () => invoke<UpdateStatus>("get_update_status");

export const installUpdate = () => invoke<ApplyUpdateResult>("install_update");
