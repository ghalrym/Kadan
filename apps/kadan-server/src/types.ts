export interface ServerConfig {
  bind_all_interfaces: boolean;
  server_port: number;
  worker_token: string;
  data_dir: string;
  auto_start_server: boolean;
  launch_at_login: boolean;
}

export interface GpuInfo {
  name: string;
  total_vram_mb: number;
  free_vram_mb: number;
}

export interface HardwareSummary {
  hostname: string;
  lan_ip: string | null;
  cpu_cores: number;
  total_ram_mb: number;
  available_ram_mb: number;
  gpus: GpuInfo[];
}

export type ServicePhase = "stopped" | "starting" | "running" | "error";

export interface ServiceStatus {
  id: string;
  label: string;
  phase: ServicePhase;
  message: string;
  url: string | null;
}

export interface ServerStatusSnapshot {
  running: boolean;
  services: ServiceStatus[];
  server_local_url: string;
  hikari_local_url: string;
  tsuchi_local_url: string;
  worker_binary_found: boolean;
  python_runtime_found: boolean;
}

export interface LogEntry {
  timestamp: number;
  service: string;
  line: string;
}
