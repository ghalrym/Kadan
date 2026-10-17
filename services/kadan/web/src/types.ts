export interface Agent {
  key: string;
  name: string;
  created_at: string;
}
export interface Message {
  id: string;
  role: "system" | "user" | "assistant" | "tool";
  content: string;
  created_at: string;
  resource_ids: string[];
  attachment_ids: string[];
  sources: { label: string; url: string }[];
}
export interface Resource {
  id: string;
  agent_key: string;
  kind: "persona" | "memory" | "space" | "deck" | "conversation";
  title: string;
  markdown: string;
  parent_id: string | null;
  tags: string[];
  messages: Message[];
  attachment_ids: string[];
  version: number;
  created_at: string;
  updated_at: string;
  deleted_at: string | null;
}
export interface Asset {
  id: string;
  title: string;
  filename: string;
  media_type: string;
  size_bytes: number;
  is_image: boolean;
  tags: string[];
  updated_at: string;
}
export interface Status {
  tsuchi: {
    state: "ready" | "unavailable";
    message: string;
    browser_url: string | null;
  };
  hikari: {
    state: "ready" | "tier_required" | "model_required" | "unavailable";
    message: string;
    tier: string | null;
    model_id: string | null;
    settings_url: string | null;
  };
}
export interface Plugin {
  id: string;
  name: string;
  description: string;
  enabled: boolean;
  skills: { name: string; description: string; instructions: string }[];
  setup: { instructions: string };
}
export interface ToolExecution {
  id: string;
  name: string;
  arguments: Record<string, unknown>;
  status: "started" | "completed" | "error";
  result?: Record<string, unknown> | null;
}
export type ChatEvent =
  | { event: "user_saved"; data: { conversation: Resource } }
  | { event: "status"; data: { message: string } }
  | {
      event: "tool_started" | "tool_finished";
      data: { conversation: Resource; execution: ToolExecution };
    }
  | { event: "delta"; data: { content: string } }
  | {
      event: "done";
      data: { conversation: Resource; finish_reason: "stop" | "length" };
    }
  | {
      event: "error";
      data: {
        code: string;
        message: string;
        status: number;
        user_saved: boolean;
      };
    };

export interface PetManifest {
  schema_version: 1;
  sprite_version: 2;
  atlas_asset_id: string;
  base_asset_id: string;
  cell_width: 192;
  cell_height: 208;
  columns: 8;
  rows: 11;
  pixelated: boolean;
  animations: { name: import("./pet").PetState; row: number; durations: number[] }[];
}
export interface PetJob {
  id: string;
  agent_key: string;
  prompt: string;
  status: "queued" | "running" | "cancelling" | "cancelled" | "interrupted" | "failed" | "ready";
  stage: string;
  completed_steps: number;
  total_steps: number;
  version: number;
  created_at: string;
  updated_at: string;
  error: string | null;
  manifest: PetManifest | null;
}
