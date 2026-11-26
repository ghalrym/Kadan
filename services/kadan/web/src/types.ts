export interface Agent {
  key: string;
  name: string;
  created_at: string;
}
export interface Runner {
  runner_id: string;
  worker_id: string;
  capacity: number;
  lease_expires_at: string;
  agent_keys: string[];
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
  kind: "persona" | "memory" | "space" | "deck" | "conversation" | "folder";
  title: string;
  markdown: string;
  parent_id: string | null;
  tags: string[];
  sources: { label: string; url: string }[];
  messages: Message[];
  attachment_ids: string[];
  version: number;
  created_at: string;
  updated_at: string;
  deleted_at: string | null;
}
/** A chat listed in the sidebar, without its messages. */
export interface ConversationSummary {
  id: string;
  agent_key: string;
  title: string;
  message_count: number;
  preview: string;
  version: number;
  created_at: string;
  updated_at: string;
}
/** Something saved in Space: a folder, page, slide deck, image or file. */
export interface SpaceItem {
  id: string;
  type: "folder" | "page" | "slides" | "image" | "file";
  title: string;
  parent_id: string | null;
  pinned: boolean;
  filename: string | null;
  media_type: string | null;
  size_bytes: number | null;
  version: number;
  created_at: string;
  updated_at: string;
  deleted_at: string | null;
}
export interface SpacePageContent extends SpaceItem {
  markdown: string;
}
export interface Slide {
  id: string;
  layout: "title" | "text" | "split" | "image";
  title: string;
  body: string;
  notes: string;
  image_asset_id: string | null;
}
export interface SpaceSlides extends SpaceItem {
  slides: Slide[];
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
    state: "ready" | "model_required" | "unavailable";
    message: string;
    model_id: string | null;
    download_state?: "not_downloaded" | "downloading" | "cancelling" | "cancelled" | "failed" | "complete" | null;
    /** Whether the selected language model is resident in Hikari; null when unknown. */
    model_state?: "unloaded" | "loading" | "ready" | "offloaded" | "unloading" | "error" | null;
    settings_url: string | null;
  };
}
export interface AgentRun {
  id: string;
  conversation_id: string;
  trigger_message_id: string | null;
  trigger_job_id: string | null;
  delivered_message_ids: string[];
  draft?: string | null;
  status: "queued" | "running" | "completed" | "failed" | "cancelled";
  error: string | null;
  created_at: string;
  updated_at: string;
}
export interface ImageJob {
  id: string;
  conversation_id: string;
  prompt: string;
  status: "queued" | "running" | "completed" | "failed" | "cancelling" | "cancelled";
  error: string | null;
  delivered_message_id: string | null;
  created_at: string;
  updated_at: string;
}
export interface RunSnapshot {
  runs: AgentRun[];
  images: ImageJob[];
  conversation: Resource;
}

export interface PetManifest {
  schema_version: 1;
  sprite_version: 1 | 2;
  atlas_asset_id: string;
  /** Null for imported pets, which have no generated base artwork. */
  base_asset_id: string | null;
  cell_width: 192;
  cell_height: 208;
  columns: 8;
  rows: 9 | 11;
  pixelated: boolean;
  animations: { name: import("./pet").PetState; row: number; durations: number[] }[];
}
export interface PetStep {
  name: string;
  method: "generated" | "canonical";
  source_asset_id: string | null;
  output_asset_id: string;
  raw_asset_id: string;
  provenance_asset_id: string;
  prompt: string;
  request_id: string;
  model_id: string;
  seed: number | null;
  strength: number | null;
  meta: string;
}
export interface PetJob {
  id: string;
  agent_key: string;
  prompt: string;
  origin: "generated" | "imported";
  reference_mode: "single_image" | "character_sheet";
  status: "queued" | "running" | "cancelling" | "cancelled" | "interrupted" | "failed" | "ready";
  stage: string;
  completed_steps: number;
  total_steps: 74 | 75;
  steps: PetStep[];
  rejections: { step: PetStep; reason: string }[];
  parent_pet_id?: string | null;
  pending_step_name: string | null;
  pending_request_id: string | null;
  pending_seed: number | null;
  version: number;
  created_at: string;
  updated_at: string;
  error: string | null;
  manifest: PetManifest | null;
}
