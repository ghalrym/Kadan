import type { CapabilityModel, CheckpointStatus } from "../hikari/api/settings";
import { downloadState } from "../hikari/api/settings";

/** What each AI model lets Kadan do, in the words a person would use. */
export const ABILITY_TEXT: Record<string, { name: string; description: string }> = {
  "inkling-2-mini": { name: "Chat & thinking", description: "Writes replies, works on coding threads and makes decisions." },
  "flux.2-klein-4b": { name: "Images", description: "Creates and edits pictures, including pets." },
  "qwen3-asr-1.7b": { name: "Dictation", description: "Turns what you say into text." },
  "qwen-tts-1.7b-custom": { name: "Reading aloud", description: "Speaks text in a natural voice." },
};

export type AbilityState = "ready" | "needs_download" | "waiting" | "downloading" | "failed" | "not_running";

export interface Ability {
  id: string;
  name: string;
  description: string;
  state: AbilityState;
  /** Download progress from 0 to 100 while downloading. */
  progress: number;
  /** The model's own name, shown small for people who want to know. */
  modelName: string;
  sizeBytes: number;
  error: string | null;
  model: CheckpointStatus | null;
}

/** The catalog marks some checkpoints experimental; here they are the defaults. */
export const modelName = (model: CheckpointStatus | null) =>
  model ? (model.display_name ?? model.id).replace(/ \(experimental\)$/, "") : "";

export const modelSize = (model: CheckpointStatus | null) => (model ? model.total_bytes || model.estimated_bytes : 0);

export function toAbility({ capability, model }: CapabilityModel): Ability {
  const text = ABILITY_TEXT[capability.id] ?? { name: capability.label, description: "" };
  const download = downloadState(model);
  const size = modelSize(model);
  const state: AbilityState =
    download === "complete" ? "ready"
      : download === "not_downloaded" ? "needs_download"
        : download === "queued" ? "waiting"
          : download === "downloading" ? "downloading"
            : download === "failed" ? "failed"
              : "not_running";
  return {
    id: capability.id,
    ...text,
    state,
    progress: model && size && download === "downloading" ? Math.min(100, Math.floor((model.downloaded_bytes / size) * 100)) : 0,
    modelName: modelName(model),
    sizeBytes: size,
    error: model?.error ?? null,
    model,
  };
}

export const STATE_LABEL: Record<AbilityState, string> = {
  ready: "Ready",
  needs_download: "Needs download",
  waiting: "Waiting to download",
  downloading: "Downloading",
  failed: "Download stopped",
  not_running: "Not running",
};

export const STATE_TONE: Record<AbilityState, "ok" | "warn" | "busy" | "error" | "muted"> = {
  ready: "ok",
  needs_download: "warn",
  waiting: "busy",
  downloading: "busy",
  failed: "error",
  not_running: "error",
};

/** One sentence about the whole server, for the top of Home. */
export function overallStatus(abilities: Ability[], storageReady: boolean): { tone: "ok" | "warn" | "busy" | "error"; title: string; detail: string } {
  const ready = abilities.filter((ability) => ability.state === "ready").length;
  const busy = abilities.some((ability) => ability.state === "downloading" || ability.state === "waiting");
  if (!storageReady) {
    return { tone: "error", title: "Kadan can’t reach its storage", detail: "Your agent can’t save or load anything right now. Restart Kadan Server; if it keeps happening, check Logs in the Kadan Server window." };
  }
  if (abilities.length && ready === abilities.length) {
    return { tone: "ok", title: "Everything is ready", detail: "Your agent can chat, create images, take calls and listen." };
  }
  if (busy) {
    return { tone: "busy", title: "Getting ready", detail: `${ready} of ${abilities.length} abilities are ready. Downloads continue in the background.` };
  }
  return { tone: "warn", title: "Some abilities need setting up", detail: `${ready} of ${abilities.length} abilities are ready. Download the rest from AI models.` };
}
