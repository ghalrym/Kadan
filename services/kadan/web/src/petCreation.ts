import type { PetJob } from "./types";

export interface CreationRequest {
  id: string;
  prompt: string;
  reference_mode: PetJob["reference_mode"];
}

export function parsePendingCreation(raw: string | null): CreationRequest | null {
  try {
    const value = JSON.parse(raw ?? "null");
    if (!value || typeof value.id !== "string" || typeof value.prompt !== "string") return null;
    const mode = value.reference_mode ?? "single_image";
    if (mode !== "single_image" && mode !== "character_sheet") return null;
    return { id: value.id, prompt: value.prompt, reference_mode: mode };
  } catch { return null; }
}

export function creationRequest(
  pending: CreationRequest | null,
  prompt: string,
  referenceMode: PetJob["reference_mode"],
  newId: () => string,
): CreationRequest {
  if (pending?.prompt === prompt.trim() && pending.reference_mode === referenceMode) return pending;
  return { id: newId(), prompt: prompt.trim(), reference_mode: referenceMode };
}
