import { agentPath, api } from "./api";
import type { PetJob } from "./types";

/** A finished pet from an agent's pet library, borrowed by a person or another agent. */
export interface PetRef {
  agent_key: string;
  pet_id: string;
}

export interface UserProfile {
  display_name: string;
  pet: PetRef | null;
  version: number;
  updated_at: string | null;
}

export const DEFAULT_PROFILE: UserProfile = { display_name: "You", pet: null, version: 0, updated_at: null };

/** Avatar colour seed for the person using Kadan; agents use their key or handle. */
export const USER_SEED = "kadan:user";

export function petRefKey(ref: PetRef | null | undefined): string {
  return ref ? `${ref.agent_key}/${ref.pet_id}` : "";
}

const jobs = new Map<string, Promise<PetJob | null>>();

/** Read a borrowed pet once per session; a missing pet resolves to null so its owner shows the default pet. */
export function loadPetRef(ref: PetRef): Promise<PetJob | null> {
  const key = petRefKey(ref);
  let job = jobs.get(key);
  if (!job) {
    job = api<PetJob>(`${agentPath(ref.agent_key)}/pets/${encodeURIComponent(ref.pet_id)}`).catch(() => {
      jobs.delete(key);
      return null;
    });
    jobs.set(key, job);
  }
  return job;
}

export function saveProfile(
  profile: UserProfile,
  change: { display_name?: string; pet?: PetRef; clear_pet?: boolean },
): Promise<UserProfile> {
  return api<UserProfile>("/profile", { ...change, expected_version: profile.version }, "PUT");
}
