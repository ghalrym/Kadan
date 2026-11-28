import { createContext, useContext, useEffect, useMemo, useState } from "react";
import { agentPath } from "./api";
import { agentPet } from "./Pet";
import type { PetSheet } from "./pet";
import { loadPetRef, petRefKey, type PetRef } from "./profile";
import type { PetJob } from "./types";

export const PetContext = createContext<Record<string, PetJob | null>>({});

export function petSheet(job: PetJob | null): PetSheet | null {
  if (!job?.manifest || job.status !== "ready") return null;
  return {
    url: `/kadan/api${agentPath(job.agent_key)}/assets/${encodeURIComponent(job.manifest.atlas_asset_id)}/content`,
    version: job.manifest.sprite_version,
    pixelated: job.manifest.pixelated,
    animations: job.manifest.animations,
  };
}

/** Load the atlas before replacing a visible pet; broken assets retain the hatchling. */
export function usePetImage(sheet: PetSheet | null) {
  const [loaded, setLoaded] = useState("");
  const [failed, setFailed] = useState("");
  const url = sheet?.url ?? "";
  useEffect(() => {
    if (!url) return;
    const image = new Image();
    image.onload = () => { setLoaded(url); setFailed(""); };
    image.onerror = () => { setFailed(url); setLoaded(""); };
    image.src = url;
    return () => { image.onload = null; image.onerror = null; };
  }, [url]);
  return { ready: !!url && loaded === url, failed: !!url && failed === url };
}

export function useAgentPet(key: string): PetSheet {
  const selected = useContext(PetContext)[key] ?? null;
  const sheet = useMemo(() => petSheet(selected), [selected]);
  const image = usePetImage(sheet);
  return sheet && image.ready ? sheet : agentPet(key);
}

/** The finished pet a person or another agent borrowed, or null while loading or when it is gone. */
export function usePetRefJob(ref: PetRef | null | undefined): PetJob | null {
  const key = petRefKey(ref);
  const [loaded, setLoaded] = useState<{ key: string; job: PetJob | null }>({ key: "", job: null });
  useEffect(() => {
    if (!ref) return;
    let alive = true;
    void loadPetRef(ref).then((job) => { if (alive) setLoaded({ key, job }); });
    return () => { alive = false; };
    // The key captures every field of the reference.
  }, [key]);
  return key && loaded.key === key ? loaded.job : null;
}

/** Sprite sheet for a borrowed pet, falling back to the default pet until it loads. */
export function usePetRefSheet(ref: PetRef | null | undefined): PetSheet {
  const job = usePetRefJob(ref);
  const sheet = useMemo(() => petSheet(job), [job]);
  const image = usePetImage(sheet);
  return sheet && image.ready ? sheet : agentPet();
}
