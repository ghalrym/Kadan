import { createContext, useContext, useEffect, useMemo, useState } from "react";
import { agentPath } from "./api";
import { agentPet } from "./Pet";
import type { PetSheet } from "./pet";
import type { PetJob } from "./types";

export const PetContext = createContext<Record<string, PetJob | null>>({});

export function petSheet(job: PetJob | null): PetSheet | null {
  if (!job?.manifest || job.status !== "ready") return null;
  return {
    url: `/api${agentPath(job.agent_key)}/assets/${encodeURIComponent(job.manifest.atlas_asset_id)}/content`,
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
