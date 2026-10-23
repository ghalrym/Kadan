// Picks the files of a Codex (ChatGPT) pet package from whatever the user
// dropped or chose: a zip, a pet folder, or loose pet.json + spritesheet files.
// The server re-validates everything; this only selects files and prefills names.

export interface PickedFile<F = File> {
  file: F;
  /** Path relative to the dropped or chosen folder, using "/" separators. */
  path: string;
}

export interface PetJsonInfo {
  displayName?: string;
  description?: string;
  spriteVersionNumber?: number;
  spritesheetPath?: string;
}

export type PetPackage<F = File> =
  | { kind: "zip"; files: F[]; zip: F }
  | { kind: "files"; files: F[]; sheet: F; manifest: F | null; info: PetJsonInfo | null }
  | { kind: "error"; message: string };

const basename = (path: string) => path.slice(path.lastIndexOf("/") + 1);
const dirname = (path: string) => path.slice(0, Math.max(0, path.lastIndexOf("/")));
const depth = (path: string) => path.split("/").length;
const isImage = (path: string) => /\.(webp|png)$/i.test(path);

export function normalizePath(path: string): string | null {
  const parts: string[] = [];
  for (const part of path.replaceAll("\\", "/").split("/")) {
    if (!part || part === ".") continue;
    if (part === "..") {
      if (!parts.length) return null;
      parts.pop();
    } else parts.push(part);
  }
  return parts.join("/");
}

export function parsePetJson(text: string): PetJsonInfo | null {
  try {
    const value = JSON.parse(text);
    if (!value || typeof value !== "object" || Array.isArray(value)) return null;
    const pick = (key: string) => (typeof value[key] === "string" ? value[key].trim() || undefined : undefined);
    return {
      displayName: pick("displayName") ?? pick("id"),
      description: pick("description"),
      spriteVersionNumber: typeof value.spriteVersionNumber === "number" ? value.spriteVersionNumber : undefined,
      spritesheetPath: pick("spritesheetPath"),
    };
  } catch {
    return null;
  }
}

/** Choose the package files; `readText` reads the pet.json so its spritesheet path can be followed. */
export async function resolvePetPackage<F>(
  picked: PickedFile<F>[],
  readText: (file: F) => Promise<string>,
): Promise<PetPackage<F>> {
  const items = picked
    .map((item) => ({ ...item, path: normalizePath(item.path) ?? basename(item.path) }))
    .filter((item) => !basename(item.path).startsWith("."));
  const zips = items.filter((item) => /\.zip$/i.test(item.path));
  if (zips.length) {
    return zips.length === 1 && items.length === 1
      ? { kind: "zip", files: [zips[0].file], zip: zips[0].file }
      : { kind: "error", message: "Choose a single zip on its own." };
  }
  const manifests = items
    .filter((item) => basename(item.path) === "pet.json")
    .sort((a, b) => depth(a.path) - depth(b.path));
  if (manifests.length > 1 && depth(manifests[0].path) === depth(manifests[1].path)) {
    return { kind: "error", message: "That selection contains more than one pet. Import one at a time." };
  }
  const images = items.filter((item) => isImage(item.path));
  const manifest = manifests[0] ?? null;
  const info = manifest ? parsePetJson(await readText(manifest.file)) : null;
  if (manifest && !info) return { kind: "error", message: "pet.json could not be read." };
  let sheet = images.length === 1 ? images[0] : undefined;
  if (manifest) {
    const declared = info?.spritesheetPath ?? "spritesheet.webp";
    const wanted = normalizePath(`${dirname(manifest.path)}/${declared}`);
    const loose = items.every((item) => !item.path.includes("/"));
    sheet = images.find((item) => item.path === wanted)
      // Only loose files have lost the directory context needed for an exact match.
      ?? (loose ? images.find((item) => wanted && item.path === basename(wanted)) : undefined);
    if (!sheet) return { kind: "error", message: `The selection is missing the spritesheet ${declared} named in pet.json.` };
  }
  if (!sheet) {
    return {
      kind: "error",
      message: images.length
        ? "Several images were selected. Include the pet.json, or choose only the spritesheet."
        : "Add the pet's spritesheet (.webp or .png), its folder, or a zip of it.",
    };
  }
  return {
    kind: "files",
    files: manifest ? [manifest.file, sheet.file] : [sheet.file],
    sheet: sheet.file,
    manifest: manifest?.file ?? null,
    info,
  };
}

/** Codex atlas version from pixel dimensions: v1 has 9 rows, v2 has 11. */
export function atlasVersion(width: number, height: number): 1 | 2 | null {
  if (width !== 1536) return null;
  return height === 1872 ? 1 : height === 2288 ? 2 : null;
}
