import type { SpaceItem } from "./types.ts";

export type SpaceTab = "all" | "page" | "slides" | "image" | "file" | "pinned";

export const SPACE_TABS: { id: SpaceTab; label: string }[] = [
  { id: "all", label: "All" },
  { id: "page", label: "Pages" },
  { id: "slides", label: "Slides" },
  { id: "image", label: "Images" },
  { id: "file", label: "Files" },
  { id: "pinned", label: "Pinned" },
];

/** Folders first, then the most recently changed. */
export function sortItems(items: SpaceItem[]): SpaceItem[] {
  return [...items].sort((a, b) =>
    Number(b.type === "folder") - Number(a.type === "folder")
    || b.updated_at.localeCompare(a.updated_at)
    || a.id.localeCompare(b.id));
}

/**
 * The rows to show. Browsing "All" without a search shows one folder level; a type tab,
 * Pinned or a search lists every match across all folders.
 */
export function visibleItems(items: SpaceItem[], tab: SpaceTab, query: string, folderId: string | null): SpaceItem[] {
  const needle = query.trim().toLowerCase();
  const flat = tab !== "all" || !!needle;
  return sortItems(items.filter((item) => {
    if (!flat && item.parent_id !== folderId) return false;
    if (tab === "pinned" && !item.pinned) return false;
    if (tab !== "all" && tab !== "pinned" && item.type !== tab) return false;
    if (needle && !`${item.title} ${item.filename ?? ""}`.toLowerCase().includes(needle)) return false;
    return true;
  }));
}

export function childrenOf(items: SpaceItem[], parentId: string): SpaceItem[] {
  return sortItems(items.filter((item) => item.parent_id === parentId));
}

/** Folders from the top of Space down to the given folder. */
export function folderTrail(items: SpaceItem[], folderId: string | null): SpaceItem[] {
  const trail: SpaceItem[] = [];
  const seen = new Set<string>();
  let current = folderId ? items.find((item) => item.id === folderId) : undefined;
  while (current && !seen.has(current.id)) {
    seen.add(current.id);
    trail.unshift(current);
    current = current.parent_id ? items.find((item) => item.id === current!.parent_id) : undefined;
  }
  return trail;
}

/** Folders an item can move into: never itself or anything inside it. */
export function moveTargets(items: SpaceItem[], moving: SpaceItem): SpaceItem[] {
  const blocked = new Set([moving.id]);
  let grew = true;
  while (grew) {
    grew = false;
    for (const item of items) {
      if (item.parent_id && blocked.has(item.parent_id) && !blocked.has(item.id)) {
        blocked.add(item.id);
        grew = true;
      }
    }
  }
  return items
    .filter((item) => item.type === "folder" && !blocked.has(item.id))
    .sort((a, b) => a.title.localeCompare(b.title));
}

export function recentItems(items: SpaceItem[], count = 5): SpaceItem[] {
  return [...items]
    .filter((item) => item.type !== "folder")
    .sort((a, b) => b.updated_at.localeCompare(a.updated_at))
    .slice(0, count);
}

export function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB"];
  let value = bytes / 1024;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit++;
  }
  return `${value >= 10 ? Math.round(value) : value.toFixed(1)} ${units[unit]}`;
}

/** "8:25 AM" today, "Yesterday", then a short date, like a file browser. */
export function itemDate(value: string, now = new Date()): string {
  const date = new Date(value);
  const days = Math.round(
    (new Date(now).setHours(0, 0, 0, 0) - new Date(date).setHours(0, 0, 0, 0)) / 86400000,
  );
  if (days === 0) return date.toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" });
  if (days === 1) return "Yesterday";
  return date.toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
    ...(date.getFullYear() === now.getFullYear() ? {} : { year: "numeric" }),
  });
}

export function itemMeta(item: SpaceItem, now = new Date()): string {
  const parts = [itemDate(item.updated_at, now)];
  if (item.size_bytes !== null && item.type !== "folder") parts.push(formatSize(item.size_bytes));
  return parts.join(" · ");
}

export const TYPE_LABEL: Record<SpaceItem["type"], string> = {
  folder: "Folder",
  page: "Page",
  slides: "Slides",
  image: "Image",
  file: "File",
};
