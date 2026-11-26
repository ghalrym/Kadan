import { useState } from "react";
import { File, FileText, Folder, Presentation } from "lucide-react";
import { agentPath } from "./api";
import type { SpaceItem } from "./types";

export function contentUrl(agentKey: string, id: string, download = false) {
  return `/kadan/api${agentPath(agentKey)}/assets/${encodeURIComponent(id)}/content${download ? "?download=true" : ""}`;
}

/** An item's type icon, or a thumbnail for images. */
export function ItemIcon({ agentKey, item, size = 18 }: { agentKey: string; item: SpaceItem; size?: number }) {
  const [broken, setBroken] = useState(false);
  if (item.type === "image" && !broken && !item.deleted_at) {
    return <img className="item-thumb" src={contentUrl(agentKey, item.id)} alt="" loading="lazy" onError={() => setBroken(true)} />;
  }
  const Icon = item.type === "folder" ? Folder : item.type === "page" ? FileText
    : item.type === "slides" ? Presentation : File;
  return <Icon size={size} />;
}
