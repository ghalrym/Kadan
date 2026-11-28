import { useState, type CSSProperties, type ReactNode } from "react";
import { File, FileText, Presentation } from "lucide-react";
import { contentUrl } from "./SpaceIcon";
import type { SpaceItem } from "./types";

/** A coloured square that sits beside sidebar labels, framed like the pet avatars. */
export function NavTile({ hue, size = 30, children }: { hue: number; size?: number; children: ReactNode }) {
  return (
    <span className="nav-tile" aria-hidden="true"
      style={{ "--tile-h": hue, "--tile-size": `${size}px` } as CSSProperties}>
      {children}
    </span>
  );
}

const ITEM_HUES: Record<string, number> = { page: 174, slides: 276, image: 198, file: 222 };

/** A recent Space item: an image thumbnail, or its type icon on a colour that matches the type. */
export function SpaceItemTile({ agentKey, item, size = 22 }: { agentKey: string; item: SpaceItem; size?: number }) {
  const [broken, setBroken] = useState(false);
  if (item.type === "image" && !broken) {
    return (
      <span className="nav-tile has-thumb" aria-hidden="true"
        style={{ "--tile-h": ITEM_HUES.image, "--tile-size": `${size}px` } as CSSProperties}>
        <img src={contentUrl(agentKey, item.id)} alt="" loading="lazy" onError={() => setBroken(true)} />
      </span>
    );
  }
  const Icon = item.type === "page" ? FileText : item.type === "slides" ? Presentation : File;
  return (
    <NavTile hue={ITEM_HUES[item.type] ?? ITEM_HUES.file} size={size}>
      <Icon size={Math.round(size * 0.55)} />
    </NavTile>
  );
}
