import { useEffect, useState, type RefObject } from "react";

/** Closes a popover on outside pointer-down or Escape, and picks whether it
 * opens above or below its trigger based on the space available. */
export function usePopover(
  open: boolean,
  close: () => void,
  rootRef: RefObject<HTMLElement | null>,
  preferredHeight = 400,
): "up" | "down" {
  const [placement, setPlacement] = useState<"up" | "down">("up");

  useEffect(() => {
    if (!open) return;
    const root = rootRef.current;
    if (root && typeof window !== "undefined") {
      const rect = root.getBoundingClientRect();
      const above = rect.top;
      const below = window.innerHeight - rect.bottom;
      setPlacement(above >= preferredHeight || above > below ? "up" : "down");
    }
    const onPointer = (event: PointerEvent) => {
      if (rootRef.current && !rootRef.current.contains(event.target as Node)) close();
    };
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        close();
      }
    };
    document.addEventListener("pointerdown", onPointer);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("pointerdown", onPointer);
      document.removeEventListener("keydown", onKey);
    };
  }, [open, close, rootRef, preferredHeight]);

  return placement;
}
