import type { ChatEvent } from "./types.ts";

export type AvatarActivity = "idle" | "waiting" | "responding" | "working";
export const activityLabel: Record<AvatarActivity, string> = {
  idle: "Idle",
  waiting: "Waiting for Hikari",
  responding: "Responding",
  working: "Working",
};

export function chatActivity(event: ChatEvent): AvatarActivity {
  if (event.event === "user_saved") return "waiting";
  if (event.event === "delta") return "responding";
  return "idle";
}

// Whole-image parallax only: no pose, mouth, or articulated animation contract.
export function avatarOffset(
  x: number,
  y: number,
  centerX: number,
  centerY: number,
) {
  const dx = x - centerX;
  const dy = y - centerY;
  const distance = Math.max(240, Math.hypot(dx, dy));
  return { x: (dx / distance) * 4, y: (dy / distance) * 4 };
}

export function avatarInitials(name: string) {
  return name
    .trim()
    .split(/\s+/)
    .slice(0, 2)
    .map((part) => Array.from(part)[0] || "")
    .join("")
    .toLocaleUpperCase();
}
