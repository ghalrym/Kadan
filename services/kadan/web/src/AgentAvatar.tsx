import { useEffect, useRef, useState } from "react";
import { agentPath } from "./api";
import StillImage from "./StillImage";
import { activityLabel, avatarInitials, avatarOffset } from "./avatar";
import type { AvatarActivity } from "./avatar";

export default function AgentAvatar({
  agentKey,
  name,
  assetId,
  activity = "idle",
  size = "small",
  follow = false,
}: {
  agentKey: string;
  name: string;
  assetId: string | null;
  activity?: AvatarActivity;
  size?: "small" | "large";
  follow?: boolean;
}) {
  const frame = useRef<HTMLSpanElement>(null);
  const portrait = useRef<HTMLSpanElement>(null);
  const [failed, setFailed] = useState(false);
  useEffect(() => setFailed(false), [agentKey, assetId]);
  useEffect(() => {
    const image = portrait.current;
    if (!image || !follow || activity !== "idle") return;
    const motion = matchMedia("(prefers-reduced-motion: reduce)");
    const pointer = matchMedia("(any-hover: hover) and (any-pointer: fine)");
    let pending = 0;
    const reset = () => {
      cancelAnimationFrame(pending);
      pending = 0;
      image.style.transform = "";
    };
    const move = (event: PointerEvent) => {
      if (
        event.pointerType !== "mouse" ||
        motion.matches ||
        !pointer.matches ||
        document.hidden
      ) {
        reset();
        return;
      }
      cancelAnimationFrame(pending);
      pending = requestAnimationFrame(() => {
        pending = 0;
        const bounds = frame.current?.getBoundingClientRect();
        if (!bounds) return;
        const offset = avatarOffset(
          event.clientX,
          event.clientY,
          bounds.x + bounds.width / 2,
          bounds.y + bounds.height / 2,
        );
        image.style.transform = `translate(${offset.x}px, ${offset.y}px) rotate(${offset.x / 2}deg)`;
      });
    };
    const leave = (event: PointerEvent) => {
      if (!event.relatedTarget) reset();
    };
    window.addEventListener("pointermove", move, { passive: true });
    window.addEventListener("pointerdown", reset, { passive: true });
    window.addEventListener("pointerout", leave);
    window.addEventListener("blur", reset);
    document.addEventListener("visibilitychange", reset);
    motion.addEventListener("change", reset);
    pointer.addEventListener("change", reset);
    return () => {
      reset();
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerdown", reset);
      window.removeEventListener("pointerout", leave);
      window.removeEventListener("blur", reset);
      document.removeEventListener("visibilitychange", reset);
      motion.removeEventListener("change", reset);
      pointer.removeEventListener("change", reset);
    };
  }, [follow, activity, agentKey, assetId]);
  return (
    <span
      ref={frame}
      className={`agent-avatar avatar-${size}`}
      data-activity={activity}
      role="img"
      aria-label={`${name} · ${activityLabel[activity]}${failed ? " · Image unavailable" : ""}`}
    >
      <span className="avatar-portrait" ref={portrait}>
        {assetId && !failed ? (
          <StillImage
            src={`/api${agentPath(agentKey)}/assets/${encodeURIComponent(assetId)}/content`}
            onError={() => setFailed(true)}
          />
        ) : (
          <span className="avatar-initials">{avatarInitials(name)}</span>
        )}
      </span>
      <span className="avatar-activity" aria-hidden="true" />
    </span>
  );
}
