import { useCallback, useEffect, useRef, useState, useSyncExternalStore } from "react";
import { hatchlingSheet } from "./hatchling";
import {
  CELL_RATIO,
  COLUMNS,
  ROWS,
  agentHue,
  cellPosition,
  cycleLength,
  lookAngle,
  lookCell,
  sheetRows,
  type PetSheet,
  type PetState,
} from "./pet";

/** The pet for an agent. Agents without an installed pet get a hatchling. */
export function agentPet(key: string): PetSheet {
  return hatchlingSheet(agentHue(key));
}

const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");
const subscribeMotion = (notify: () => void) => {
  reducedMotion.addEventListener("change", notify);
  return () => reducedMotion.removeEventListener("change", notify);
};
const motionSnapshot = () => reducedMotion.matches;

// One shared pointer listener feeds every pet that follows the cursor.
const lookers = new Set<{ follow: (x: number, y: number) => void; stop: () => void }>();
let lastMove = 0;
addEventListener(
  "pointermove",
  (event) => {
    if (event.pointerType !== "mouse") return;
    lastMove = performance.now();
    for (const look of lookers) look.follow(event.clientX, event.clientY);
  },
  { passive: true },
);
const stopLooking = () => {
  for (const look of lookers) look.stop();
};
addEventListener("blur", stopLooking);
addEventListener("pointerout", (event) => {
  if (event.pointerType === "mouse" && event.relatedTarget === null) stopLooking();
});
const LOOK_HOLD_MS = 2400;

interface Props {
  sheet: PetSheet;
  state?: PetState;
  /** Display width in CSS pixels. Hatchlings stay crisp at multiples of 48. */
  size?: number;
  look?: boolean;
  /** Explicit look pose for reviewing a candidate atlas. */
  lookDirection?: number;
  /** Draw a ground shadow that reacts to jumps. */
  grounded?: boolean;
  className?: string;
}

export default function Pet({
  sheet,
  state = "idle",
  size = 96,
  look = false,
  lookDirection,
  grounded = false,
  className = "",
}: Props) {
  const element = useRef<HTMLSpanElement>(null);
  const looking = useRef(false);
  const reduceMotion = useSyncExternalStore(subscribeMotion, motionSnapshot);
  const { row, durations } = sheet.animations?.find((animation) => animation.name === state) ?? ROWS[state];

  useEffect(() => {
    const node = element.current;
    if (!node) return;
    const show = (column: number) => {
      if (!looking.current) node.style.backgroundPosition = cellPosition(row, column, sheet.version);
    };
    if (lookDirection !== undefined && sheet.version === 2) {
      const cell = lookCell(lookDirection);
      node.style.backgroundPosition = cellPosition(cell.row, cell.column, sheet.version);
      return;
    }
    if (reduceMotion) {
      show(0);
      return;
    }
    let frame = 0;
    let timer = 0;
    const advance = () => {
      show(frame);
      timer = window.setTimeout(() => {
        frame = (frame + 1) % durations.length;
        advance();
      }, durations[frame]);
    };
    advance();
    return () => clearTimeout(timer);
  }, [row, durations, sheet.version, reduceMotion, lookDirection]);

  useEffect(() => {
    const node = element.current;
    if (!node || !look || sheet.version !== 2 || state !== "idle" || reduceMotion || lookDirection !== undefined) return;
    let pending = 0;
    let release = 0;
    const stop = () => {
      cancelAnimationFrame(pending);
      clearTimeout(release);
      looking.current = false;
      node.style.backgroundPosition = cellPosition(row, 0, sheet.version);
    };
    const follow = (x: number, y: number) => {
      cancelAnimationFrame(pending);
      pending = requestAnimationFrame(() => {
        const box = node.getBoundingClientRect();
        const dx = x - (box.left + box.width / 2);
        const dy = y - (box.top + box.height / 2);
        clearTimeout(release);
        // Inside the pet's own box is the neutral deadzone.
        if (Math.hypot(dx, dy) < box.width * 0.6) return stop();
        const cell = lookCell(lookAngle(dx, dy));
        looking.current = true;
        node.style.backgroundPosition = cellPosition(cell.row, cell.column, sheet.version);
        release = window.setTimeout(() => {
          if (performance.now() - lastMove >= LOOK_HOLD_MS) stop();
        }, LOOK_HOLD_MS);
      });
    };
    const looker = { follow, stop };
    lookers.add(looker);
    return () => {
      lookers.delete(looker);
      stop();
    };
  }, [look, state, row, sheet.version, reduceMotion, lookDirection]);

  return (
    <span
      aria-hidden="true"
      className={`pet-frame ${grounded ? "grounded" : ""} ${className}`}
      data-state={state}
    >
      <span
        ref={element}
        className={`pet ${sheet.pixelated ? "pixelated" : ""}`}
        style={{
          width: size,
          height: Math.round(size * CELL_RATIO),
          backgroundImage: `url("${sheet.url}")`,
          backgroundSize: `${COLUMNS * 100}% ${sheetRows(sheet.version) * 100}%`,
        }}
      />
    </span>
  );
}

/** A resting pet state plus short reactions that play once and settle back. */
export function useMood(resting: PetState) {
  const [reaction, setReaction] = useState<{ state: PetState; until: number } | null>(null);
  useEffect(() => {
    if (!reaction) return;
    const timer = setTimeout(() => setReaction(null), reaction.until - Date.now());
    return () => clearTimeout(timer);
  }, [reaction]);
  const react = useCallback((state: PetState, loops = 1) => {
    setReaction({ state, until: Date.now() + cycleLength(state) * loops });
  }, []);
  return [reaction?.state ?? resting, react] as const;
}
