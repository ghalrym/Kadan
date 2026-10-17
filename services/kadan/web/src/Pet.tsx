import { useCallback, useEffect, useRef, useState } from "react";
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

// One shared pointer listener feeds every pet that follows the cursor.
const lookers = new Set<(x: number, y: number) => void>();
let lastMove = 0;
addEventListener(
  "pointermove",
  (event) => {
    lastMove = performance.now();
    for (const look of lookers) look(event.clientX, event.clientY);
  },
  { passive: true },
);
const LOOK_HOLD_MS = 2400;

interface Props {
  sheet: PetSheet;
  state?: PetState;
  /** Display width in CSS pixels. Hatchlings stay crisp at multiples of 48. */
  size?: number;
  look?: boolean;
  /** Draw a ground shadow that reacts to jumps. */
  grounded?: boolean;
  className?: string;
}

export default function Pet({
  sheet,
  state = "idle",
  size = 96,
  look = false,
  grounded = false,
  className = "",
}: Props) {
  const element = useRef<HTMLSpanElement>(null);
  const looking = useRef(false);
  const { row, durations } = ROWS[state];

  useEffect(() => {
    const node = element.current;
    if (!node) return;
    const show = (column: number) => {
      if (!looking.current) node.style.backgroundPosition = cellPosition(row, column, sheet.version);
    };
    if (reducedMotion.matches) {
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
  }, [row, durations, sheet.version]);

  useEffect(() => {
    const node = element.current;
    if (!node || !look || sheet.version !== 2 || state !== "idle" || reducedMotion.matches) return;
    let pending = 0;
    let release = 0;
    const stop = () => {
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
    lookers.add(follow);
    return () => {
      lookers.delete(follow);
      cancelAnimationFrame(pending);
      clearTimeout(release);
      looking.current = false;
    };
  }, [look, state, row, sheet.version]);

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
