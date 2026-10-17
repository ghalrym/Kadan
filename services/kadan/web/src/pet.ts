// Codex pet atlas contract: 8 columns of 192x208 cells. v1 sheets have the 9
// standard rows; v2 sheets add two rows of 16 clockwise look directions.
export type PetState =
  | "idle"
  | "running-right"
  | "running-left"
  | "waving"
  | "jumping"
  | "failed"
  | "waiting"
  | "running"
  | "review";

export interface PetSheet {
  url: string;
  version: 1 | 2;
  pixelated?: boolean;
  animations?: { name: PetState; row: number; durations: number[] }[];
}

export const COLUMNS = 8;
export const CELL_RATIO = 208 / 192;

const frames = (count: number, each: number, last: number) => [
  ...Array<number>(count - 1).fill(each),
  last,
];

export const ROWS: Record<PetState, { row: number; durations: number[] }> = {
  idle: { row: 0, durations: [280, 110, 110, 140, 140, 320] },
  "running-right": { row: 1, durations: frames(8, 120, 220) },
  "running-left": { row: 2, durations: frames(8, 120, 220) },
  waving: { row: 3, durations: frames(4, 140, 280) },
  jumping: { row: 4, durations: frames(5, 140, 280) },
  failed: { row: 5, durations: frames(8, 140, 240) },
  waiting: { row: 6, durations: frames(6, 150, 260) },
  running: { row: 7, durations: frames(6, 120, 220) },
  review: { row: 8, durations: frames(6, 150, 280) },
};

export const sheetRows = (version: 1 | 2) => (version === 2 ? 11 : 9);

export const cycleLength = (state: PetState) =>
  ROWS[state].durations.reduce((total, ms) => total + ms, 0);

/** Degrees clockwise from 12 o'clock for a screen-space vector. */
export function lookAngle(dx: number, dy: number) {
  return ((Math.atan2(dx, -dy) * 180) / Math.PI + 360) % 360;
}

/** Atlas cell for the nearest of the 16 look directions in rows 9 and 10. */
export function lookCell(angle: number) {
  const step = Math.round((((angle % 360) + 360) % 360) / 22.5) % 16;
  return { row: 9 + Math.floor(step / 8), column: step % 8 };
}

/** Background offsets that place one atlas cell in an element of any size. */
export function cellPosition(row: number, column: number, version: 1 | 2) {
  return `${(column / (COLUMNS - 1)) * 100}% ${(row / (sheetRows(version) - 1)) * 100}%`;
}

const HUES = [350, 16, 38, 152, 174, 198, 222, 252, 276, 312];

/** Stable signature hue for an agent, shared by its hatchling and the UI accent. */
export function agentHue(key: string) {
  let hash = 0x811c9dc5;
  for (const char of key) hash = Math.imul(hash ^ char.charCodeAt(0), 0x01000193);
  return HUES[(hash >>> 0) % HUES.length];
}
