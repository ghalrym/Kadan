import { COLUMNS, ROWS, type PetSheet, type PetState } from "./pet.ts";

// The default pet: a sprout-topped blob drawn on a 48x52 pixel grid, which is
// a 192x208 Codex cell at 1/4 scale. Rendered with image-rendering: pixelated.
export const GRID_WIDTH = 48;
export const GRID_HEIGHT = 52;
export const ATLAS_ROWS = 11;

const NONE = 0,
  LINE = 1,
  BASE = 2,
  SHADE = 3,
  LIGHT = 4,
  BELLY = 5,
  EYE = 6,
  SHINE = 7,
  BLUSH = 8,
  LEAF = 9,
  LEAF_DARK = 10,
  GLASS = 11;

type Eyes = "open" | "closed" | "happy" | "sad" | "focus" | "wide";
type Mouth = "smile" | "open" | "flat" | "frown" | "o";
interface Pose {
  dx?: number;
  dy?: number;
  sx?: number;
  sy?: number;
  eyes?: Eyes;
  ex?: number;
  ey?: number;
  mouth?: Mouth;
  /** Arm offsets: outward, then downward. */
  armL?: [number, number];
  armR?: [number, number];
  step?: number;
  sway?: number;
  wilt?: boolean;
  drop?: boolean;
  lens?: number;
}

function drawCell(pose: Pose) {
  const grid = new Uint8Array(GRID_WIDTH * GRID_HEIGHT);
  const inside = (x: number, y: number) =>
    x >= 0 && x < GRID_WIDTH && y >= 0 && y < GRID_HEIGHT;
  const get = (x: number, y: number) => (inside(x, y) ? grid[y * GRID_WIDTH + x] : NONE);
  const set = (x: number, y: number, color: number) => {
    x = Math.round(x);
    y = Math.round(y);
    if (inside(x, y)) grid[y * GRID_WIDTH + x] = color;
  };
  const ellipse = (
    cx: number,
    cy: number,
    rx: number,
    ry: number,
    color: (u: number, v: number) => number,
  ) => {
    const painted: number[] = [];
    for (let y = Math.floor(cy - ry - 1); y <= Math.ceil(cy + ry + 1); y++)
      for (let x = Math.floor(cx - rx - 1); x <= Math.ceil(cx + rx + 1); x++) {
        const u = (x + 0.5 - cx) / rx;
        const v = (y + 0.5 - cy) / ry;
        if (u * u + v * v <= 1 && inside(x, y)) {
          grid[y * GRID_WIDTH + x] = color(u, v);
          painted.push(y * GRID_WIDTH + x);
        }
      }
    return painted;
  };
  // Outline every empty (or, for front layers, foreign) pixel touching a shape.
  const outline = (shape?: Set<number>) => {
    const copy = grid.slice();
    for (let y = 0; y < GRID_HEIGHT; y++)
      for (let x = 0; x < GRID_WIDTH; x++) {
        const index = y * GRID_WIDTH + x;
        if (shape ? shape.has(index) : copy[index] !== NONE) continue;
        const touches = [
          [x + 1, y],
          [x - 1, y],
          [x, y + 1],
          [x, y - 1],
        ].some(([nx, ny]) =>
          shape
            ? shape.has(ny * GRID_WIDTH + nx)
            : inside(nx, ny) && copy[ny * GRID_WIDTH + nx] !== NONE && copy[ny * GRID_WIDTH + nx] !== LINE,
        );
        if (touches) grid[index] = LINE;
      }
  };

  const sx = pose.sx ?? 1;
  const sy = pose.sy ?? 1;
  const rx = 15 * sx;
  const ry = 13 * sy;
  const ground = 46 + (pose.dy ?? 0);
  const cx = 24 + (pose.dx ?? 0);
  const cy = ground - ry;

  const step = pose.step ?? 0;
  for (const [side, lift] of [
    [-1, Math.max(0, step)],
    [1, Math.max(0, -step)],
  ])
    ellipse(cx + side * (6.5 + step * side * 1.2), ground + 0.4 - lift * 2, 3.6, 2.6, () => SHADE);

  ellipse(cx, cy, rx, ry, (u, v) => {
    if ((u / 0.5) ** 2 + ((v - 0.62) / 0.36) ** 2 <= 1) return BELLY;
    if (u * 0.5 + v * 0.85 > 0.62) return SHADE;
    if (((u + 0.45) / 0.22) ** 2 + ((v + 0.52) / 0.15) ** 2 <= 1) return LIGHT;
    return BASE;
  });

  const top = Math.round(cy - ry);
  const sway = pose.sway ?? 0;
  const stem = pose.wilt ? 3 : 4;
  let tipX = cx;
  let tipY = top;
  for (let i = 1; i <= stem; i++) {
    tipX = cx + (pose.wilt ? 2 : sway) * (i / stem);
    tipY = top - i + 0.5;
    set(tipX, tipY, LEAF_DARK);
  }
  const leaf = (_u: number, v: number) => (v > 0.25 ? LEAF_DARK : LEAF);
  if (pose.wilt) {
    ellipse(tipX + 3.2, tipY + 2.2, 3.2, 1.6, leaf);
    ellipse(tipX - 2.4, tipY + 1.6, 2.4, 1.4, leaf);
  } else {
    ellipse(tipX + 3.2, tipY - 1.2, 3.6, 2, leaf);
    ellipse(tipX - 2.6, tipY - 0.2, 2.6, 1.5, leaf);
  }
  outline();

  for (const [side, [out, down]] of [
    [-1, pose.armL ?? [0, 0]],
    [1, pose.armR ?? [0, 0]],
  ] as const) {
    const arm = new Set(
      ellipse(cx + side * (rx - 1.5 + out), cy + 3 + down, 2.6, 3.4, (_u, v) =>
        v > 0.3 ? SHADE : BASE,
      ),
    );
    outline(arm);
  }

  const ex = pose.ex ?? 0;
  const ey = pose.ey ?? 0;
  const eyeY = Math.round(cy - 2 + ey);
  const gap = 6 * sx;
  for (const side of [-1, 1]) {
    // Eyes are 4 pixels wide: columns x-1 to x+2, bottom row at y.
    const x = Math.round(cx + ex + side * gap - 0.5);
    const y = eyeY;
    const paint = (rows: string[], mirrored = false) =>
      rows.forEach((line, dy) =>
        [...(mirrored && side > 0 ? [...line].reverse().join("") : line)].forEach((pixel, dx) => {
          if (pixel !== ".") set(x - 1 + dx, y - rows.length + 1 + dy, pixel === "o" ? SHINE : EYE);
        }),
      );
    switch (pose.eyes ?? "open") {
      case "open":
        paint([".##.", "#o##", "#o##", "####", ".##."]);
        break;
      case "wide":
        paint([".##.", "#o##", "#o##", "####", "####", ".##."]);
        break;
      case "closed":
        paint(["####"]);
        break;
      case "happy":
        paint([".##.", "#..#"]);
        break;
      case "sad":
        paint(["#...", ".##.", "#..."], true);
        break;
      case "focus":
        paint(["####", "####"]);
        break;
    }
    const blushX = Math.round(cx + side * (gap + 3.5));
    for (const bx of [blushX, blushX + side])
      if (![NONE, LINE].includes(get(bx, eyeY + 2))) set(bx, eyeY + 2, BLUSH);
  }

  const mx = Math.round(cx + ex * 0.6);
  const my = Math.round(cy + 2 + ey * 0.5);
  switch (pose.mouth ?? "smile") {
    case "smile":
      set(mx - 1, my, LINE);
      set(mx, my + 1, LINE);
      set(mx + 1, my, LINE);
      break;
    case "open":
      for (let dx = -1; dx <= 1; dx++) set(mx + dx, my, LINE);
      set(mx - 1, my + 1, LINE);
      set(mx, my + 1, BLUSH);
      set(mx + 1, my + 1, LINE);
      set(mx, my + 2, LINE);
      break;
    case "flat":
      set(mx, my, LINE);
      set(mx + 1, my, LINE);
      break;
    case "frown":
      set(mx - 1, my + 1, LINE);
      set(mx, my, LINE);
      set(mx + 1, my + 1, LINE);
      break;
    case "o":
      set(mx, my, LINE);
      set(mx + 1, my, LINE);
      set(mx, my + 1, LINE);
      set(mx + 1, my + 1, LINE);
      break;
  }

  if (pose.drop) {
    const x = Math.round(cx + rx * 0.74);
    const y = Math.round(cy - ry * 0.5);
    const drop = new Set<number>();
    for (const [dx, dy] of [
      [0, -1],
      [-1, 0],
      [0, 0],
      [1, 0],
      [-1, 1],
      [0, 1],
      [1, 1],
      [0, 2],
    ]) {
      set(x + dx, y + dy, GLASS);
      drop.add((y + dy) * GRID_WIDTH + x + dx);
    }
    outline(drop);
  }

  if (pose.lens !== undefined) {
    const lx = cx + rx + 0.5;
    const ly = cy - 3 + pose.lens;
    for (let i = 0; i < 4; i++) {
      set(lx - 3 - i * 0.6, ly + 4 + i, LINE);
      set(lx - 2 - i * 0.6, ly + 4 + i, LINE);
    }
    for (let y = Math.floor(ly - 6); y <= ly + 6; y++)
      for (let x = Math.floor(lx - 6); x <= lx + 6; x++) {
        const d = Math.hypot(x + 0.5 - lx, y + 0.5 - ly);
        if (d <= 3.4) set(x, y, d > 1.2 && d < 2.6 && x < lx && y < ly ? SHINE : GLASS);
        else if (d <= 4.6) set(x, y, LINE);
      }
  }
  return grid;
}

function mirror(grid: Uint8Array) {
  const flipped = new Uint8Array(grid.length);
  for (let y = 0; y < GRID_HEIGHT; y++)
    for (let x = 0; x < GRID_WIDTH; x++)
      flipped[y * GRID_WIDTH + x] = grid[y * GRID_WIDTH + GRID_WIDTH - 1 - x];
  return flipped;
}

const wave = (i: number, count: number) => Math.sin((i / count) * Math.PI * 2);
const run = Array.from({ length: 8 }, (_, i): Pose => ({
  dx: 1,
  dy: [0, -1, -2, -1][i % 4],
  sx: 1.03,
  sy: 0.98,
  ex: 2,
  mouth: "open",
  step: wave(i, 8),
  armL: [0, -wave(i, 8) * 2],
  armR: [0, wave(i, 8) * 2],
  sway: -2,
}));

const POSES: Record<PetState, Pose[]> = {
  idle: [{}, { eyes: "closed" }, {}, { sy: 0.96, sx: 1.02, sway: 1 }, { sy: 0.96, sx: 1.02, sway: 1 }, {}],
  "running-right": run,
  "running-left": run,
  waving: [
    { armR: [0, -3] },
    { armR: [1, -8], eyes: "happy", mouth: "open", sway: 1 },
    { armR: [0, -11], eyes: "happy", mouth: "open", sway: -1 },
    { armR: [1, -6] },
  ],
  jumping: [
    { sy: 0.86, sx: 1.1, eyes: "closed", mouth: "flat", armL: [1, 1], armR: [1, 1] },
    { dy: -6, sy: 1.08, sx: 0.94, mouth: "open", armL: [0, -6], armR: [0, -6], sway: -1 },
    { dy: -9, eyes: "happy", mouth: "open", armL: [0, -9], armR: [0, -9] },
    { dy: -4, sy: 1.05, sx: 0.96, eyes: "happy", armL: [1, -3], armR: [1, -3], sway: 1 },
    { sy: 0.9, sx: 1.07, eyes: "happy" },
  ],
  failed: [
    { eyes: "wide", mouth: "o", armL: [1, -3], armR: [1, -3] },
    { eyes: "sad", mouth: "frown", armL: [1, -2], armR: [1, -2], drop: true },
    { eyes: "sad", mouth: "frown", sy: 0.95, sx: 1.03, drop: true },
    { eyes: "sad", mouth: "frown", sy: 0.9, sx: 1.06, wilt: true, drop: true },
    { eyes: "closed", mouth: "frown", sy: 0.88, sx: 1.07, ey: 1, wilt: true, drop: true },
    { eyes: "closed", mouth: "frown", sy: 0.88, sx: 1.07, ey: 1, wilt: true },
    { eyes: "sad", mouth: "frown", sy: 0.89, sx: 1.07, ey: 1, wilt: true },
    { eyes: "closed", mouth: "frown", sy: 0.9, sx: 1.06, ey: 1, wilt: true },
  ],
  waiting: [
    { ey: -1, armL: [-1, -2], armR: [-1, -2] },
    { ey: -1, dy: -1, armL: [-1, -3], armR: [-1, -3], sway: 1 },
    { ey: -1, dy: -1, mouth: "o", armL: [-1, -3], armR: [-1, -3], sway: 1 },
    { ey: -1, armL: [-1, -2], armR: [-1, -2] },
    { ey: -1, eyes: "closed", armL: [-1, -2], armR: [-1, -2] },
    { ey: -1, armL: [-1, -2], armR: [-1, -2], sway: -1 },
  ],
  running: Array.from({ length: 6 }, (_, i): Pose => ({
    eyes: "focus",
    ey: 1,
    mouth: "flat",
    dy: -(i % 2),
    armL: i % 2 ? [-1, -4] : [-1, -1],
    armR: i % 2 ? [-1, -1] : [-1, -4],
    sway: i % 2 ? 2 : -2,
    drop: i >= 3,
  })),
  review: [0, -1, -1, 0, 1, 0].map(
    (lens, i): Pose => ({
      eyes: i === 4 ? "closed" : "focus",
      ex: 1,
      mouth: "flat",
      armR: [0, -2],
      lens,
    }),
  ),
};

function lookPose(step: number): Pose {
  const angle = (step * 22.5 * Math.PI) / 180;
  const vx = Math.sin(angle);
  const vy = -Math.cos(angle);
  return {
    dx: Math.round(vx),
    ex: Math.round(vx * 2),
    ey: Math.round(vy * 1.4),
    sway: Math.round(-vx * 2),
    sy: vy > 0.6 ? 0.97 : 1,
  };
}

/** Every used cell of the v2 atlas: 9 standard rows and 16 look directions. */
export function atlasCells() {
  const cells: { row: number; column: number; grid: Uint8Array }[] = [];
  for (const [state, poses] of Object.entries(POSES) as [PetState, Pose[]][])
    poses.forEach((pose, column) => {
      const grid = drawCell(pose);
      cells.push({
        row: ROWS[state].row,
        column,
        grid: state === "running-left" ? mirror(grid) : grid,
      });
    });
  for (let step = 0; step < 16; step++)
    cells.push({
      row: 9 + Math.floor(step / 8),
      column: step % 8,
      grid: drawCell(lookPose(step)),
    });
  return cells;
}

function hsl(h: number, s: number, l: number): [number, number, number] {
  const a = s * Math.min(l, 1 - l);
  const f = (n: number) => {
    const k = (n + h / 30) % 12;
    return Math.round(255 * (l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1))));
  };
  return [f(0), f(8), f(4)];
}

export function palette(hue: number) {
  const leaf = hue >= 90 && hue <= 200 ? 336 : 128;
  return [
    [0, 0, 0],
    hsl(hue, 0.4, 0.13),
    hsl(hue, 0.72, 0.64),
    hsl(hue, 0.52, 0.47),
    hsl(hue, 0.95, 0.85),
    hsl(hue, 0.8, 0.86),
    [30, 25, 40],
    [255, 255, 255],
    [255, 146, 164],
    hsl(leaf, 0.62, 0.58),
    hsl(leaf, 0.55, 0.4),
    [214, 238, 255],
  ] as [number, number, number][];
}

/** RGBA pixels for a full 8x11 atlas of 48x52 cells. */
export function renderAtlas(hue: number) {
  const width = GRID_WIDTH * COLUMNS;
  const height = GRID_HEIGHT * ATLAS_ROWS;
  const data = new Uint8ClampedArray(width * height * 4);
  const colors = palette(hue);
  for (const { row, column, grid } of atlasCells())
    grid.forEach((color, index) => {
      if (color === NONE) return;
      const x = column * GRID_WIDTH + (index % GRID_WIDTH);
      const y = row * GRID_HEIGHT + Math.floor(index / GRID_WIDTH);
      const offset = (y * width + x) * 4;
      data.set([...colors[color], 255], offset);
    });
  return { width, height, data };
}

const sheets = new Map<number, PetSheet>();

export function hatchlingSheet(hue: number): PetSheet {
  const cached = sheets.get(hue);
  if (cached) return cached;
  const { width, height, data } = renderAtlas(hue);
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  canvas.getContext("2d")!.putImageData(new ImageData(data, width, height), 0, 0);
  const sheet: PetSheet = { url: canvas.toDataURL("image/png"), version: 2, pixelated: true };
  sheets.set(hue, sheet);
  return sheet;
}
