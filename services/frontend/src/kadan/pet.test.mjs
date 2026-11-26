import assert from "node:assert/strict";
import { test } from "node:test";
import { ROWS, agentHue, cellPosition, cycleLength, lookAngle, lookCell } from "./pet.ts";
import { ATLAS_ROWS, GRID_HEIGHT, GRID_WIDTH, atlasCells, renderAtlas } from "./hatchling.ts";

test("animation rows follow the Codex atlas contract", () => {
  assert.deepEqual(
    Object.entries(ROWS).map(([state, { row, durations }]) => [state, row, durations.length]),
    [
      ["idle", 0, 6],
      ["running-right", 1, 8],
      ["running-left", 2, 8],
      ["waving", 3, 4],
      ["jumping", 4, 5],
      ["failed", 5, 8],
      ["waiting", 6, 6],
      ["running", 7, 6],
      ["review", 8, 6],
    ],
  );
  assert.equal(cycleLength("idle"), 1100);
  assert.equal(cycleLength("waving"), 700);
});

test("look angles run clockwise from 12 o'clock across rows 9 and 10", () => {
  assert.equal(lookAngle(0, -1), 0);
  assert.equal(lookAngle(1, 0), 90);
  assert.equal(lookAngle(0, 1), 180);
  assert.equal(lookAngle(-1, 0), 270);
  assert.deepEqual(lookCell(0), { row: 9, column: 0 });
  assert.deepEqual(lookCell(157.5), { row: 9, column: 7 });
  assert.deepEqual(lookCell(180), { row: 10, column: 0 });
  assert.deepEqual(lookCell(337.5), { row: 10, column: 7 });
  assert.deepEqual(lookCell(355), { row: 9, column: 0 });
});

test("cell positions address the atlas corners", () => {
  assert.equal(cellPosition(0, 0, 2), "0% 0%");
  assert.equal(cellPosition(10, 7, 2), "100% 100%");
  assert.equal(cellPosition(8, 7, 1), "100% 100%");
});

test("agent hues are stable per key", () => {
  assert.equal(agentHue("research"), agentHue("research"));
  assert.ok(Number.isInteger(agentHue("")));
});

test("hatchlings fill every used cell and leave the rest transparent", () => {
  const cells = atlasCells();
  const used = new Set(cells.map(({ row, column }) => `${row}:${column}`));
  for (const { row, durations } of Object.values(ROWS))
    for (let column = 0; column < 8; column++)
      assert.equal(used.has(`${row}:${column}`), column < durations.length);
  for (let column = 0; column < 8; column++) {
    assert.ok(used.has(`9:${column}`));
    assert.ok(used.has(`10:${column}`));
  }
  for (const { grid } of cells) assert.ok(grid.some((pixel) => pixel !== 0));

  const { width, height, data } = renderAtlas(200);
  assert.equal(width, GRID_WIDTH * 8);
  assert.equal(height, GRID_HEIGHT * ATLAS_ROWS);
  const alpha = (row, column) => {
    let total = 0;
    for (let y = 0; y < GRID_HEIGHT; y++)
      for (let x = 0; x < GRID_WIDTH; x++)
        total += data[((row * GRID_HEIGHT + y) * width + column * GRID_WIDTH + x) * 4 + 3];
    return total;
  };
  assert.equal(alpha(ROWS.waving.row, 4), 0);
  assert.ok(alpha(ROWS.waving.row, 3) > 0);
});
