import assert from "node:assert/strict";
import test from "node:test";
import { creationRequest, parsePendingCreation } from "./petCreation.ts";

test("old pending submissions keep their UUID and single-image behavior", () => {
  const pending = parsePendingCreation(JSON.stringify({ id: "old", prompt: "Fox" }));
  assert.equal(pending.reference_mode, "single_image");
  assert.equal(creationRequest(pending, " Fox ", "single_image", () => "new"), pending);
});

test("sheet submission retries retain mode and UUID after reopening", () => {
  const sheet = creationRequest(null, "Fox", "character_sheet", () => "sheet");
  const pending = parsePendingCreation(JSON.stringify(sheet));
  assert.equal(creationRequest(pending, "Fox", "character_sheet", () => "new"), pending);
  assert.equal(pending.id, "sheet");
  assert.equal(pending.reference_mode, "character_sheet");
  assert.equal(creationRequest(pending, "Fox", "single_image", () => "changed").id, "changed");
  assert.equal(creationRequest(pending, "Owl", "character_sheet", () => "owl").id, "owl");
});

test("invalid saved modes are discarded", () => {
  for (const raw of [null, "{", '{"id":"old","prompt":"Fox","reference_mode":"unknown"}']) {
    assert.equal(parsePendingCreation(raw), null);
  }
});
