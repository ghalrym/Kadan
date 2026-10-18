import assert from "node:assert/strict";
import { test } from "node:test";
import { atlasVersion, normalizePath, parsePetJson, resolvePetPackage } from "./petImport.ts";

const files = (entries) => Object.entries(entries).map(([path, text]) => ({ file: { path, text }, path }));
const read = async (file) => file.text;
const paths = (result) => result.files.map((file) => file.path);

test("a zip is sent on its own", async () => {
  const result = await resolvePetPackage(files({ "fox.zip": "" }), read);
  assert.equal(result.kind, "zip");
  assert.deepEqual(paths(result), ["fox.zip"]);
  const mixed = await resolvePetPackage(files({ "fox.zip": "", "pet.json": "{}" }), read);
  assert.equal(mixed.kind, "error");
});

test("a dropped folder follows pet.json to its spritesheet and skips extras", async () => {
  const result = await resolvePetPackage(
    files({
      "fox/pet.json": JSON.stringify({ displayName: "Copper Fox", description: "Small", spritesheetPath: "art/sheet.png" }),
      "fox/art/sheet.png": "",
      "fox/preview.png": "",
      "fox/.DS_Store": "",
    }),
    read,
  );
  assert.equal(result.kind, "files");
  assert.deepEqual(paths(result), ["fox/pet.json", "fox/art/sheet.png"]);
  assert.equal(result.info.displayName, "Copper Fox");
});

test("loose files match the spritesheet by name", async () => {
  const result = await resolvePetPackage(
    files({ "pet.json": JSON.stringify({ id: "ember" }), "spritesheet.webp": "", "notes.png": "" }),
    read,
  );
  assert.deepEqual(paths(result), ["pet.json", "spritesheet.webp"]);
  assert.equal(result.info.displayName, "ember");
});

test("a lone spritesheet is enough", async () => {
  const result = await resolvePetPackage(files({ "ember.webp": "" }), read);
  assert.deepEqual(paths(result), ["ember.webp"]);
  assert.equal(result.manifest, null);
});

test("ambiguous or incomplete selections explain what is missing", async () => {
  for (const [entries, message] of [
    [{ "a.png": "", "b.png": "" }, /Several images/],
    [{ "pet.json": "{}" }, /Add the pet's spritesheet/],
    [{ "pet.json": "{", "s.webp": "" }, /could not be read/],
    [{ "a/pet.json": "{}", "b/pet.json": "{}", "a/spritesheet.webp": "" }, /more than one pet/],
  ]) {
    const result = await resolvePetPackage(files(entries), read);
    assert.equal(result.kind, "error");
    assert.match(result.message, message);
  }
});

test("paths cannot escape the chosen folder", () => {
  assert.equal(normalizePath("fox/./art/../sheet.webp"), "fox/sheet.webp");
  assert.equal(normalizePath("../sheet.webp"), null);
  assert.equal(parsePetJson("[]"), null);
});

test("atlas version follows Codex dimensions", () => {
  assert.equal(atlasVersion(1536, 1872), 1);
  assert.equal(atlasVersion(1536, 2288), 2);
  assert.equal(atlasVersion(1024, 2288), null);
});
