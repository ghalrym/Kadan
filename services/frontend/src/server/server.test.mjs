import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { runInNewContext } from "node:vm";
import ts from "typescript";

function load(file, modules = {}) {
  const source = ts.transpileModule(readFileSync(new URL(file, import.meta.url), "utf8"), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, target: ts.ScriptTarget.ES2022 },
  }).outputText;
  const exports = {};
  runInNewContext(source, { exports, require: (name) => modules[name] ?? {} });
  return exports;
}

const settings = load("../hikari/api/settings.ts", { "../data/models": load("../hikari/data/models.ts") });
const abilities = load("./abilities.ts", { "../hikari/api/settings": settings });
const plain = (value) => JSON.parse(JSON.stringify(value));

const model = (status, extra = {}) => ({
  id: "m", display_name: "Model (experimental)", status, total_bytes: 0, estimated_bytes: 1000,
  downloaded_bytes: 0, error: null, ...extra,
});
const row = (id, m) => ({ capability: { id, label: id }, model: m });

test("abilities use plain names and states for each model's download", () => {
  const ready = abilities.toAbility(row("inkling-2-mini", model("complete")));
  assert.equal(ready.name, "Chat & thinking");
  assert.equal(ready.state, "ready");
  assert.equal(ready.modelName, "Model");

  const downloading = abilities.toAbility(row("flux.2-klein-4b", model("downloading", { total_bytes: 200, downloaded_bytes: 50 })));
  assert.equal(downloading.name, "Images");
  assert.equal(downloading.state, "downloading");
  assert.equal(downloading.progress, 25);

  assert.equal(abilities.toAbility(row("qwen3-asr-1.7b", model("not_downloaded"))).state, "needs_download");
  assert.equal(abilities.toAbility(row("qwen3-asr-1.7b", model("failed"))).state, "failed");
  assert.equal(abilities.toAbility(row("qwen3-asr-1.7b", model("downloading"))).state, "waiting");
  assert.equal(abilities.toAbility(row("flux.2-klein-4b", null)).state, "not_running");
});

test("the Home summary explains the server in one sentence", () => {
  const list = (...states) => states.map((state) => ({ state }));
  assert.equal(abilities.overallStatus(list("ready", "ready"), true).title, "Everything is ready");
  assert.equal(plain(abilities.overallStatus(list("ready", "downloading"), true)).tone, "busy");
  assert.match(abilities.overallStatus(list("ready", "needs_download"), true).detail, /1 of 2/);
  assert.equal(abilities.overallStatus(list("ready"), false).tone, "error");
});

test("Activity and memory labels avoid technical terms", () => {
  const stub = new Proxy({}, { get: () => () => null });
  const activity = load("./ActivityPage.tsx", { react: { useState: (v) => [v, () => {}] }, "react/jsx-runtime": stub, "./api": stub, "./common": stub });
  assert.equal(activity.REQUEST_LABEL.LLM, "Chat reply");
  assert.equal(activity.REQUEST_LABEL.STT, "Dictation");
  assert.equal(activity.requestOutcome({ status: 200 }).label, "Done");
  assert.equal(activity.requestOutcome({ status: 503 }).label, "Failed");
  assert.equal(activity.duration(850), "850 ms");
  assert.equal(activity.duration(2400), "2.4 s");
  assert.equal(activity.duration(125000), "2 min 5 s");

  const home = load("./HomePage.tsx", { "react-router": stub, "lucide-react": stub, "react/jsx-runtime": stub });
  assert.equal(home.meterLabel("GPU 0 memory used"), "Graphics card 1 memory");
  assert.equal(home.meterLabel("System RAM used"), "Computer memory");
});

test("This computer summarizes Kadan's services in one line", () => {
  const stub = new Proxy({}, { get: () => () => null });
  const page = load("./ThisComputer.tsx", { react: { useState: (v) => [v, () => {}], useEffect() {}, useCallback: (f) => f }, "react/jsx-runtime": stub, "lucide-react": stub, "../kadan/desktop": stub, "./common": stub });
  const status = (running, ...phases) => ({ running, services: phases.map((phase) => ({ phase })) });
  assert.equal(page.hostSummary(status(true, "running", "running", "running")).label, "Running on this computer");
  assert.equal(page.hostSummary(status(true, "running", "starting", "stopped")).label, "Starting…");
  assert.equal(page.hostSummary(status(false, "stopped", "stopped", "stopped")).label, "Stopped");
  assert.equal(page.hostSummary(status(true, "running", "error", "running")).tone, "error");
});

test("Kadan names where it runs in plain words", () => {
  const stub = new Proxy({}, { get: () => () => null });
  const settings = load("../kadan/ServerSettings.tsx", { react: stub, "react/jsx-runtime": stub, "lucide-react": stub, "./desktop": stub });
  assert.equal(settings.runLocationLabel({ mode: "host", active_host: "http://10.0.0.2:8000" }), "This computer");
  assert.equal(settings.runLocationLabel({ mode: "connect", active_host: "http://10.0.0.2:8000" }), "http://10.0.0.2:8000");
  assert.equal(settings.runLocationLabel({ active_host: "http://10.0.0.2:8000" }), "This computer");
});
