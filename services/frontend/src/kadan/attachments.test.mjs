import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { registerHooks } from "node:module";
import { test } from "node:test";
import ts from "typescript";
import { ApiError } from "./api.ts";

const url = new URL("./attachments.ts", import.meta.url).href;
const loader = registerHooks({ load(target, context, next) {
  if (target !== url) return next(target, context);
  return { format: "module", shortCircuit: true, source: ts.transpileModule(readFileSync(new URL(url), "utf8"), {
    compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 },
  }).outputText };
} });
const { ChatAttachments, MAX_ATTACHMENT_BYTES } = await import(url);
loader.deregister();
const tick = () => new Promise(resolve => setImmediate(resolve));
const file = (name = "report.txt", content = "abc") => new File([content], name);
const asset = (id, filename = "report.txt") => ({ id, filename, agent_key: "agent", size_bytes: 3, deleted_at: null });
function setup(saved = new Map(), agent = "agent", conversation = "chat") {
  const requests = [];
  let nextId = 0, changes = 0;
  const storage = { getItem: key => saved.get(key) ?? null, setItem: (key, value) => saved.set(key, value), removeItem: key => saved.delete(key) };
  const request = (path, body) => new Promise((resolve, reject) => requests.push({ path, body, resolve, reject }));
  const manager = new ChatAttachments(agent, conversation, storage, () => changes++, request, () => `file-${++nextId}`);
  manager.start();
  return { manager, requests, saved, changes: () => changes };
}

test("uploads serialize with stable IDs and retain selection order and authoritative filenames", async () => {
  const { manager, requests, saved } = setup();
  manager.add([file(), file("second.txt")]);
  assert.equal(manager.ready, false);
  assert.throws(() => manager.ids(), /Finish uploading/);
  assert.equal(requests.length, 1);
  assert.equal(requests[0].body.get("asset_id"), "file-1");
  assert.equal(await requests[0].body.get("file").text(), "abc");
  requests[0].resolve(asset("file-1", "normalized.txt"));
  await tick();
  assert.equal(requests.length, 2);
  requests[1].resolve(asset("file-2", "second.txt"));
  await tick();
  assert.deepEqual(manager.ids(), ["file-1", "file-2"]);
  assert.equal(manager.entries[0].name, "normalized.txt");
  assert.equal(JSON.parse(saved.get("kadan.attachments.agent.chat"))[1].status, "ready");
});

test("lost upload response is checked by the same ID before retry and confirmed without another upload", async () => {
  const { manager, requests } = setup();
  manager.add([file()]);
  requests[0].reject(new Error("connection lost"));
  await tick();
  assert.match(manager.entries[0].error, /Upload unconfirmed/);
  manager.retry("file-1");
  manager.retry("file-1");
  assert.equal(requests.length, 2);
  assert.equal(requests[1].path, "/agents/agent/assets/file-1");
  assert.equal(requests[1].body, undefined);
  requests[1].resolve(asset("file-1"));
  await tick();
  assert.deepEqual(manager.ids(), ["file-1"]);
  assert.equal(requests.length, 2);
});

test("a confirmed missing upload retries retained bytes using the original ID", async () => {
  const { manager, requests } = setup();
  manager.add([file()]);
  requests[0].reject(new Error("lost"));
  await tick();
  manager.retry("file-1");
  requests[1].reject(new ApiError("missing", 404));
  await tick();
  assert.equal(requests[2].body.get("asset_id"), "file-1");
  requests[2].resolve(asset("file-1"));
  await tick();
  assert.equal(manager.ready, true);
});

test("reload recovers saved uploads but requires original bytes when an unconfirmed upload is missing", async () => {
  const first = setup();
  first.manager.add([file(), file("second.txt")]);
  first.manager.stop();
  const next = setup(first.saved);
  assert.equal(next.requests[0].body, undefined);
  next.requests[0].resolve(asset("file-1"));
  await tick();
  next.requests[1].reject(new ApiError("missing", 404));
  await tick();
  assert.equal(next.manager.entries[1].status, "reselect");
  assert.equal(next.manager.ready, false);
  assert.throws(() => next.manager.reselect("file-2", file("wrong.txt")), /Reselect the original/);
  next.manager.reselect("file-2", file("second.txt"));
  next.requests[2].reject(new ApiError("missing", 404));
  await tick();
  assert.equal(next.requests[3].body.get("asset_id"), "file-2");
  next.requests[3].resolve(asset("file-2", "second.txt"));
  await tick();
  assert.deepEqual(next.manager.ids(), ["file-1", "file-2"]);
});

test("failed recovery reads and mismatched or deleted assets never count as successful uploads", async () => {
  for (const result of [new ApiError("unavailable", 503), { ...asset("file-1"), agent_key: "other" },
    asset("other"), { ...asset("file-1"), deleted_at: undefined }, { ...asset("file-1"), size_bytes: 4 }, { ...asset("file-1"), deleted_at: "today" }]) {
    const { manager, requests } = setup();
    manager.add([file()]);
    requests[0].reject(new Error("lost"));
    await tick();
    manager.retry("file-1");
    if (result instanceof Error) requests[1].reject(result); else requests[1].resolve(result);
    await tick();
    assert.equal(manager.entries[0].status, "error");
    assert.equal(manager.ready, false);
    assert.equal(requests.length, 2);
  }
});

test("remove detaches locally and a late upload cannot resurrect a removed file", async () => {
  const { manager, requests, saved } = setup();
  manager.add([file()]);
  manager.remove("file-1");
  requests[0].resolve(asset("file-1"));
  await tick();
  assert.deepEqual(manager.entries, []);
  assert.equal(saved.size, 0);
  assert.equal(requests.length, 1);
});

test("stopping ignores late callbacks and drafts stay scoped to agent and conversation", async () => {
  const first = setup();
  first.manager.add([file()]);
  const before = first.changes();
  first.manager.stop();
  assert.deepEqual(setup(first.saved, "agent", "other").manager.entries, []);
  assert.deepEqual(setup(first.saved, "other", "chat").manager.entries, []);
  first.requests[0].resolve(asset("file-1"));
  await tick();
  assert.equal(first.changes(), before);
  assert.equal(first.manager.entries[0].status, "uploading");
  first.manager.start();
  assert.equal(first.requests[1].body, undefined);
  first.requests[1].resolve(asset("file-1"));
  await tick();
  assert.deepEqual(first.manager.ids(), ["file-1"]);
});

test("selection count and aggregate byte limits reject the entire addition", () => {
  const { manager, requests } = setup();
  assert.throws(() => manager.add(Array.from({ length: 11 }, () => file())), /10 files/);
  assert.throws(() => manager.add([{ name: "large", size: MAX_ATTACHMENT_BYTES + 1 }]), /100 MiB/);
  manager.add([file()]);
  assert.throws(() => manager.add([{ name: "large", size: MAX_ATTACHMENT_BYTES }]), /100 MiB/);
  assert.equal(manager.entries.length, 1);
  assert.equal(requests.length, 1);
});
