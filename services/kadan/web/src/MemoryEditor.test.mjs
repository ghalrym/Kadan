import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { runInNewContext } from "node:vm";
import ts from "typescript";

const source = ts.transpileModule(readFileSync(new URL("./MemoryEditor.tsx", import.meta.url), "utf8"), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX },
}).outputText;
const memory = {
  id: "memory-one", agent_key: "guide", kind: "memory", title: "Morning",
  markdown: "Coffee", version: 2, created_at: "2026-10-25T10:00:00Z", updated_at: "2026-10-25T10:00:00Z",
  sources: [{ label: "Original note", url: "https://example.org/note" }],
};
const key = `kadan.memory-draft.guide.${memory.id}`;

function editor({ draft, load, write, confirmation = true } = {}) {
  const slots = [], effects = [], requests = [], confirmed = [], listeners = new Map();
  const storage = new Map(draft ? [[key, JSON.stringify(draft)]] : []);
  let cursor = 0, tree, selected = memory;
  const react = {
    useState(initial) {
      const i = cursor++;
      if (!(i in slots)) slots[i] = typeof initial === "function" ? initial() : initial;
      return [slots[i], value => { slots[i] = typeof value === "function" ? value(slots[i]) : value; }];
    },
    useRef(initial) { return slots[cursor++] ??= { current: initial }; },
    useEffect(effect, dependencies) {
      const i = cursor++, previous = slots[i];
      if (!previous || dependencies.some((value, n) => value !== previous.dependencies[n])) {
        const slot = slots[i] = { dependencies };
        effects.push(() => { previous?.cleanup?.(); slot.cleanup = effect(); });
      }
    },
  };
  const jsx = (type, props) => ({ type, props });
  const modules = {
    react, "react/jsx-runtime": { jsx, jsxs: jsx }, "./Markdown": { default: "Markdown" },
    "./api": {
      agentPath: key => `/agents/${key}`,
      async api(path, body, method) {
        requests.push({ path, body, method });
        if (!body) return load ? load() : selected;
        return write ? write(body) : { ...selected, ...body, version: body.expected_version + 1 };
      },
    },
  };
  const exports = {};
  runInNewContext(source, {
    exports, require: name => { assert.ok(name in modules, name); return modules[name]; },
    sessionStorage: {
      getItem: key => storage.get(key) ?? null, setItem: (key, value) => storage.set(key, value), removeItem: key => storage.delete(key),
    },
    window: { confirm: () => confirmation, addEventListener: (name, handler) => listeners.set(name, handler), removeEventListener: name => listeners.delete(name) },
  });
  function render() {
    cursor = 0;
    tree = exports.default({ agentKey: "guide", memory: selected, onSaved: value => { selected = value; confirmed.push(value); } });
    while (effects.length) effects.shift()();
    return tree;
  }
  render();
  function nodes(node) {
    if (!node || typeof node !== "object") return [];
    const children = Array.isArray(node) ? node : [node.props?.children];
    return [node, ...children.flatMap(child => nodes(child))];
  }
  function button(label) {
    const value = nodes(tree).find(node => node.type === "button" && node.props.children === label);
    assert.ok(value, label);
    return value.props;
  }
  function text() {
    return JSON.stringify(tree);
  }
  return {
    render, requests, confirmed, storage, listeners, text, button,
    receive(value) { selected = value; render(); render(); },
    markdownLimit: () => nodes(tree).find(node => node.type === "textarea").props.maxLength,
    async edit() { await button("Correct memory").onClick(); await new Promise(resolve => setImmediate(resolve)); render(); },
    change(title, markdown) {
      nodes(tree).find(node => node.type === "input").props.onChange({ target: { value: title } }); render();
      nodes(tree).find(node => node.type === "textarea").props.onChange({ target: { value: markdown } }); render();
    },
    async save() {
      await nodes(tree).find(node => node.type === "form").props.onSubmit({ preventDefault() {} }); render();
    },
    async check() { await button("Check saved version").onClick(); await new Promise(resolve => setImmediate(resolve)); render(); },
    unmount() { slots.forEach(slot => slot?.cleanup?.()); },
  };
}

test("memory correction reads current record and saves only edited text at its version", async () => {
  const ui = editor();
  assert.match(ui.text(), /Original note/);
  await ui.edit();
  ui.change("Updated", "Tea");
  await ui.save();
  assert.equal(ui.requests.length, 2);
  assert.equal(ui.requests[0].path, "/agents/guide/memories/memory-one");
  assert.deepEqual(JSON.parse(JSON.stringify(ui.requests[1])), {
    path: "/agents/guide/memories/memory-one", method: "PUT",
    body: { title: "Updated", markdown: "Tea", expected_version: 2 },
  });
  assert.match(ui.text(), /Correction saved/);
  assert.match(ui.text(), /future recall is not guaranteed/);
  assert.equal(ui.storage.has(key), false);
});

test("failed save retains correction and prohibits retry until a successful read", async () => {
  let latest = memory, readsFail = false;
  const ui = editor({ load: () => { if (readsFail) throw new Error("Read unavailable"); return latest; }, write: () => { throw new Error("Conflict"); } });
  await ui.edit();
  ui.change("Correction", "Tea");
  await ui.save();
  assert.equal(ui.button("Save correction").disabled, true);
  assert.match(ui.text(), /Correction/);
  assert.match(ui.text(), /may already have been saved/);
  await ui.save();
  assert.equal(ui.requests.filter(request => request.body).length, 1);
  readsFail = true;
  await ui.check();
  assert.equal(ui.button("Save correction").disabled, true);
  readsFail = false;
  latest = { ...memory, version: 4, markdown: "Someone else's correction" };
  await ui.check();
  assert.equal(ui.button("Save correction").disabled, false);
  assert.match(ui.text(), /Someone else's correction/);
  assert.match(ui.text(), /Tea/);
  assert.equal(JSON.parse(ui.storage.get(key)).base.version, 4);
});

test("lost write acknowledgement is settled by reading matching saved text without a second write", async () => {
  let latest = memory;
  const ui = editor({ load: () => latest, write: body => {
    latest = { ...memory, ...body, version: 3 };
    throw new Error("Response lost");
  } });
  await ui.edit(); ui.change("Corrected", "Tea"); await ui.save(); await ui.check();
  assert.match(ui.text(), /No further save is needed/);
  assert.equal(ui.button("Save correction").disabled, true);
  ui.button("Done").onClick(); ui.render();
  assert.equal(ui.requests.filter(request => request.body).length, 1);
  assert.equal(ui.storage.has(key), false);
});

test("restored correction requires checking storage and preserves its text", async () => {
  const ui = editor({ draft: { base: memory, title: "Corrected", markdown: "Tea" } });
  assert.equal(ui.button("Save correction").disabled, true);
  await ui.save(); assert.equal(ui.requests.length, 0);
  await ui.check();
  assert.match(ui.text(), /Tea/);
  assert.equal(ui.button("Save correction").disabled, false);
});

test("discard confirmation can retain draft and successful discard removes it", async () => {
  const keep = editor({ confirmation: false });
  await keep.edit(); keep.change("Corrected", "Tea");
  keep.button("Discard correction").onClick(); keep.render();
  assert.equal(keep.storage.has(key), true);
  const discard = editor({ draft: { base: memory, title: "Corrected", markdown: "Tea" } });
  discard.button("Discard correction").onClick(); discard.render();
  assert.equal(discard.storage.has(key), false);
});

test("leaving while a save completes does not overwrite another view or erase recovery", async () => {
  let complete;
  const ui = editor({ write: () => new Promise(resolve => { complete = resolve; }) });
  await ui.edit(); ui.change("Corrected", "Tea");
  const saving = ui.save();
  ui.unmount();
  complete({ ...memory, version: 3, title: "Corrected", markdown: "Tea" });
  await saving;
  assert.equal(ui.confirmed.length, 1); // Initial read only.
  assert.equal(JSON.parse(ui.storage.get(key)).markdown, "Tea");
});

test("an unavailable initial read does not open an editor or write a correction", async () => {
  const ui = editor({ load: () => { throw new Error("Memory unavailable"); } });
  await ui.edit();
  assert.match(ui.text(), /Memory unavailable/);
  assert.ok(ui.button("Correct memory"));
  assert.equal(ui.storage.has(key), false);
});


test("title corrections preserve an existing memory beyond runner text limits", async () => {
  const largeMemory = { ...memory, markdown: "x".repeat(48001) };
  const ui = editor({ load: () => largeMemory });
  await ui.edit();
  assert.equal(ui.markdownLimit(), 2000000);
  ui.change("Updated title", largeMemory.markdown);
  await ui.save();
  assert.equal(ui.requests[1].body.markdown, largeMemory.markdown);
  assert.match(ui.text(), /Correction saved/);
});


test("Workspace refresh updates displayed memory while preserving an unsaved correction's version", async () => {
  const ui = editor();
  ui.receive({ ...memory, markdown: "Refreshed memory", version: 3 });
  assert.match(ui.text(), /Refreshed memory/);
  await ui.edit();
  ui.change("Correction", "Tea");
  ui.receive({ ...memory, markdown: "External correction", version: 4 });
  assert.match(ui.text(), /Tea/);
  await ui.save();
  assert.equal(ui.requests[1].body.expected_version, 3);
});
