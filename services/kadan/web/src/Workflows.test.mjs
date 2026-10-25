import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { runInNewContext } from "node:vm";
import ts from "typescript";
import * as workflows from "./workflows.ts";

const source = ts.transpileModule(readFileSync(new URL("./Workflows.tsx", import.meta.url), "utf8"),
  { compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX } }).outputText;
const selected = { id: "a", version: 3, title: "Review", markdown: "Inspect the current task.", kind: "space", tags: ["workflow"] };
const settle = () => new Promise(resolve => setImmediate(resolve));
const deferred = () => { let resolve, reject; const promise = new Promise((yes, no) => { resolve = yes; reject = no; }); return { promise, resolve, reject }; };

function panel(api = async () => { throw new Error("Unexpected request"); }, props = {}) {
  const slots = [], requests = [], runs = [];
  let cursor = 0, tree;
  const modules = {
    react: {
      useState(initial) {
        const index = cursor++;
        if (!(index in slots)) slots[index] = typeof initial === "function" ? initial() : initial;
        return [slots[index], value => { slots[index] = typeof value === "function" ? value(slots[index]) : value; }];
      },
      useRef(initial) { const index = cursor++; return slots[index] ??= { current: initial }; },
    },
    "react/jsx-runtime": { jsx: (type, props) => ({ type, props }), jsxs: (type, props) => ({ type, props }) },
    "./api": { agentPath: key => `/agents/${key}`, api: async (path, body) => { requests.push({ path, body }); return api(path, body); } },
    "./ContextCheck": { default: "ContextCheck" },
    "./Markdown": { default: "Markdown" },
    "./workflows": workflows,
  };
  const exports = {};
  runInNewContext(source, { exports, require: name => { assert.ok(name in modules, name); return modules[name]; } });
  function render() {
    cursor = 0;
    tree = exports.default({ agentKey: "alpha", task: "Review this new change", disabled: false,
      browserUrl: null, onRun: value => runs.push(value), ...props });
  }
  function nodes(node = tree) {
    if (!node || typeof node !== "object") return [];
    return [node, ...[node.props?.children].flat(Infinity).flatMap(child => nodes(child ?? null))];
  }
  const button = name => nodes().find(node => node.type === "button" && node.props.children === name)?.props;
  const select = () => nodes().find(node => node.type === "select").props;
  render();
  return { render, nodes, requests, runs, button, select,
    async import(value) {
      const input = nodes().find(node => node.type === "input" && node.props.type === "file");
      input.props.onChange({ target: { files: [{ size: 100, text: async () => JSON.stringify(value) }] } });
      await settle(); render();
    },
    async load() { await button("Reload list").onClick(); await settle(); render(); },
  };
}

const bundle = { schema_version: 1, title: "Imported review", instructions: "Read this entire imported method.\nDo only the current task." };

test("import previews full instructions; only explicit Save creates a Space and never runs", async () => {
  const ui = panel(async (path, body) => ({ ...selected, title: body.title, markdown: body.instructions, version: 1 }));
  await ui.import(bundle);
  assert.equal(ui.requests.length, 0);
  assert.equal(ui.runs.length, 0);
  assert.ok(ui.nodes().some(node => node.type === "pre" && node.props.children === bundle.instructions));
  ui.button("Save workflow").onClick();
  await settle(); ui.render();
  assert.equal(ui.requests.length, 1);
  assert.deepEqual(ui.requests[0].body, bundle);
  assert.equal(ui.runs.length, 0);
  assert.ok(ui.button("Run now"));
});

test("rapid Save clicks create once; uncertain failure requires a successful list check", async () => {
  const create = deferred();
  const ui = panel(async (path, body) => body ? create.promise : { items: [], next_offset: null });
  await ui.import(bundle);
  const save = ui.button("Save workflow").onClick;
  save(); save();
  assert.equal(ui.requests.length, 1);
  create.reject(new Error("Response lost"));
  await settle(); ui.render();
  assert.equal(ui.button("Save workflow").disabled, true);
  ui.button("Save workflow").onClick();
  assert.equal(ui.requests.length, 1);
  await ui.load();
  assert.equal(ui.button("Save workflow").disabled, false);
  assert.equal(ui.runs.length, 0);
});

test("latest selection wins delayed responses; Run now requires its explicit click", async () => {
  const first = deferred(), second = deferred();
  const ui = panel(path => path.endsWith("/a") ? first.promise : second.promise);
  const choose = ui.select().onChange;
  choose({ target: { value: "a" } });
  choose({ target: { value: "b" } });
  second.resolve({ ...selected, id: "b", title: "Latest" });
  await settle(); ui.render();
  first.resolve(selected);
  await settle(); ui.render();
  assert.equal(ui.select().value, "b");
  assert.equal(ui.runs.length, 0);
  ui.button("Run now").onClick();
  assert.deepEqual(JSON.parse(JSON.stringify(ui.runs)), [{ id: "b", version: 3 }]);
});

test("failed reload clears stale selectable workflows", async () => {
  let failed = false;
  const ui = panel(async () => { if (failed) throw new Error("Unavailable"); return { items: [selected], next_offset: null }; });
  await ui.load();
  assert.equal(ui.nodes().filter(node => node.type === "option").length, 2);
  failed = true;
  await ui.load();
  assert.equal(ui.nodes().filter(node => node.type === "option").length, 1);
  assert.equal(ui.button("Run now"), undefined);
});


test("templates can be imported and saved while inference is unavailable", async () => {
  const ui = panel(async (path, body) => ({ ...selected, title: body.title, markdown: body.instructions, version: 1 }), { runDisabled: true });
  await ui.import(bundle);
  assert.equal(ui.button("Save workflow").disabled, false);
  ui.button("Save workflow").onClick();
  await settle(); ui.render();
  assert.equal(ui.requests.length, 1);
  assert.equal(ui.button("Run now").disabled, true);
  assert.equal(ui.runs.length, 0);
});
