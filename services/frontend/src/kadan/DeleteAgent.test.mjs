import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { runInNewContext } from "node:vm";
import ts from "typescript";

const source = ts.transpileModule(
  readFileSync(new URL("./DeleteAgent.tsx", import.meta.url), "utf8"),
  { compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX } },
).outputText;

function openDialog(agent = { key: "mi", name: "Mi" }, apiImpl = async () => undefined) {
  const slots = [], effects = [], requests = [];
  const storage = new Map([
    ["kadan.draft.mi", "unsent"],
    ["kadan.memory-draft.mi.m1", "edit"],
    ["kadan.draft.other", "keep"],
  ]);
  let cursor = 0, tree, deleted = 0, closed = 0;
  const react = {
    useState(initial) {
      const index = cursor++;
      if (!(index in slots)) slots[index] = typeof initial === "function" ? initial() : initial;
      return [slots[index], value => {
        slots[index] = typeof value === "function" ? value(slots[index]) : value;
      }];
    },
    useRef(initial) {
      const index = cursor++;
      return slots[index] ??= { current: initial };
    },
    useEffect(effect, dependencies) {
      const index = cursor++;
      const previous = slots[index];
      if (!previous || dependencies.some((value, i) => value !== previous.dependencies[i])) {
        const slot = slots[index] = { dependencies };
        effects.push(() => { previous?.cleanup?.(); slot.cleanup = effect(); });
      }
    },
  };
  const jsx = (type, props) => ({ type, props });
  const modules = {
    react,
    "react/jsx-runtime": { jsx, jsxs: jsx },
    "lucide-react": { Trash2: "Trash2" },
    "./api": {
      agentPath: key => `/agents/${key}`,
      async api(path, body, method) {
        requests.push({ path, body, method });
        return apiImpl(path, body, method);
      },
    },
  };
  const exports = {};
  runInNewContext(source, {
    exports,
    sessionStorage: {
      get length() { return storage.size; },
      key: i => Array.from(storage.keys())[i] ?? null,
      removeItem: k => storage.delete(k),
    },
    require: name => { assert.ok(name in modules, name); return modules[name]; },
  });
  function render() {
    cursor = 0;
    tree = exports.default({ agent, onDeleted: () => deleted++, onClose: () => closed++ });
    effects.splice(0).forEach(effect => effect());
  }
  function elements(node = tree) {
    if (!node || typeof node !== "object") return [];
    return [node, ...[node.props?.children].flat(Infinity).flatMap(child => elements(child ?? null))];
  }
  render();
  return {
    render, requests, storage,
    deleted: () => deleted,
    closed: () => closed,
    find: predicate => elements().find(predicate),
  };
}

test("delete stays disabled until the user types '<agent name> Delete', then sends DELETE and clears local drafts", async () => {
  const view = openDialog({ key: "mi", name: "Mi" });
  const submitBtn = () => view.find(node => node.props?.className === "danger-primary");
  const input = () => view.find(node => node.type === "input");

  assert.equal(submitBtn().props.disabled, true);

  input().props.onChange({ target: { value: "Mi" } });
  view.render();
  assert.equal(submitBtn().props.disabled, true);

  input().props.onChange({ target: { value: "Mi delete" } });
  view.render();
  assert.equal(submitBtn().props.disabled, true);

  input().props.onChange({ target: { value: "Mi Delete" } });
  view.render();
  assert.equal(submitBtn().props.disabled, false);

  await view.find(node => node.type === "form").props.onSubmit({ preventDefault() {} });
  view.render();

  assert.deepEqual(JSON.parse(JSON.stringify(view.requests)), [
    { path: "/agents/mi", method: "DELETE" },
  ]);
  assert.equal(view.deleted(), 1);
  assert.equal(view.closed(), 1);
  assert.equal(view.storage.has("kadan.draft.mi"), false);
  assert.equal(view.storage.has("kadan.memory-draft.mi.m1"), false);
  assert.equal(view.storage.get("kadan.draft.other"), "keep");
});

test("delete failure surfaces an alert without closing the dialog", async () => {
  const view = openDialog({ key: "mi", name: "Mi" }, async () => { throw new Error("Server offline"); });
  view.find(node => node.type === "input").props.onChange({ target: { value: "Mi Delete" } });
  view.render();
  await view.find(node => node.type === "form").props.onSubmit({ preventDefault() {} });
  view.render();
  assert.equal(view.deleted(), 0);
  assert.equal(view.closed(), 0);
  assert.equal(view.find(node => node.props?.role === "alert").props.children, "Server offline");
});
