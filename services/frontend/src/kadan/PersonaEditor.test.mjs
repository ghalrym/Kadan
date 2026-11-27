import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { runInNewContext } from "node:vm";
import ts from "typescript";

const source = ts.transpileModule(
  readFileSync(new URL("./PersonaEditor.tsx", import.meta.url), "utf8"),
  { compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX } },
).outputText;
const settle = () => new Promise(resolve => setImmediate(resolve));

async function openEditor(responses = []) {
  const slots = [], effects = [], requests = [];
  let cursor = 0, tree, closed = 0, confirmed = true;
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
    "lucide-react": { Check: "Check", Eye: "Eye", Pencil: "Pencil", X: "X" },
    "./Markdown": { default: "Markdown" },
    "./api": {
      agentPath: key => `/agents/${key}`,
      async api(path, body, method) {
        requests.push({ path, body, method });
        const next = responses.shift();
        if (next instanceof Error) throw next;
        return next;
      },
    },
  };
  const exports = {};
  runInNewContext(source, {
    exports,
    window: { confirm: () => confirmed },
    require: name => { assert.ok(name in modules, name); return modules[name]; },
  });
  function render() {
    cursor = 0;
    tree = exports.default({ agentKey: "mi", agentName: "Mi", onClose: () => closed++ });
    effects.splice(0).forEach(effect => effect());
  }
  function elements(node = tree) {
    if (!node || typeof node !== "object") return [];
    return [node, ...[node.props?.children].flat(Infinity).flatMap(child => elements(child ?? null))];
  }
  render();
  await settle();
  render();
  return {
    render, requests, elements,
    closed: () => closed,
    setConfirm: value => { confirmed = value; },
    find: predicate => elements().find(predicate),
  };
}

test("loads the agent's single persona, switches between Write and Preview, and saves via PUT", async () => {
  const initial = { id: "p1", title: "Mi", markdown: "# Mi\nCalm and brief.", version: 2 };
  const saved = { ...initial, markdown: "# Mi\nDirect and warm.", version: 3 };
  const view = await openEditor([initial, saved]);

  assert.deepEqual(view.requests[0], { path: "/agents/mi/persona", body: undefined, method: undefined });
  const textarea = view.find(node => node.type === "textarea");
  assert.equal(textarea.props.value, "# Mi\nCalm and brief.");

  textarea.props.onChange({ target: { value: "# Mi\nDirect and warm." } });
  view.render();

  const previewBtn = view.find(node => node.type === "button" && Array.isArray(node.props.children) && node.props.children.includes("Preview"));
  previewBtn.props.onClick();
  view.render();
  assert.equal(view.find(node => node.type === "Markdown").props.children, "# Mi\nDirect and warm.");

  const form = view.find(node => node.type === "form");
  await form.props.onSubmit({ preventDefault() {} });
  view.render();
  assert.deepEqual(JSON.parse(JSON.stringify(view.requests[1])), {
    path: "/agents/mi/persona",
    body: { markdown: "# Mi\nDirect and warm.", expected_version: 2 },
    method: "PUT",
  });
  assert.equal(view.find(node => node.props?.role === "status").props.children, "Saved");
});

test("conflict reload shows the latest saved version without discarding the draft", async () => {
  const initial = { id: "p1", title: "Mi", markdown: "Original", version: 1 };
  const remote = { id: "p1", title: "Mi", markdown: "Updated elsewhere", version: 2 };
  const view = await openEditor([initial, new Error("Version conflict"), remote]);

  view.find(node => node.type === "textarea").props.onChange({ target: { value: "My draft" } });
  view.render();
  await view.find(node => node.type === "form").props.onSubmit({ preventDefault() {} });
  view.render();

  const alert = view.find(node => node.props?.role === "alert");
  assert.ok(alert);
  const reloadBtn = view.elements(alert).find(node => node.type === "button");
  reloadBtn.props.onClick();
  await settle();
  view.render();

  assert.equal(view.find(node => node.type === "textarea").props.value, "My draft");
  assert.equal(view.find(node => node.type === "Markdown").props.children, "Updated elsewhere");
});
