import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { runInNewContext } from "node:vm";
import ts from "typescript";

const source = ts.transpileModule(readFileSync(new URL("./RequestReferences.tsx", import.meta.url), "utf8"),
  { compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX } }).outputText;
const saved = (id, kind = "space", agent_key = "alpha", deleted_at = null) => ({
  id, kind, agent_key, deleted_at, version: 5, title: `Title ${id}`, markdown: "Saved text",
});
function picker(resources, initial = []) {
  let cursor = 0, tree, selected = initial, agentKey = "alpha", disabled = false;
  const effects = [], cleanups = [], dependencies = [];
  const slots = [];
  const timers = new Map(), listeners = new Map();
  let timerId = 0;
  const container = { contains: node => node?.inside === true };
  const jsx = (type, props) => ({ type, props });
  const modules = {
    react: { useEffect(effect, deps) {
      const index = cursor++;
      if (!dependencies[index] || deps.some((value, i) => value !== dependencies[index][i])) {
        dependencies[index] = deps;
        effects.push(() => { cleanups[index]?.(); cleanups[index] = effect(); });
      }
    }, useRef(initial) {
      const index = cursor++;
      return slots[index] ??= { current: initial };
    }, useState(initial) {
      const index = cursor++;
      if (!(index in slots)) slots[index] = initial;
      return [slots[index], value => { slots[index] = value; }];
    } },
    "react/jsx-runtime": { jsx, jsxs: jsx },
    "lucide-react": new Proxy({}, { get: (_, name) => name }),
    "./api": { agentPath: key => `/agents/${key}`, api: async path => typeof resources === "function" ?
      resources(path) : resources.filter(item => path.includes("/memories?") ? item.kind === "memory" : item.kind !== "memory") },
  };
  const exports = {};
  runInNewContext(source, {
    exports, require: name => modules[name],
    document: {
      addEventListener: (type, listener) => listeners.set(type, listener),
      removeEventListener: (type, listener) => { if (listeners.get(type) === listener) listeners.delete(type); },
    },
    setTimeout: callback => { timers.set(++timerId, callback); return timerId; },
    clearTimeout: id => timers.delete(id),
  });
  function render() {
    cursor = 0;
    tree = exports.default({ agentKey, selected, disabled,
      onChange: value => { selected = value; } });
    if (tree.props.ref) tree.props.ref.current = container;
    if (effects.length) { effects.splice(0).forEach(effect => effect()); render(); }
  }
  function elements(node = tree) {
    if (!node || typeof node !== "object") return [];
    return [node, ...[node.props?.children].flat(Infinity).flatMap(child => elements(child ?? null))];
  }
  render();
  return { render, elements, selected: () => selected,
    query(value) {
      elements().find(item => item.props?.["aria-label"] === "Search references").props.onChange({ target: { value } });
      render();
    },
    start() {
      const callbacks = [...timers.values()];
      timers.clear();
      callbacks.forEach(callback => callback());
      render();
    },
    pointer(inside) { listeners.get("pointerdown")?.({ target: { inside } }); render(); },
    blur(relatedTarget) { tree.props.onBlur({ currentTarget: container, relatedTarget }); render(); },
    escape() { tree.props.onKeyDown({ key: "Escape", stopPropagation() {} }); render(); },
    listening: () => listeners.size,
    switchAgent(key) { agentKey = key; selected = []; render(); },
    disable() { disabled = true; render(); },
    unmount() { cleanups.forEach(cleanup => cleanup?.()); },
    search() { elements().find(item => item.type === "form").props.onSubmit({ preventDefault() {} }); render(); },
    async open() { elements().find(item => item.type === "button").props.onClick();
      await new Promise(resolve => setImmediate(resolve)); render(); },
  };
}

test("picker lists only active same-agent Markdown resources and selects explicit versions", async () => {
  const view = picker([saved("plan"), saved("fact", "memory"), saved("deck", "deck"),
    saved("other", "space", "beta"), saved("trash", "memory", "alpha", "today")]);
  await view.open();
  const choices = view.elements().filter(item => item.type === "input" && item.props.type === "checkbox");
  assert.equal(choices.length, 2);
  choices[0].props.onChange();
  assert.deepEqual(JSON.parse(JSON.stringify(view.selected())),
    [{ resource_id: "plan", expected_version: 5, title: "Title plan", characters: 10 }]);
});

test("internal pointer blur preserves results for label activation", async () => {
  const view = picker([saved("plan")]);
  await view.open();
  view.pointer(true);
  view.blur(null);
  assert.equal(choices(view).length, 1);
  choices(view)[0].props.onChange();
  view.render();
  assert.equal(view.selected()[0].resource_id, "plan");
  assert.equal(choices(view)[0].props.checked, true);
});

test("internal focus stays open and outside pointer, keyboard focus and Escape close without clearing selection", async () => {
  const initial = [{ resource_id: "plan", expected_version: 5, title: "Title plan" }];
  for (const leave of [view => view.pointer(false), view => view.blur({ inside: false }), view => view.escape()]) {
    const view = picker([saved("plan")], initial);
    await view.open();
    view.blur({ inside: true });
    assert.equal(choices(view).length, 1);
    leave(view);
    assert.equal(choices(view).length, 0);
    assert.deepEqual(view.selected(), initial);
    assert.equal(view.listening(), 0);
  }
  const view = picker([]);
  await view.open();
  assert.equal(view.listening(), 1);
  view.unmount();
  assert.equal(view.listening(), 0);
});

test("query changes discard pending results and failures before dispatching the debounced search", async () => {
  for (const fail of [false, true]) {
    const pending = [];
    const view = picker(path => { const value = deferred(); pending.push({ ...value, path }); return value.promise; });
    await view.open();
    view.query("invoice");
    assert.equal(pending.length, 2);
    assert.equal(choices(view).length, 0);
    if (fail) pending[0].reject(new Error("Old search failure"));
    else pending[0].resolve([saved("old-plan")]);
    pending[1].resolve([]);
    await new Promise(resolve => setImmediate(resolve)); view.render();
    assert.equal(choices(view).length, 0);
    assert.ok(text(view).includes("Loading references"));
    assert.ok(!text(view).includes("Old search failure"));
    view.start();
    assert.equal(pending.length, 4);
    assert.ok(pending.slice(2).every(item => item.path.endsWith("q=invoice")));
    pending[2].resolve([saved("invoice")]); pending[3].resolve([]);
    await new Promise(resolve => setImmediate(resolve)); view.render();
    choices(view)[0].props.onChange();
    assert.equal(view.selected()[0].resource_id, "invoice");
  }
});

test("query changes clear loaded rows and still search when reverting to the last query", async () => {
  const paths = [];
  const view = picker(path => {
    paths.push(path);
    return path.includes("/workspace/documents?") && path.endsWith("q=") ? [saved("plan")] : [];
  });
  await view.open();
  assert.equal(choices(view).length, 1);
  view.query("invoice");
  assert.equal(choices(view).length, 0);
  view.query("");
  view.start();
  await new Promise(resolve => setImmediate(resolve)); view.render();
  assert.equal(paths.length, 4);
  assert.ok(paths.every(path => path.endsWith("q=")));
  assert.equal(choices(view).length, 1);
});

test("submitting before the debounce dispatches once and closing cancels queued searches", async () => {
  const paths = [];
  const view = picker(path => { paths.push(path); return []; });
  await view.open();
  view.query("invoice");
  view.search();
  view.start();
  assert.equal(paths.length, 4);
  view.query("cancelled");
  view.escape();
  view.start();
  assert.equal(paths.length, 4);
  view.unmount();
});

test("picker disables a fourth selection and permits removing selected items", async () => {
  const resources = [saved("one"), saved("two"), saved("three"), saved("four")];
  const initial = resources.slice(0, 3).map(item => ({ resource_id: item.id, expected_version: item.version, title: item.title }));
  const view = picker(resources, initial);
  await view.open();
  const choices = view.elements().filter(item => item.type === "input" && item.props.type === "checkbox");
  assert.equal(choices[3].props.disabled, true);
  assert.equal(choices[0].props.disabled, false);
  choices[0].props.onChange();
  assert.equal(view.selected().length, 2);
});


const choices = view => view.elements().filter(item => item.type === "input" && item.props.type === "checkbox");
const text = view => JSON.stringify(view.elements().map(item => item.props?.children));
const deferred = () => { let resolve, reject; const promise = new Promise((done, fail) => { resolve = done; reject = fail; }); return { promise, resolve, reject }; };

test("refresh clears stale rows, ignores older results and leaves no stale choices on failure", async () => {
  const pending = [];
  const view = picker(() => { const value = deferred(); pending.push(value); return value.promise; });
  await view.open();
  pending[0].resolve([saved("initial")]); pending[1].resolve([]);
  await new Promise(resolve => setImmediate(resolve)); view.render();
  assert.equal(choices(view).length, 1);
  view.search();
  assert.equal(choices(view).length, 0);
  view.search();
  pending[4].resolve([saved("newer")]); pending[5].resolve([]);
  await new Promise(resolve => setImmediate(resolve)); view.render();
  pending[2].resolve([saved("older")]); pending[3].resolve([]);
  await new Promise(resolve => setImmediate(resolve)); view.render();
  choices(view)[0].props.onChange();
  assert.equal(view.selected()[0].resource_id, "newer");
  view.search();
  assert.equal(choices(view).length, 0);
  pending[6].reject(new Error("Unavailable")); pending[7].resolve([]);
  await new Promise(resolve => setImmediate(resolve)); view.render();
  assert.equal(choices(view).length, 0);
  assert.ok(text(view).includes("Unavailable"));
});

test("source page boundary signals truncated results even when filtering removes rows", async () => {
  const view = picker(Array.from({ length: 200 }, (_, i) => saved(`other-${i}`, "space", "beta")));
  await view.open();
  assert.equal(choices(view).length, 0);
  assert.ok(text(view).includes("Search to narrow"));
  assert.ok(text(view).includes("12,000"));
  const below = picker(Array.from({ length: 199 }, (_, i) => saved(`other-${i}`, "space", "beta")));
  await below.open();
  assert.ok(!text(below).includes("Search to narrow"));
});

test("changing agents fences late success and failure from the previous picker", async () => {
  for (const fail of [false, true]) {
    const pending = [];
    const view = picker(() => { const value = deferred(); pending.push(value); return value.promise; });
    await view.open();
    view.switchAgent("beta");
    await view.open();
    pending[2].resolve([saved("current", "space", "beta")]); pending[3].resolve([]);
    await new Promise(resolve => setImmediate(resolve)); view.render();
    if (fail) pending[0].reject(new Error("Old failure"));
    else pending[0].resolve([saved("old")]);
    pending[1].resolve([]);
    await new Promise(resolve => setImmediate(resolve)); view.render();
    assert.equal(choices(view).length, 1);
    assert.ok(!text(view).includes("Old failure"));
    choices(view)[0].props.onChange();
    assert.equal(view.selected()[0].resource_id, "current");
  }
});

test("closing the picker invalidates its pending read and disabled search submits nothing", async () => {
  const pending = [];
  const view = picker(() => { const value = deferred(); pending.push(value); return value.promise; });
  await view.open();
  await view.open();
  pending[0].resolve([saved("old")]); pending[1].resolve([]);
  await new Promise(resolve => setImmediate(resolve)); view.render();
  assert.equal(choices(view).length, 0);
  await view.open();
  view.disable();
  view.search();
  assert.equal(pending.length, 4);
  view.unmount();
  pending[2].resolve([]); pending[3].resolve([]);
});
