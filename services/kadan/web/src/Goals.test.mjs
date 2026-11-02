import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { runInNewContext } from "node:vm";
import ts from "typescript";

const source = ts.transpileModule(readFileSync(new URL("./Goals.tsx", import.meta.url), "utf8"), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX },
}).outputText;
const exports = {};
const jsx = (type, props) => ({ type, props });
runInNewContext(source, {
  exports,
  require: name => {
    if (name === "react") return {};
    if (name === "./api") return {};
    assert.equal(name, "react/jsx-runtime");
    return { jsx, jsxs: jsx };
  },
});
function elements(node) {
  if (!node || typeof node !== "object") return [];
  return [node, ...[node.props?.children].flat(Infinity).flatMap(child => elements(child))];
}
function text(node) {
  if (node == null || typeof node === "boolean") return "";
  if (Array.isArray(node)) return node.map(text).join("");
  return typeof node === "object" ? text(node.props?.children) : String(node);
}
const task = extra => ({ id: "goal", status: "active", goal: "Prepare the proposal", progress: "Outline saved",
  next_step: "Draft the budget", blocker: null, wait_kind: null, wake_at: null, ...extra });
const view = extra => exports.GoalList({ tasks: [], loaded: true, error: "", actionError: "", pending: "",
  onControl() {}, onRefresh() {}, ...extra });

test("initial goal loading and failure never claim there are no goals", () => {
  assert.match(text(view({ loaded: false })), /Loading goals/);
  const failed = text(view({ loaded: false, error: "Offline" }));
  assert.match(failed, /Couldn’t load goals/);
  assert.doesNotMatch(failed, /No saved goals/);
});

test("waiting for user input explains how to continue and preserves progress", () => {
  const result = text(view({ tasks: [task({ status: "waiting", wait_kind: "user", blocker: "Which budget?" })] }));
  assert.match(result, /Waiting for you/);
  assert.match(result, /Outline saved/);
  assert.match(result, /Which budget/);
  assert.match(result, /Reply in chat to continue/);
});

test("a failed refresh retains goals and disables edits to stale state", () => {
  const tree = view({ tasks: [task()], error: "Offline" });
  assert.match(text(tree), /Showing the last update/);
  assert.match(text(tree), /Prepare the proposal/);
  const controls = elements(tree).filter(node => node.type === "button" && node.props.children !== "Refresh");
  assert.equal(controls.length, 2);
  assert.ok(controls.every(node => node.props.disabled));
});

test("paused goals offer explicit Resume and pass the original saved goal", () => {
  const goal = task({ status: "paused", version: 5 });
  const calls = [];
  const tree = view({ tasks: [goal], onControl: (...args) => calls.push(args) });
  const button = elements(tree).find(node => node.type === "button" && node.props.children === "Resume");
  button.props.onClick();
  assert.equal(calls[0][0], goal);
  assert.equal(calls[0][1], "resume");
});

test("finished goals cannot restart while failed goals require explicit Resume", () => {
  const tree = view({ tasks: [task({ status: "completed" }), task({ id: "failed", status: "failed", blocker: "Worker unavailable" })] });
  assert.match(text(tree), /Finished.*Failed/s);
  assert.match(text(tree), /Worker unavailable/);
  const controls = elements(tree).filter(node => node.type === "button");
  assert.equal(controls.length, 1);
  assert.equal(controls[0].props.children, "Resume");
  assert.match(text(tree), /Blocker:/);
  assert.equal(elements(tree).find(node => node.type === "details").props.open, undefined);
});

test("a pending change locks every goal control", () => {
  const tree = view({ tasks: [task(), task({ id: "other" })], pending: "goal" });
  assert.ok(elements(tree).filter(node => node.type === "button").every(node => node.props.disabled));
  assert.match(text(tree), /Saving/);
  assert.match(text(tree), /Actions already started may continue/);
});

const settle = () => new Promise(resolve => setImmediate(resolve));
function mountedGoals(api) {
  const slots = [], effects = [], timers = [];
  let cursor = 0, tree;
  const output = {};
  const react = {
    useState(initial) {
      const index = cursor++;
      if (!(index in slots)) slots[index] = initial;
      return [slots[index], value => { slots[index] = typeof value === "function" ? value(slots[index]) : value; }];
    },
    useRef(initial) { return slots[cursor++] ??= { current: initial }; },
    useEffect(effect, dependencies) {
      const index = cursor++;
      const previous = slots[index];
      if (!previous || dependencies.some((value, i) => value !== previous.dependencies[i])) {
        const current = slots[index] = { dependencies };
        effects.push(() => { previous?.cleanup?.(); current.cleanup = effect(); });
      }
    },
  };
  runInNewContext(source, {
    exports: output,
    setTimeout: callback => { timers.push(callback); return callback; },
    clearTimeout: callback => { const index = timers.indexOf(callback); if (index >= 0) timers.splice(index, 1); },
    require: name => {
      if (name === "react") return react;
      if (name === "./api") return { api, agentPath: key => `/agents/${key}` };
      assert.equal(name, "react/jsx-runtime");
      return { jsx, jsxs: jsx };
    },
  });
  const render = () => {
    cursor = 0;
    tree = output.default({ agentKey: "alpha", conversationId: "chat" });
    while (effects.length) effects.shift()();
    return tree.props;
  };
  return { render, timers };
}
const savedTask = extra => task({ agent_key: "alpha", conversation_id: "chat", version: 3, ...extra });

test("controls send the displayed version once and reject a synchronous second click", async () => {
  const calls = [];
  let resolveChange;
  const mounted = mountedGoals(async (path, body) => {
    calls.push({ path, body });
    if (!body) return [savedTask()];
    return new Promise(resolve => { resolveChange = resolve; });
  });
  mounted.render(); await settle();
  const view = mounted.render();
  view.onControl(view.tasks[0], "pause");
  view.onControl(view.tasks[0], "cancel");
  assert.equal(calls.filter(call => call.body).length, 1);
  assert.equal(calls[1].path, "/agents/alpha/tasks/goal/pause");
  assert.equal(calls[1].body.expected_version, 3);
  resolveChange(savedTask({ status: "paused", version: 4 })); await settle();
  assert.equal(mounted.render().tasks[0].status, "paused");
});

test("a poll started before a control cannot replace its confirmed newer state", async () => {
  let reads = 0, resolveOld;
  const mounted = mountedGoals(async (_path, body) => {
    if (body) return savedTask({ status: "paused", version: 4 });
    reads += 1;
    if (reads === 1) return [savedTask()];
    return new Promise(resolve => { resolveOld = resolve; });
  });
  mounted.render(); await settle();
  const view = mounted.render();
  mounted.timers.shift()();
  view.onControl(view.tasks[0], "pause"); await settle();
  resolveOld([savedTask()]); await settle();
  assert.equal(mounted.render().tasks[0].version, 4);
});

test("an uncertain mutation requires a fresh read and is never repeated automatically", async () => {
  const calls = [];
  let resolveReload;
  const mounted = mountedGoals(async (_path, body) => {
    calls.push(body);
    if (body) throw new Error("Connection lost");
    if (calls.length === 1) return [savedTask()];
    return new Promise(resolve => { resolveReload = resolve; });
  });
  mounted.render(); await settle();
  const view = mounted.render();
  view.onControl(view.tasks[0], "cancel"); await settle();
  const failed = mounted.render();
  assert.match(failed.actionError, /Checking the saved state/);
  assert.ok(failed.error);
  failed.onControl(failed.tasks[0], "cancel"); await settle();
  assert.equal(calls.filter(Boolean).length, 1);
  resolveReload([savedTask({ status: "cancelled", version: 4 })]); await settle();
  const refreshed = mounted.render();
  assert.equal(refreshed.tasks[0].status, "cancelled");
  assert.equal(refreshed.error, "");
});
