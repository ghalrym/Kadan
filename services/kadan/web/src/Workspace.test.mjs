import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { runInNewContext } from "node:vm";
import ts from "typescript";
import * as documentNavigation from "./workspaceDocuments.ts";

const source = ts.transpileModule(
  readFileSync(new URL("./Workspace.tsx", import.meta.url), "utf8"),
  { compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX } },
).outputText;
const record = (id, kind = "memory", parent_id = null) => ({
  id, title: id, kind, parent_id, markdown: `${id} text`, tags: [],
  updated_at: "2026-10-22T00:00:00Z",
});

// Exercise the component's effects and handlers with local hooks, timers and API
// promises. No mounted UI, browser, storage service or inference is involved.
function workspace() {
  const hooks = [], pendingEffects = [], timers = new Map(), requests = [];
  let cursor, dirty, tree, timerId = 0, writes = 0, agentKey = "agent";
  const request = (args) => new Promise((resolve, reject) => {
    requests.push({ args, resolve, reject });
  });
  const jsx = (type, props) => ({ type, props });
  const modules = {
    react: {
      useState(initial) {
        const index = cursor++;
        hooks[index] ??= { value: initial };
        return [hooks[index].value, next => {
          writes++;
          const value = typeof next === "function" ? next(hooks[index].value) : next;
          if (!Object.is(value, hooks[index].value)) {
            hooks[index].value = value;
            dirty = true;
          }
        }];
      },
      useEffect(effect, deps) {
        const index = cursor++;
        if (hooks[index]?.deps.every((dep, i) => Object.is(dep, deps[i]))) return;
        pendingEffects.push(() => {
          hooks[index]?.cleanup?.();
          hooks[index] = { deps, cleanup: effect() };
        });
      },
    },
    "react/jsx-runtime": { jsx, jsxs: jsx },
    "lucide-react": {},
    "./Markdown": { default: "Markdown" },
    "./workspaceDocuments": documentNavigation,
    "./api": {
      agentPath: key => `/agents/${key}`,
      api: path => request([path]),
      allResources: (...args) => request(args),
    },
  };
  const exports = {};
  runInNewContext(source, {
    exports,
    require: name => { assert.ok(name in modules, name); return modules[name]; },
    setTimeout: callback => { timers.set(++timerId, callback); return timerId; },
    clearTimeout: id => timers.delete(id),
  });
  function render() {
    do {
      dirty = false;
      cursor = 0;
      tree = exports.default({ agentKey, browserUrl: null });
      pendingEffects.splice(0).forEach(effect => effect());
    } while (dirty);
  }
  function find(predicate, node) {
    if (!node || typeof node !== "object") return;
    if (predicate(node)) return node;
    for (const child of [node.props?.children].flat(Infinity)) {
      const found = find(predicate, child);
      if (found) return found;
    }
  }
  function text(node) {
    if (node == null || typeof node === "boolean") return "";
    if (typeof node === "string" || typeof node === "number") return String(node);
    return [node?.props?.children].flat(Infinity).map(child => text(child)).join("");
  }
  const element = predicate => find(predicate, tree);
  const button = label => element(node => node.type === "button" &&
    (text(node) === label || node.props["aria-label"] === label ||
      (node.props.className?.includes("resource-row") &&
        find(child => child.type === "strong" && text(child) === label, node))));
  const click = label => { assert.ok(button(label), label); button(label).props.onClick(); render(); };
  render();
  click("Memories");
  return {
    requests, render, text: () => text(tree), find: element, button, click,
    writes: () => writes,
    search() { return element(node => node.type === "input").props.value; },
    query(value) {
      element(node => node.type === "input").props.onChange({ target: { value } });
      render();
    },
    agent(value) { agentKey = value; render(); },
    start() {
      const callbacks = [...timers.values()];
      timers.clear();
      callbacks.forEach(callback => callback());
    },
    async settle(index, value, failure = false) {
      requests[index][failure ? "reject" : "resolve"](failure ? new Error(value) : value);
      await new Promise(resolve => setImmediate(resolve));
      render();
    },
    unmount() { hooks.forEach(hook => hook.cleanup?.()); },
  };
}

for (const [tab, kind] of [["Memories", "memory"], ["Documents", "space"], ["Files", "files"]]) {
  test(`${tab} failure retries the same search without claiming an empty collection`, async () => {
    const view = workspace();
    view.click(tab);
    view.query("my notes & plans");
    view.start();
    assert.equal(view.requests.length, 1);
    const args = view.requests[0].args;
    if (kind === "files") assert.deepEqual(args, ["/agents/agent/assets?q=my%20notes%20%26%20plans"]);
    else assert.deepEqual(args, ["agent", kind === "space" ? undefined : kind, "my notes & plans"]);
    await view.settle(0, "Storage unavailable", true);
    assert.match(view.text(), /Storage unavailable/);
    assert.doesNotMatch(view.text(), /No matching records|No .* yet|0 (records|documents)/);
    view.click("Retry");
    assert.equal(view.search(), "my notes & plans");
    assert.match(view.text(), /Searching workspace/);
    assert.doesNotMatch(view.text(), /Storage unavailable|No matching records/);
    view.start();
    assert.deepEqual(view.requests[1].args, args);
    await view.settle(1, []);
    assert.match(view.text(), /No matching records/);
    assert.match(view.text(), /0 (records|documents)/);
    assert.equal(view.button("Retry"), undefined);
  });
}

test("Document contents retry keeps the Space, search and collection without reloading the list", async () => {
  const view = workspace();
  view.click("Documents");
  view.query("project");
  view.start();
  await view.settle(0, [record("Project Space", "space")]);
  view.click("Project Space");
  await view.settle(1, "Contents unavailable", true);
  assert.match(view.text(), /Project Space text/);
  assert.match(view.text(), /Contents unavailable/);
  assert.doesNotMatch(view.text(), /No subdocuments/);
  assert.equal(view.find(node => node.props?.className === "resource-list")?.props.children
    .some(node => node?.props?.role === "alert"), false);
  view.click("Retry");
  assert.equal(view.search(), "project");
  assert.match(view.text(), /Project Space text/);
  assert.match(view.text(), /Loading documents/);
  assert.doesNotMatch(view.text(), /Contents unavailable|No subdocuments/);
  assert.equal(view.requests.length, 3);
  assert.deepEqual(view.requests[2].args, ["agent"]);
  await view.settle(2, [record("Child", "space", "Project Space"), record("Other")]);
  assert.match(view.text(), /Child/);
  assert.doesNotMatch(view.text(), /Other/);
  view.click("Child");
  assert.match(view.text(), /Child text/);
});

test("a successful empty Space read shows the empty state", async () => {
  const view = workspace();
  view.click("Documents");
  view.start();
  await view.settle(0, [record("Space", "space")]);
  view.click("Space");
  await view.settle(1, []);
  assert.match(view.text(), /No subdocuments/);
  assert.equal(view.button("Retry"), undefined);
});

for (const failure of [false, true]) {
  test(`an older search ${failure ? "failure" : "success"} cannot replace a newer search`, async () => {
    const view = workspace();
    view.start();
    view.query("latest");
    view.start();
    if (failure) {
      await view.settle(0, "Old error", true);
      assert.match(view.text(), /Searching workspace/);
      assert.doesNotMatch(view.text(), /Old error|No matching records/);
    }
    await view.settle(1, [record("Latest")]);
    if (!failure) await view.settle(0, [record("Old")]);
    assert.match(view.text(), /Latest/);
    assert.doesNotMatch(view.text(), /Old|Searching workspace/);
  });
}

test("switching collections and agents ignores earlier collection responses", async () => {
  const view = workspace();
  view.start();
  view.click("Files");
  view.start();
  view.agent("other");
  view.start();
  assert.deepEqual(view.requests[2].args, ["/agents/other/assets?q="]);
  await view.settle(2, [{ ...record("Current file"), filename: "current.txt" }]);
  await view.settle(0, [record("Old memory")]);
  await view.settle(1, "Old agent error", true);
  assert.match(view.text(), /Current file/);
  assert.doesNotMatch(view.text(), /Old|Searching workspace/);
});

for (const failure of [false, true]) {
  test(`an older Space ${failure ? "failure" : "success"} cannot replace the selected Space`, async () => {
    const view = workspace();
    view.click("Documents");
    view.start();
    await view.settle(0, [record("First", "space"), record("Second", "space")]);
    view.click("First");
    view.click("Second");
    await view.settle(2, [record("Current child", "space", "Second")]);
    await view.settle(1, failure ? "Old contents error" : [record("Old child", "space", "First")], failure);
    assert.match(view.text(), /Second text/);
    assert.match(view.text(), /Current child/);
    assert.doesNotMatch(view.text(), /Old|Loading documents/);
    view.click("Close record");
    assert.match(view.text(), /Select a (record|document)/);
    assert.equal(view.button("Retry"), undefined);
  });
}

for (const [navigation, navigate] of [
  ["changing collection", view => view.click("Files")],
  ["changing agent", view => view.agent("other")],
  ["closing the record", view => view.click("Close record")],
]) test(`${navigation} dismisses a Space and ignores its pending contents failure`, async () => {
  const view = workspace();
  view.click("Documents");
  view.start();
  await view.settle(0, [record("Space", "space")]);
  view.click("Space");
  navigate(view);
  await view.settle(1, "Old contents error", true);
  assert.match(view.text(), /Select a (record|document)/);
  assert.doesNotMatch(view.text(), /Space text|Old contents error/);
});

test("overlapping Space retries accept only the latest response", async () => {
  const view = workspace();
  view.click("Documents");
  view.start();
  await view.settle(0, [record("Space", "space")]);
  view.click("Space");
  await view.settle(1, "Unavailable", true);
  const retry = view.button("Retry").props.onClick;
  view.click("Retry");
  retry();
  // Commit a second activation of the same control while the first read waits.
  view.render();
  await view.settle(3, [record("Latest child", "space", "Space")]);
  await view.settle(2, "Older retry failed", true);
  assert.match(view.text(), /Space text/);
  assert.match(view.text(), /Latest child/);
  assert.doesNotMatch(view.text(), /Older retry failed|Loading documents/);
});

test("unmount cancels debounced reads and prevents pending reads from writing state", async () => {
  const idle = workspace();
  idle.unmount();
  idle.start();
  assert.equal(idle.requests.length, 0);
  const pending = workspace();
  pending.click("Documents");
  pending.start();
  await pending.settle(0, [record("Space", "space")]);
  pending.click("Space");
  pending.unmount();
  const writes = pending.writes();
  await pending.settle(1, "Late failure", true);
  assert.equal(pending.writes(), writes);
});


test("document search keeps the open document and its parent navigation", async () => {
  const view = workspace();
  view.click("Documents");
  view.start();
  const documents = [record("Project", "space"), record("Plan", "space", "Project")];
  await view.settle(0, documents);
  view.click("Plan");
  await view.settle(1, documents);
  view.query("unmatched");
  view.start();
  await view.settle(2, []);
  assert.match(view.text(), /No matching records/);
  assert.match(view.text(), /Plan text/);
  view.click("Back to Project");
  assert.match(view.text(), /Project text/);
  assert.equal(view.search(), "unmatched");
});

test("documents exclude memories, personas and conversations at every level", async () => {
  const view = workspace();
  view.click("Documents");
  view.start();
  const documents = [record("Project", "space"), record("Slides", "deck", "Project")];
  const records = [...documents, record("Private memory", "memory", "Project"),
    record("Persona", "persona"), record("Chat", "conversation", "Project")];
  await view.settle(0, records);
  assert.match(view.text(), /Slides/);
  assert.doesNotMatch(view.text(), /Private memory|Persona|Chat/);
  view.click("Project");
  await view.settle(1, records);
  assert.match(view.text(), /Slides/);
  assert.doesNotMatch(view.text(), /Private memory|Persona|Chat/);
});

test("a document opened from search resolves its full breadcrumb and returns to an ancestor", async () => {
  const view = workspace();
  view.click("Documents");
  view.query("Plan");
  view.start();
  const plan = record("Plan", "space", "Project");
  await view.settle(0, [plan]);
  view.click("Plan");
  await view.settle(1, [record("Project", "space"), plan]);
  const path = view.find(node => node.props?.["aria-label"] === "Document path");
  assert.equal(path.props.children[1][0].props.children[1].props.children, "Project");
  view.click("Project");
  assert.match(view.text(), /Project text/);
  assert.equal(view.search(), "Plan");
});
