import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { runInNewContext } from "node:vm";
import ts from "typescript";

const source = ts.transpileModule(
  readFileSync(new URL("./Personas.tsx", import.meta.url), "utf8"),
  { compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX } },
).outputText;
const draftKey = "kadan.persona-draft.agent";
const original = { id: "persona", title: "Original", markdown: "Original instructions", version: 1 };
const saved = { ...original, title: "Guide", markdown: "Saved instructions", version: 2 };

// Exercise the component's handlers and effects with in-memory hook/API boundaries.
function editor(selected = original) {
  const storage = new Map([[draftKey, JSON.stringify({
    selected, title: "  Guide  ", markdown: "Draft instructions",
  })]]);
  const states = [], effects = [], requests = [], changes = [], busy = [];
  let cursor, tree;
  const react = {
    useState(initial) {
      const index = cursor++;
      if (!(index in states)) states[index] = typeof initial === "function" ? initial() : initial;
      return [states[index], value => {
        states[index] = typeof value === "function" ? value(states[index]) : value;
      }];
    },
    useEffect(effect) { effects.push(effect); },
  };
  const jsx = (type, props) => ({ type, props });
  const modules = {
    react,
    "react/jsx-runtime": { jsx, jsxs: jsx },
    "lucide-react": {},
    "./Markdown": { default: "Markdown" },
    "./api": {
      agentPath: key => `/api/agents/${key}`,
      api: (path, body, method) => new Promise((resolve, reject) => {
        requests.push({ path, body, method, resolve, reject });
      }),
    },
  };
  const exports = {};
  runInNewContext(source, {
    exports,
    require: name => { assert.ok(name in modules, name); return modules[name]; },
    sessionStorage: {
      getItem: key => storage.get(key),
      setItem: (key, value) => storage.set(key, value),
      removeItem: key => storage.delete(key),
    },
  });
  function render() {
    cursor = 0;
    tree = exports.default({
      agentKey: "agent", personas: selected ? [selected] : [], blocked: false,
      onChange: rows => changes.push(rows), onBusy: value => busy.push(value),
    });
    effects.splice(0).forEach(effect => effect());
  }
  function find(predicate, node = tree) {
    if (!node || typeof node !== "object") return;
    if (predicate(node)) return node;
    for (const child of [node.props?.children].flat(Infinity)) {
      const found = find(predicate, child ?? null);
      if (found) return found;
    }
  }
  const element = type => find(node => node.type === type);
  render();
  return {
    render, requests, changes, busy, storage,
    draft: () => JSON.parse(storage.get(draftKey)),
    title: () => element("input").props.value,
    markdown: () => element("textarea").props.value,
    edit(type, value) { element(type).props.onChange({ target: { value } }); render(); },
    save: () => element("form").props.onSubmit({ preventDefault() {} }),
    compare: () => find(node => node.props?.children === "Load latest for comparison").props.onClick(),
    done: () => Boolean(find(node => node.props?.children === "Done")),
    error: () => find(node => node.props?.role === "alert")?.props.children[0],
    comparison: () => element("details"),
  };
}

for (const selected of [null, original]) {
  test(`successful ${selected ? "update" : "creation"} adopts saved fields and removes the browser draft`, async () => {
    const view = editor(selected);
    const saving = view.save();
    const request = view.requests[0];
    assert.equal(request.path, `/api/agents/agent/resources${selected ? "/persona" : ""}`);
    assert.equal(request.body.title, "Guide");
    assert.equal(request.body.markdown, "Draft instructions");
    assert.equal(request.body.expected_version, selected?.version);
    assert.equal(request.method, selected ? "PUT" : undefined);
    request.resolve(saved);
    await saving;
    view.render();
    assert.equal(view.title(), saved.title);
    assert.equal(view.markdown(), saved.markdown);
    assert.equal(view.done(), true);
    assert.equal(view.storage.has(draftKey), false);
    assert.equal(view.changes[0][0], saved);
    assert.deepEqual(view.busy, [true, false]);
  });
}

for (const message of ["Network unavailable", "Resource version conflict"]) {
  test(`${message} retains the draft; comparison updates only the saved baseline`, async () => {
    const view = editor();
    const draft = view.storage.get(draftKey);
    const saving = view.save();
    view.requests[0].reject(new Error(message));
    await saving;
    view.render();
    assert.equal(view.error(), message);
    assert.equal(view.storage.get(draftKey), draft);
    assert.equal(view.changes.length, 0);
    view.compare();
    view.requests[1].resolve(saved);
    await new Promise(resolve => setImmediate(resolve));
    view.render();
    assert.equal(view.title(), "  Guide  ");
    assert.equal(view.markdown(), "Draft instructions");
    assert.equal(view.draft().selected.version, saved.version);
    assert.ok(view.comparison());
    const retry = view.save();
    assert.equal(view.requests[2].body.expected_version, saved.version);
    view.requests[2].resolve(saved);
    await retry;
    view.render();
    assert.equal(view.storage.has(draftKey), false);
    assert.equal(view.comparison(), undefined);
  });
}

for (const fields of [["input"], ["textarea"], ["input", "textarea"]]) {
  test(`save preserves newer ${fields.join(" and ")} edits while synchronizing unchanged fields`, async () => {
    const view = editor();
    const saving = view.save();
    for (const field of fields) view.edit(field, `New ${field}`);
    view.requests[0].resolve(saved);
    await saving;
    view.render();
    assert.equal(view.title(), fields.includes("input") ? "New input" : saved.title);
    assert.equal(view.markdown(), fields.includes("textarea") ? "New textarea" : saved.markdown);
    assert.equal(view.done(), false);
    assert.equal(view.draft().title, view.title());
    assert.equal(view.draft().markdown, view.markdown());
    assert.equal(view.draft().selected.version, saved.version);
  });
}
