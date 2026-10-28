import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { runInNewContext } from "node:vm";
import ts from "typescript";

const sources = Object.fromEntries(["App", "AgentForm"].map(name => [name,
  ts.transpileModule(readFileSync(new URL(`./${name}.tsx`, import.meta.url), "utf8"), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX },
  }).outputText,
]));
const a = { key: "a", name: "A" };
const b = { key: "b", name: "B" };
const c = { key: "c", name: "C" };

function elements(node) {
  if (!node || typeof node !== "object") return [];
  return [node, ...[node.props?.children].flat(Infinity).flatMap(elements)];
}
function find(tree, predicate) {
  const node = elements(tree).find(predicate);
  assert.ok(node, "Expected element in current component output");
  return node;
}

// Direct root/form callback units with hook, list-loading and browser-state fakes.
// NamespaceApp is an unrendered boundary; no DOM, service or editor is mounted.
function app({ agents = [a, b], storage = new Map([["tsuchi-namespace", "a"]]), url = new URL("https://tsuchi.test/") } = {}) {
  let scope, tree, form, listedAgents = agents, version;
  const requests = [], history = [], loads = [];
  const react = {
    useState(initial) {
      const owner = scope, index = owner.cursor++;
      if (!(index in owner.slots))
        owner.slots[index] = typeof initial === "function" ? initial() : initial;
      return [owner.slots[index], value => {
        owner.writes++;
        owner.slots[index] = typeof value === "function" ? value(owner.slots[index]) : value;
      }];
    },
    useRef(initial) {
      const index = scope.cursor++;
      return scope.slots[index] ??= { current: initial };
    },
    useEffect(effect, dependencies) {
      const owner = scope, index = owner.cursor++;
      const previous = owner.slots[index];
      if (!previous || dependencies.some((value, i) => value !== previous.dependencies[i])) {
        owner.slots[index] = { dependencies, cleanup: previous?.cleanup, effect };
        owner.effects.push(() => {
          owner.slots[index].cleanup?.();
          owner.slots[index].cleanup = effect();
        });
      }
    },
  };
  function hooks() {
    return {
      slots: [], effects: [], cursor: 0, writes: 0,
      run(component, props) {
        scope = this;
        this.cursor = 0;
        const output = component(props);
        this.effects.splice(0).forEach(effect => effect());
        return output;
      },
      cleanup() { this.slots.forEach(slot => slot?.cleanup?.()); },
      replayEffects() {
        this.cleanup();
        this.slots.forEach(slot => { if (slot?.effect) slot.cleanup = slot.effect(); });
      },
    };
  }
  const jsx = (type, props, key) => ({ type, props, key });
  const modules = {
    react,
    "react/jsx-runtime": { jsx, jsxs: jsx },
    "lucide-react": {},
    "./components": { Dialog: "Dialog", ErrorNotice: "ErrorNotice", Loading: "Loading" },
    "./AssetEditor": {}, "./ResourceEditor": {}, "./LibraryPages": {}, "./OverviewPages": {},
    "./api": {
      json: (method, body) => ({ method, body }),
      request: (path, options) => new Promise((resolve, reject) => requests.push({ path, options, resolve, reject })),
      useLoad: (_load, dependencies) => {
        if (version !== dependencies[0]) {
          if (version !== undefined) listedAgents = null;
          version = dependencies[0];
          loads.push(version);
        }
        return { data: listedAgents, loading: listedAgents === null, error: "" };
      },
    },
  };
  for (const name of ["AgentForm", "App"]) {
    const exports = {};
    runInNewContext(sources[name], {
      exports,
      require: name => { assert.ok(name in modules, name); return modules[name]; },
      Error, URLSearchParams,
      window: {
        location: { search: url.search },
        history: { replaceState: (...args) => { history.push(args); url.href = new URL(args[2], url).href; } },
      },
      localStorage: { getItem: key => storage.get(key) ?? null, setItem: (key, value) => storage.set(key, value) },
    });
    modules[`./${name}`] = exports;
  }
  const root = hooks();
  function render() {
    tree = root.run(modules["./App"].App);
    const node = elements(tree).find(node => node.type === modules["./AgentForm"].AgentForm);
    if (!node) {
      form?.owner.cleanup();
      form = null;
    } else {
      form ??= { owner: hooks() };
      form.tree = form.owner.run(node.type, node.props);
    }
  }
  function namespace() {
    return elements(tree).find(node => node.props?.agent);
  }
  function formView() {
    assert.ok(form, "Creation form must be mounted");
    const owner = form.owner;
    const field = index => elements(form.tree).filter(node => node.type === "input")[index];
    return {
      name: () => field(0).props.value,
      key: () => field(1).props.value,
      editName(value) { field(0).props.onChange({ target: { value } }); render(); },
      editKey(value) { field(1).props.onChange({ target: { value } }); render(); },
      error: () => find(form.tree, node => node.type === "ErrorNotice").props.message,
      busy: () => find(form.tree, node => node.type === "button").props.disabled,
      submit() {
        const pending = find(form.tree, node => node.type === "form").props.onSubmit({ preventDefault() {} });
        render();
        return pending;
      },
      writes: () => owner.writes,
      replayEffects: () => owner.replayEffects(),
    };
  }
  render();
  return {
    render, namespace, form: formView, requests, history, loads, storage, url,
    hasForm: () => form !== null,
    open() { namespace().props.onNewAgent(); render(); },
    close() { find(tree, node => node.type === "Dialog").props.onClose(); render(); },
    switch(key) { namespace().props.onSwitch(key); render(); },
    list(value) { listedAgents = value; render(); },
    reload() { return app({ agents: listedAgents, storage, url }); },
  };
}

test("dismissed pending creation cannot select C or refresh away namespace B", async () => {
  const view = app();
  view.open();
  const form = view.form();
  form.editName("C");
  const pending = form.submit();
  assert.equal(form.busy(), true);
  view.close();
  assert.equal(view.hasForm(), false, "Pending creation remains dismissible");
  view.switch("b");
  const namespace = view.namespace();
  const writes = form.writes();
  view.requests[0].resolve(c);
  await pending;
  view.render();
  assert.deepEqual(view.loads, [0], "Refreshing the root list would unmount the active namespace and its draft");
  assert.equal(view.namespace().key, namespace.key);
  assert.equal(view.namespace().type, namespace.type);
  assert.equal(view.namespace().props.agent, b);
  assert.equal(view.storage.get("tsuchi-namespace"), "b");
  assert.equal(view.history.length, 1);
  assert.equal(form.writes(), writes, "Unmounted form must receive no completion writes");
});

test("dismissed late failure cannot paint errors or affect a newly opened form", async () => {
  const view = app();
  view.open();
  const old = view.form();
  old.editName("C");
  const pending = old.submit();
  view.close();
  view.switch("b");
  view.open();
  const current = view.form();
  current.editName("D");
  const writes = old.writes();
  view.requests[0].reject(new Error("Late creation failure"));
  await pending;
  view.render();
  assert.equal(old.writes(), writes, "Unmounted form must receive no error or busy updates");
  assert.equal(current.name(), "D");
  assert.equal(current.error(), "");
  assert.equal(current.busy(), false);
  assert.equal(view.namespace().props.agent, b);
  assert.deepEqual(view.loads, [0]);
});

for (const search of ["", "?agent=c&resource=old-memory"]) {
  test(`current creation remembers B across reload and clears selection URL ${search || "/"}`, async () => {
    const view = app({ agents: [a, c], url: new URL(`https://tsuchi.test/${search}`) });
    assert.equal(view.namespace().props.agent, search ? c : a);
    view.open();
    view.form().editName("B");
    const pending = view.form().submit();
    view.requests[0].resolve(b);
    await pending;
    view.render();
    assert.equal(view.storage.get("tsuchi-namespace"), "b");
    assert.deepEqual(view.history, [[null, "", "/"]]);
    assert.equal(view.url.search, "");
    assert.deepEqual(view.loads, [0, 1]);
    assert.equal(view.namespace(), undefined, "Current success still reloads the root agent list");
    assert.equal(view.hasForm(), false);
    view.list([a, b, c]);
    assert.equal(view.namespace().props.agent, b);
    assert.equal(view.hasForm(), false);
    assert.equal(view.reload().namespace().props.agent, b);
  });
}

test("current duplicate error preserves selection, fields and deep link until a successful retry", async () => {
  const view = app({ agents: [a, c], url: new URL("https://tsuchi.test/?agent=c&resource=old-memory") });
  view.open();
  const form = view.form();
  form.editName("  New namespace  ");
  form.editKey("a");
  const failed = form.submit();
  assert.equal(view.requests[0].path, "/api/agents");
  assert.equal(view.requests[0].options.method, "POST");
  assert.equal(view.requests[0].options.body.name, "New namespace");
  assert.equal(view.requests[0].options.body.key, "a");
  view.requests[0].reject(new Error("Namespace key already exists."));
  await failed;
  view.render();
  assert.equal(form.error(), "Namespace key already exists.");
  assert.equal(form.busy(), false);
  assert.equal(form.name(), "  New namespace  ");
  assert.equal(form.key(), "a");
  assert.equal(view.namespace().props.agent, c);
  assert.equal(view.storage.get("tsuchi-namespace"), "a");
  assert.equal(view.url.search, "?agent=c&resource=old-memory");
  assert.deepEqual(view.history, []);
  assert.deepEqual(view.loads, [0]);
  form.editKey("b");
  const retry = form.submit();
  assert.equal(form.error(), "");
  assert.equal(form.busy(), true);
  assert.equal(view.requests[1].options.body.key, "b");
  view.requests[1].resolve(b);
  await retry;
  view.render();
  view.list([a, b, c]);
  assert.equal(view.namespace().props.agent, b);
  assert.equal(view.storage.get("tsuchi-namespace"), "b");
});

test("non-Error creation failure retains the existing fallback and permits retry", async () => {
  const view = app();
  view.open();
  const form = view.form();
  form.editName("B");
  const pending = form.submit();
  view.requests[0].reject(null);
  await pending;
  view.render();
  assert.equal(form.error(), "Unable to create namespace.");
  assert.equal(form.busy(), false);
  assert.equal(view.namespace().props.agent, a);
  assert.deepEqual(view.loads, [0]);
});

test("onboarding creation uses the same remembered selection semantics", async () => {
  const view = app({ agents: [] });
  const form = view.form();
  form.editName("B");
  const pending = form.submit();
  view.requests[0].resolve(b);
  await pending;
  view.render();
  assert.equal(view.storage.get("tsuchi-namespace"), "b");
  assert.deepEqual(view.history, [[null, "", "/"]]);
  view.list([b]);
  assert.equal(view.namespace().props.agent, b);
});

test("namespace selector still persists explicit selection and clears deep links", () => {
  const view = app({ url: new URL("https://tsuchi.test/?agent=a&resource=memory") });
  view.switch("b");
  assert.equal(view.namespace().props.agent, b);
  assert.equal(view.storage.get("tsuchi-namespace"), "b");
  assert.deepEqual(view.history, [[null, "", "/"]]);
  assert.equal(view.reload().namespace().props.agent, b);
  assert.deepEqual(view.loads, [0]);
});

test("closing an idle creation form leaves remembered and deep-link selection alone", () => {
  const view = app({ url: new URL("https://tsuchi.test/?agent=b&resource=memory") });
  view.open();
  view.form().editName("C");
  view.close();
  assert.equal(view.namespace().props.agent, b);
  assert.equal(view.storage.get("tsuchi-namespace"), "a");
  assert.equal(view.url.search, "?agent=b&resource=memory");
  assert.deepEqual(view.requests, []);
  assert.deepEqual(view.history, []);
  assert.deepEqual(view.loads, [0]);
});

test("effect setup after cleanup still allows a current creation to complete", async () => {
  const view = app();
  view.open();
  const form = view.form();
  form.replayEffects();
  form.editName("C");
  const pending = form.submit();
  view.requests[0].resolve(c);
  await pending;
  view.render();
  assert.deepEqual(view.loads, [0, 1]);
  view.list([a, b, c]);
  assert.equal(view.namespace().props.agent, c);
});
