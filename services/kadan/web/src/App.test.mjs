import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { runInNewContext } from "node:vm";
import ts from "typescript";

const source = ts.transpileModule(
  readFileSync(new URL("./App.tsx", import.meta.url), "utf8"),
  { compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX } },
).outputText;
const agent = { key: "guide", name: "Guide" };
const persona = { id: "persona", agent_key: agent.key, kind: "persona", markdown: "Be precise." };
const chain = { id: "chain", agent_key: agent.key, kind: "conversation" };

// Call App with in-memory hooks and service boundaries; no browser or mounted UI.
async function app(overrides = {}) {
  const slots = [], effects = [], requests = [];
  let cursor = 0, tree;
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
    useCallback(callback, dependencies) {
      const index = cursor++;
      if (!slots[index] || dependencies.some((value, i) => value !== slots[index].dependencies[i])) {
        slots[index] = { callback, dependencies };
      }
      return slots[index].callback;
    },
    useEffect(effect, dependencies) {
      const index = cursor++;
      if (!slots[index] || dependencies.some((value, i) => value !== slots[index].dependencies[i])) {
        const previous = slots[index];
        slots[index] = { dependencies };
        effects.push(() => {
          previous?.cleanup?.();
          slots[index].cleanup = effect();
        });
      }
    },
  };
  const jsx = (type, props) => ({ type, props });
  const modules = {
    react,
    "react/jsx-runtime": { jsx, jsxs: jsx },
    "react-dom": { flushSync: callback => callback() },
    "lucide-react": {},
    "./Chat": { default: "Chat" },
    "./Workspace": { default: "Workspace" },
    "./Personas": { default: "Personas" },
    "./RenameAgent": { default: "RenameAgent" },
    "./RunnerAssignment": { default: "RunnerAssignment" },
    "./agentRuns": { latestConversation: (current, incoming) => current.id === incoming.id && incoming.version > current.version ? incoming : current },
    "./Hatch": { default: "Hatch" },
    "./Pet": {},
    "./pet": { agentHue: () => 252 },
    "./PetEditor": { default: "PetEditor" },
    "./PetContext": { PetContext: { Provider: "PetProvider" } },
    "./api": {
      agentPath: key => `/agents/${key}`,
      async allResources(key, kind) {
        requests.push({ key, kind });
        return overrides.allResources ? overrides.allResources(key, kind) : [persona];
      },
      async api(path, ...args) {
        if (overrides.api) return overrides.api(path, ...args);
        if (path === "/agents") return [agent];
        if (path === `/agents/${agent.key}/chain`) return chain;
        if (path.endsWith("/pet")) return null;
        if (path === "/status") return null;
        assert.fail(`Unexpected API call: ${path}`);
      },
    },
  };
  const exports = {};
  runInNewContext(source, {
    exports,
    require: name => { assert.ok(name in modules, name); return modules[name]; },
    document: {},
    matchMedia: () => ({ matches: false, addEventListener() {}, removeEventListener() {} }),
    setInterval: () => 0,
    clearInterval() {},
  });
  function render() {
    cursor = 0;
    tree = exports.default();
    effects.splice(0).forEach(effect => effect());
  }
  function findAll(predicate, node = tree) {
    if (!node || typeof node !== "object") return [];
    return [
      ...(predicate(node) ? [node] : []),
      ...[node.props?.children].flat(Infinity).flatMap(child => findAll(predicate, child ?? null)),
    ];
  }
  for (let i = 0; i < 3; i++) {
    render();
    await new Promise(resolve => setImmediate(resolve));
  }
  return { render, findAll, requests };
}

test("navigation offers Messages and Workspace without a standalone Persona view", async () => {
  const view = await app();
  const navigation = view.findAll(node => node.props?.["aria-label"] === "Views")[0];
  const buttons = view.findAll(node => node.type === "button", navigation);
  assert.deepEqual(buttons.map(node => node.props.children.at(-1)), ["Messages", "Workspace"]);
  const settings = view.findAll(node => node.props?.["aria-label"] === "Agent settings")[0];
  assert.deepEqual(
    view.findAll(node => node.type === "button", settings).map(node => node.props.children.at(-1)),
    ["Edit pet", "Rename agent", "Runner assignment"],
  );
  assert.equal(view.findAll(node => node.type === "Personas").length, 0);
  buttons.at(-1).props.onClick();
  view.render();
  assert.equal(view.findAll(node => node.type === "Workspace").length, 1);
  assert.equal(view.findAll(node => node.props?.className === "workspace-pane")[0].props.hidden, false);
  assert.equal(view.findAll(node => node.type === "Chat").length, 0);
  buttons.at(-2).props.onClick();
  view.render();
  assert.equal(view.findAll(node => node.type === "Chat").length, 1);
  assert.equal(view.findAll(node => node.type === "Workspace").length, 1);
  assert.equal(view.findAll(node => node.props?.className === "workspace-pane")[0].props.hidden, true);
});

test("hiding the editor preserves agent-scoped persona loading for chat", async () => {
  const view = await app();
  assert.ok(view.requests.length > 0);
  for (const request of view.requests) {
    assert.deepEqual(request, { key: agent.key, kind: "persona" });
  }
  const chat = view.findAll(node => node.type === "Chat")[0];
  assert.equal(chat.props.agentKey, agent.key);
  assert.equal(chat.props.chain, chain);
  assert.equal(chat.props.personas[0], persona);
});

function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}
const tick = () => new Promise(resolve => setImmediate(resolve));

test("a new agent never receives the old agent's resources when its load fails", async () => {
  const other = { key: "other", name: "Other" };
  let selected = agent;
  const view = await app({
    api: async path => {
      if (path === "/agents") return [selected];
      if (path === "/status" || path.endsWith("/pet")) return null;
      if (path === "/agents/guide/chain") return chain;
      if (path === "/agents/other/chain") throw Error("conversation unavailable");
      assert.fail(path);
    },
    allResources: async key => key === "guide" ? [persona] : [],
  });
  assert.equal(view.findAll(node => node.type === "Chat")[0].props.chain, chain);
  selected = other;
  view.findAll(node => node.props?.["aria-label"] === "Refresh services and agent")[0].props.onClick();
  await tick();
  view.render();
  assert.equal(view.findAll(node => node.type === "Chat").length, 0);
  await tick();
  view.render();
  assert.equal(view.findAll(node => node.type === "Chat").length, 0);
  assert.match(view.findAll(node => node.props?.role === "alert")[0].props.children[0], /conversation unavailable/);
});

test("late resource responses from a previous refresh cannot replace the latest resources", async () => {
  const delayed = deferred();
  let calls = 0;
  const newest = { ...chain, id: "newest" };
  const view = await app({
    api: async path => {
      if (path === "/agents") return [agent];
      if (path === "/status" || path.endsWith("/pet")) return null;
      if (path.endsWith("/chain")) return ++calls === 1 ? chain : calls === 2 ? delayed.promise : newest;
      assert.fail(path);
    },
  });
  const refresh = () => view.findAll(node => node.props?.["aria-label"] === "Refresh services and agent")[0].props.onClick();
  refresh(); await tick(); view.render();
  refresh(); await tick(); view.render(); await tick(); view.render();
  assert.equal(view.findAll(node => node.type === "Chat")[0].props.chain, newest);
  delayed.resolve({ ...chain, id: "stale" });
  await tick(); view.render();
  assert.equal(view.findAll(node => node.type === "Chat")[0].props.chain, newest);
});

test("a late pet selection from another agent cannot replace the current pet", async () => {
  const delayed = deferred();
  const other = { key: "other", name: "Other" };
  const otherPet = { id: "other-pet", version: 1 };
  let selected = agent;
  const view = await app({
    api: async path => {
      if (path === "/agents") return [selected];
      if (path === "/status") return null;
      if (path === "/agents/guide/pet") return delayed.promise;
      if (path === "/agents/other/pet") return otherPet;
      if (path.endsWith("/chain")) return { ...chain, agent_key: selected.key };
      assert.fail(path);
    },
  });
  selected = other;
  view.findAll(node => node.props?.["aria-label"] === "Refresh services and agent")[0].props.onClick();
  await tick(); view.render(); await tick(); view.render();
  delayed.resolve({ id: "old-pet", version: 1 });
  await tick(); view.render();
  const pets = view.findAll(node => node.type === "PetProvider")[0].props.value;
  assert.equal(pets.other, otherPet);
  assert.equal(pets.guide, undefined);
});

test("the newest agent refresh wins when an earlier agent request finishes late", async () => {
  const delayed = deferred();
  const other = { key: "other", name: "Other" };
  let calls = 0;
  const view = await app({
    api: async path => {
      if (path === "/agents") return ++calls === 1 ? [agent] : calls === 2 ? delayed.promise : [other];
      if (path === "/status" || path.endsWith("/pet")) return null;
      if (path.endsWith("/chain")) return { ...chain, agent_key: path.split("/")[2] };
      assert.fail(path);
    },
    allResources: async key => [{ ...persona, agent_key: key }],
  });
  const refresh = () => view.findAll(node => node.props?.["aria-label"] === "Refresh services and agent")[0].props.onClick();
  refresh(); refresh(); await tick(); view.render(); await tick(); view.render();
  assert.equal(view.findAll(node => node.type === "Chat")[0].props.agentKey, "other");
  delayed.resolve([agent]); await tick(); view.render();
  assert.equal(view.findAll(node => node.type === "Chat")[0].props.agentKey, "other");
});

test("a failed refresh for the same agent retains its last loaded conversation", async () => {
  let fail = false;
  const view = await app({
    api: async path => {
      if (path === "/agents") return [agent];
      if (path === "/status" || path.endsWith("/pet")) return null;
      if (path.endsWith("/chain")) {
        if (fail) throw Error("unavailable");
        return chain;
      }
      assert.fail(path);
    },
  });
  fail = true;
  view.findAll(node => node.props?.["aria-label"] === "Refresh services and agent")[0].props.onClick();
  await tick(); view.render(); await tick(); view.render();
  assert.equal(view.findAll(node => node.type === "Chat")[0].props.chain, chain);
  assert.match(view.findAll(node => node.props?.role === "alert")[0].props.children[0], /unavailable/);
});

test("a delayed refresh cannot replace newer messages received during that refresh", async () => {
  const delayed = deferred();
  const initial = { ...chain, version: 1 };
  const newer = { ...chain, version: 3 };
  let calls = 0;
  const view = await app({
    api: async path => {
      if (path === "/agents") return [agent];
      if (path === "/status" || path.endsWith("/pet")) return null;
      if (path.endsWith("/chain")) return ++calls === 1 ? initial : delayed.promise;
      assert.fail(path);
    },
  });
  view.findAll(node => node.props?.["aria-label"] === "Refresh services and agent")[0].props.onClick();
  await tick(); view.render();
  view.findAll(node => node.type === "Chat")[0].props.onChange(newer);
  delayed.resolve({ ...chain, version: 2 });
  await tick(); view.render();
  assert.equal(view.findAll(node => node.type === "Chat")[0].props.chain, newer);
});

test("a previous agent's chat callback cannot update the newly loaded agent", async () => {
  const other = { key: "other", name: "Other" };
  const otherChain = { ...chain, agent_key: "other", version: 1 };
  const otherPersona = { ...persona, agent_key: "other" };
  let selected = agent;
  const view = await app({
    api: async path => {
      if (path === "/agents") return [selected];
      if (path === "/status" || path.endsWith("/pet")) return null;
      if (path.endsWith("/chain")) return selected.key === "other" ? otherChain : { ...chain, version: 1 };
      assert.fail(path);
    },
    allResources: async key => key === "other" ? [otherPersona] : [persona],
  });
  const previous = view.findAll(node => node.type === "Chat")[0].props.onChange;
  selected = other;
  view.findAll(node => node.props?.["aria-label"] === "Refresh services and agent")[0].props.onClick();
  await tick(); view.render(); await tick(); view.render();
  previous({ ...chain, version: 10 });
  view.render();
  const chat = view.findAll(node => node.type === "Chat")[0];
  assert.equal(chat.props.agentKey, "other");
  assert.equal(chat.props.chain, otherChain);
  assert.equal(chat.props.personas[0], otherPersona);
});
