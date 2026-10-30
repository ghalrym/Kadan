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
const settle = () => new Promise(resolve => setImmediate(resolve));
function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}

// Call App with in-memory hooks and service boundaries; no browser or mounted UI.
async function app(petResponses = [], agentResponses = [[agent]]) {
  const overrides = Array.isArray(petResponses) ? {} : petResponses;
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
      const previous = slots[index];
      if (!previous || dependencies.some((value, i) => value !== previous.dependencies[i])) {
        slots[index] = { dependencies, callback };
      }
      return slots[index].callback;
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
    "react-dom": { flushSync: callback => callback() },
    "lucide-react": {},
    "./Chat": { default: "Chat" },
    "./Workspace": { default: "Workspace" },
    "./Personas": { default: "Personas" },
    "./RenameAgent": { default: "RenameAgent" },
    "./RunnerAssignment": { default: "RunnerAssignment" },
    "./agentRuns": { latestConversation: (current, incoming) => current.id === incoming.id && incoming.version > current.version ? incoming : current },
    "./Hatch": { default: "Hatch" },
    "./Pet": { agentPet: () => ({ url: "hatchling" }) },
    "./pet": { agentHue: () => 252 },
    "./PetEditor": { default: "PetEditor" },
    "./PetContext": { PetContext: { Provider: "PetProvider" } },
    "./api": {
      agentPath: key => `/agents/${key}`,
      async allResources(key, kind) {
        requests.push({ key, kind });
        return overrides.allResources ? overrides.allResources(key, kind) : [persona];
      },
      async api(path, body, method) {
        if (overrides.api) return overrides.api(path, body, method);
        if (path === "/agents") return agentResponses.shift() ?? [agent];
        if (path.endsWith("/chain")) return chain;
        if (path.endsWith("/pet")) {
          requests.push({ path, body, method });
          const response = petResponses.shift();
          if (response instanceof Error) throw response;
          return response ?? null;
        }
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
  for (const request of view.requests.filter(request => request.kind)) {
    assert.deepEqual(request, { key: agent.key, kind: "persona" });
  }
  const chat = view.findAll(node => node.type === "Chat")[0];
  assert.equal(chat.props.agentKey, agent.key);
  assert.equal(chat.props.chain, chain);
  assert.equal(chat.props.personas[0], persona);
});

test("retry reloads the selected pet after a failed effect and clears its error on success", async () => {
  const selected = { id: "fox", version: 1 };
  const view = await app([new Error("Storage unavailable"), selected]);
  const banner = () => view.findAll(node => node.props?.role === "alert")[0];
  assert.match(banner().props.children[0], /Storage unavailable/);
  assert.equal(view.requests.filter(request => request.path?.endsWith("/pet")).length, 1);
  view.findAll(node => node.type === "button", banner())[0].props.onClick();
  view.render();
  await new Promise(resolve => setImmediate(resolve));
  view.render();
  assert.equal(view.requests.filter(request => request.path?.endsWith("/pet")).length, 2);
  assert.equal(banner(), undefined);
  assert.equal(view.findAll(node => node.type === "PetProvider")[0].props.value.guide, selected);
});

test("a failed pet retry stays visible and can be retried again", async () => {
  const view = await app([new Error("First failure"), new Error("Still unavailable"), null]);
  const banner = () => view.findAll(node => node.props?.role === "alert")[0];
  for (const message of [/First failure/, /Still unavailable/]) {
    assert.match(banner().props.children[0], message);
    view.findAll(node => node.type === "button", banner())[0].props.onClick();
    view.render();
    await new Promise(resolve => setImmediate(resolve));
    view.render();
  }
  assert.equal(banner(), undefined);
  assert.equal(view.requests.filter(request => request.path?.endsWith("/pet")).length, 3);
});

function openPetEditor(view) {
  const button = view.findAll(node => node.type === "button"
    && Array.isArray(node.props.children) && node.props.children.includes("Edit pet"))[0];
  button.props.onClick();
  view.render();
  return view.findAll(node => node.type === "PetEditor")[0];
}

test("a successful editor refresh clears an initial selected-pet load error", async () => {
  const selected = { id: "fox", version: 1 };
  const view = await app([new Error("Storage unavailable"), selected]);
  assert.match(view.findAll(node => node.props?.role === "alert")[0].props.children[0], /Storage unavailable/);
  await openPetEditor(view).props.onRefreshSelection();
  view.render();
  assert.equal(view.findAll(node => node.props?.role === "alert").length, 0);
  assert.equal(view.findAll(node => node.type === "PetProvider")[0].props.value.guide, selected);
});

test("an older effect failure cannot restore the error after an editor refresh succeeds", async () => {
  const initial = deferred();
  const selected = { id: "fox", version: 1 };
  const view = await app([initial.promise, selected]);
  await openPetEditor(view).props.onRefreshSelection();
  view.render();
  initial.reject(new Error("Old request failed"));
  await settle();
  view.render();
  assert.equal(view.findAll(node => node.props?.role === "alert").length, 0);
  assert.equal(view.findAll(node => node.type === "PetProvider")[0].props.value.guide, selected);
});

for (const reconciled of [false, true]) {
  test(`${reconciled ? "reconciled" : "successful"} pet selection clears an initial load error`, async () => {
    const selected = { id: "fox", version: 1 };
    const responses = [new Error("Storage unavailable"), ...(reconciled ? [new Error("Response lost"), selected] : [selected])];
    const view = await app(responses);
    assert.equal(view.findAll(node => node.props?.role === "alert").length, 1);
    await openPetEditor(view).props.onSelect("fox");
    view.render();
    assert.equal(view.findAll(node => node.props?.role === "alert").length, 0);
    assert.equal(view.findAll(node => node.type === "PetProvider")[0].props.value.guide, selected);
  });
}

async function switchAgent(view) {
  view.findAll(node => node.type === "button" && node.props.title === "Refresh")[0].props.onClick();
  await settle();
  view.render();
  await settle();
  view.render();
}

for (const next of [[], [{ key: "other", name: "Other" }]]) {
  test(`switching to ${next.length ? "another agent" : "no agent"} clears the previous pet error`, async () => {
    const view = await app([new Error("Old error"), null], [[agent], next]);
    assert.equal(view.findAll(node => node.props?.role === "alert").length, 1);
    await switchAgent(view);
    assert.equal(view.findAll(node => node.props?.role === "alert").length, 0);
  });
}

test("a delayed selection from the old agent cannot block or replace the new agent's pet", async () => {
  const delayed = deferred();
  const selected = { id: "otter", version: 1 };
  const view = await app([null, delayed.promise, selected], [[agent], [{ key: "other", name: "Other" }]]);
  const pending = openPetEditor(view).props.onSelect("fox");
  await switchAgent(view);
  assert.equal(view.findAll(node => node.type === "PetProvider")[0].props.value.other, selected);
  delayed.resolve({ id: "fox", version: 1 });
  await pending;
  view.render();
  assert.equal(view.findAll(node => node.type === "PetProvider")[0].props.value.other, selected);
});

test("a delayed failed read from an old agent cannot restore its error after a switch", async () => {
  const delayed = deferred();
  const view = await app([delayed.promise, null], [[agent], [{ key: "other", name: "Other" }]]);
  await switchAgent(view);
  delayed.reject(new Error("Old error"));
  await settle();
  view.render();
  assert.equal(view.findAll(node => node.props?.role === "alert").length, 0);
});

test("a delayed selection remains stale after switching away and back to the same agent", async () => {
  const delayed = deferred();
  const current = { id: "new-fox", version: 2 };
  const view = await app([null, delayed.promise, null, current], [[agent], [{ key: "other", name: "Other" }], [agent]]);
  const pending = openPetEditor(view).props.onSelect("old-fox");
  await switchAgent(view);
  await switchAgent(view);
  assert.equal(view.findAll(node => node.type === "PetProvider")[0].props.value.guide, current);
  delayed.resolve({ id: "old-fox", version: 1 });
  await pending;
  view.render();
  assert.equal(view.findAll(node => node.type === "PetProvider")[0].props.value.guide, current);
});

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
