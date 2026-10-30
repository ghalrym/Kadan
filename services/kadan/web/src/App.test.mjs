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
async function app() {
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
    useCallback: callback => callback,
    useEffect(effect, dependencies) {
      const index = cursor++;
      if (!slots[index] || dependencies.some((value, i) => value !== slots[index][i])) {
        slots[index] = dependencies;
        effects.push(effect);
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
        return [persona];
      },
      async api(path) {
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
