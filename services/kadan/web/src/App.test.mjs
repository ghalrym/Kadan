import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { registerHooks } from "node:module";
import { afterEach, beforeEach, test } from "node:test";
import ts from "typescript";

const moduleUrl = (source) => `data:text/javascript,${encodeURIComponent(source)}`;
// Execute App's function and callbacks with state and child boundaries replaced.
// Effects are disabled: no mounted tree, browser, timers or service requests.
const reactUrl = moduleUrl(`
  export const state = [];
  let cursor = 0;
  export const begin = () => { cursor = 0; };
  export const useEffect = () => {};
  export const useCallback = callback => callback;
  export const useRef = current => ({ current });
  export function useState(initial) {
    const index = cursor++;
    if (!(index in state)) state[index] = typeof initial === 'function' ? initial() : initial;
    return [state[index], next => {
      state[index] = typeof next === 'function' ? next(state[index]) : next;
    }];
  }
`);
const jsxUrl = moduleUrl(`
  export const elements = [];
  export const jsx = (type, props, key) => {
    const element = { type, props, key };
    elements.push(element);
    return element;
  };
  export const jsxs = jsx;
`);
const appUrl = new URL("./App.tsx", import.meta.url).href;
const mocks = {
  react: reactUrl,
  "react/jsx-runtime": jsxUrl,
  "react-dom": moduleUrl("export const flushSync = callback => callback();"),
  "lucide-react": moduleUrl("export const BookUser='', FolderOpen='', Menu='', MessageCircle='', Plus='', Palette='', Pencil='', Server='', RefreshCw='', X='';"),
  "./api": moduleUrl("export const agentPath = () => ''; export const api = () => { throw Error('Unexpected request'); }; export const allResources = api;"),
  "./Pet": moduleUrl("export default 'Pet'; export const agentPet = () => null; export const useMood = () => [];"),
  "./pet": moduleUrl("export const agentHue = () => 0;"),
  "./PetContext": moduleUrl("export const PetContext = { Provider: 'Provider' }; export const useAgentPet = () => null;"),
  ...Object.fromEntries(["Chat", "Workspace", "Personas", "Hatch", "PetEditor", "RenameAgent", "RunnerAssignment"].map(name => [`./${name}`, moduleUrl(`export default '${name}';`)])),
};
const loader = registerHooks({
  resolve(specifier, context, nextResolve) {
    if (context.parentURL === appUrl) {
      const url = mocks[specifier] ?? (specifier === "./agentRuns" ? new URL("./agentRuns.ts", import.meta.url).href : null);
      if (url) return { url, shortCircuit: true };
    }
    return nextResolve(specifier, context);
  },
  load(url, context, nextLoad) {
    if (url !== appUrl) return nextLoad(url, context);
    const { outputText } = ts.transpileModule(readFileSync(new URL(url), "utf8"), {
      compilerOptions: { module: ts.ModuleKind.ESNext, jsx: ts.JsxEmit.ReactJSX },
    });
    return { format: "module", source: outputText, shortCircuit: true };
  },
});
const { state, begin } = await import(reactUrl);
const { elements } = await import(jsxUrl);
const { default: App } = await import(appUrl);
loader.deregister();

const originals = new Map(["localStorage", "matchMedia"].map(key => [key, Object.getOwnPropertyDescriptor(globalThis, key)]));
// Slots follow App's useState calls; refs and effects do not consume these slots.
const AGENTS = 0, RENAMING_AGENT = 3, RUNNER_AGENT = 4, AGENT_KEY = 5, CHAIN = 7, LOADING = 11, BUSY = 12;
const original = { id: "original", agent_key: "agent", version: 3, messages: [] };
const replacement = { ...original, id: "replacement", version: 1 };
beforeEach(() => {
  state.length = 0;
  state[AGENTS] = [{ key: "agent", name: "Agent" }];
  state[AGENT_KEY] = "agent";
  state[CHAIN] = original;
  Object.defineProperty(globalThis, "localStorage", { configurable: true, value: { getItem: () => "agent" } });
  Object.defineProperty(globalThis, "matchMedia", { configurable: true, value: () => ({ matches: false }) });
});
afterEach(() => {
  for (const [key, descriptor] of originals) {
    if (descriptor) Object.defineProperty(globalThis, key, descriptor);
    else delete globalThis[key];
  }
});
function chat() {
  begin();
  elements.length = 0;
  App();
  const result = elements.find(element => element.type === "Chat");
  assert.ok(result, "App must provide the selected chat");
  return result;
}

test("changing conversations remounts chat while new snapshots of the same conversation keep its identity", () => {
  const first = chat();
  state[CHAIN] = { ...original, version: 4 };
  assert.equal(chat().key, first.key);
  state[CHAIN] = replacement;
  assert.notEqual(chat().key, first.key);
});

test("a previous chat callback cannot overwrite the newly selected conversation", () => {
  const previous = chat().props.onChange;
  state[CHAIN] = replacement;
  chat();
  previous({ ...original, version: 10 });
  assert.equal(state[CHAIN], replacement);
});

test("the selected chat accepts newer messages and rejects late older snapshots", () => {
  const update = chat().props.onChange;
  const newer = { ...original, version: 4, messages: [{ id: "new-message" }] };
  update(newer);
  assert.equal(state[CHAIN], newer);
  update(original);
  assert.equal(state[CHAIN], newer);
});


for (const slot of [LOADING, BUSY]) {
  test(`rename is disabled while ${slot === LOADING ? "loading agents" : "busy"}`, () => {
    state[LOADING] = false;
    state[BUSY] = false;
    chat();
    const rename = () => elements.find(element => element.type === "button"
      && Array.isArray(element.props.children) && element.props.children.includes("Rename agent"));
    assert.equal(rename().props.disabled, false);
    state[slot] = true;
    chat();
    assert.equal(rename().props.disabled, true);
  });
}

test("renaming updates only the selected agent's name without replacing the conversation", () => {
  const other = { key: "other", name: "Other" };
  state[AGENTS] = [{ key: "agent", name: "Agent", selected_pet_id: "pet" }, other];
  state[RENAMING_AGENT] = "agent";
  const previousChat = chat();
  const rename = elements.find(element => element.type === "RenameAgent");
  assert.ok(rename);
  rename.props.onUpdated({ key: "agent", name: "Renamed" });
  assert.deepEqual(state[AGENTS][0], { key: "agent", name: "Renamed", selected_pet_id: "pet" });
  assert.equal(state[AGENTS][1], other);
  assert.equal(state[AGENT_KEY], "agent");
  assert.equal(state[CHAIN], original);
  const updatedChat = chat();
  assert.equal(updatedChat.key, previousChat.key);
  assert.equal(updatedChat.props.agentName, "Renamed");
});

test("runner assignment belongs to the selected agent and closes on agent change", () => {
  state[LOADING] = false;
  chat();
  const open = elements.find(element => element.type === "button"
    && Array.isArray(element.props.children) && element.props.children.includes("Runner assignment"));
  assert.equal(open.props.disabled, false);
  open.props.onClick();
  chat();
  const assignment = elements.find(element => element.type === "RunnerAssignment");
  assert.equal(assignment.key, "agent");
  assert.equal(assignment.props.agent.key, "agent");
  assignment.props.onClose();
  assert.equal(state[RUNNER_AGENT], "");
  state[RUNNER_AGENT] = "agent";
  state[AGENTS].push({ key: "other", name: "Other" });
  state[AGENT_KEY] = "other";
  chat();
  assert.equal(elements.some(element => element.type === "RunnerAssignment"), false);
});
