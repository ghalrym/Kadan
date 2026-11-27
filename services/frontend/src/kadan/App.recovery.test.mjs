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
  "lucide-react": moduleUrl("export const Egg='', Menu='';"),
  "./api": moduleUrl("export const agentPath = () => ''; export const api = () => { throw Error('Unexpected request'); }; export const allResources = api;"),
  "./desktop": moduleUrl("export const isDesktopApp = () => false; export const getDesktopServerConfig = async () => ({ active_host: '', saved_servers: [] });"),
  "./Pet": moduleUrl("export default 'Pet'; export const agentPet = () => null; export const useMood = () => [];"),
  "./pet": moduleUrl("export const agentHue = () => 0;"),
  "./PetContext": moduleUrl("export const PetContext = { Provider: 'Provider' }; export const useAgentPet = () => null;"),
  ...Object.fromEntries(["Chat", "Hatch", "PetEditor", "PersonaEditor", "RenameAgent", "RunnerAssignment", "Plugins", "Sidebar", "SpaceLibrary", "ConfirmDialog", "AgentPage", "ServerSettings"].map(name => [`./${name}`, moduleUrl(`export default '${name}';`)])),
};
const loader = registerHooks({
  resolve(specifier, context, nextResolve) {
    if (context.parentURL === appUrl) {
      const url = mocks[specifier] ?? (["./agentRuns", "./conversations"].includes(specifier) ? new URL(`${specifier}.ts`, import.meta.url).href : null);
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

const originals = new Map(["matchMedia", "document"].map(key => [key, Object.getOwnPropertyDescriptor(globalThis, key)]));
// Slots follow App's useState calls; refs and effects do not consume these slots.
const AGENT = 0, EDITING_PERSONA = 3, RENAMING = 4, ASSIGNING_RUNNER = 5, CONVERSATIONS = 7, SELECTION = 8,
  CHAIN = 9, PAGE = 10, LOADING = 15, BUSY = 16, RESOURCE_OWNER = 22;
const original = { id: "original", agent_key: "agent", title: "New chat", version: 3, messages: [] };
const replacement = { ...original, id: "replacement", version: 1 };
const select = (conversation) => {
  state[CHAIN] = conversation;
  state[SELECTION] = { owner: "agent", id: conversation.id };
};
beforeEach(() => {
  state.length = 0;
  state[AGENT] = { key: "agent", name: "Agent" };
  select(original);
  state[RESOURCE_OWNER] = "agent";
  Object.defineProperty(globalThis, "matchMedia", { configurable: true, value: () => ({ matches: false }) });
  Object.defineProperty(globalThis, "document", { configurable: true, value: {} });
});
afterEach(() => {
  for (const [key, descriptor] of originals) {
    if (descriptor) Object.defineProperty(globalThis, key, descriptor);
    else delete globalThis[key];
  }
});
function render() {
  begin();
  elements.length = 0;
  App();
}
function find(type) {
  return elements.find(element => element.type === type);
}
function chat() {
  render();
  const result = find("Chat");
  assert.ok(result, "App must provide the selected chat");
  return result;
}

test("changing conversations remounts chat while new snapshots of the same conversation keep its identity", () => {
  const first = chat();
  state[CHAIN] = { ...original, version: 4 };
  assert.equal(chat().key, first.key);
  select(replacement);
  assert.notEqual(chat().key, first.key);
});

test("a previous chat callback cannot overwrite the newly selected conversation", () => {
  const previous = chat().props.onChange;
  select(replacement);
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

test("renaming updates the agent's name without replacing the conversation", () => {
  state[AGENT] = { key: "agent", name: "Agent", selected_pet_id: "pet" };
  state[RENAMING] = true;
  const previousChat = chat();
  const rename = elements.find(element => element.type === "RenameAgent");
  assert.ok(rename);
  rename.props.onUpdated({ key: "agent", name: "Renamed" });
  assert.deepEqual(state[AGENT], { key: "agent", name: "Renamed", selected_pet_id: "pet" });
  assert.equal(state[CHAIN], original);
  const updatedChat = chat();
  assert.equal(updatedChat.key, previousChat.key);
  assert.equal(updatedChat.props.agentName, "Renamed");
});

test("the agent page opens runner assignment, which closes again", () => {
  state[LOADING] = false;
  render();
  find("Sidebar").props.onPage("agent");
  render();
  const page = find("AgentPage");
  assert.equal(page.props.disabled, false);
  page.props.onRunner();
  render();
  const assignment = find("RunnerAssignment");
  assert.equal(assignment.props.agent.key, "agent");
  assignment.props.onClose();
  assert.equal(state[ASSIGNING_RUNNER], false);
  render();
  assert.equal(find("RunnerAssignment"), undefined);
});

test("hatching is only offered while there is no agent", () => {
  chat();
  assert.equal(find("Hatch"), undefined);
  state[AGENT] = null;
  state[LOADING] = false;
  render();
  assert.ok(find("Hatch"));
  assert.equal(find("Chat"), undefined);
});

test("plugins open while chat is busy and returning keeps the conversation", () => {
  state[BUSY] = true;
  const previous = chat();
  const sidebar = find("Sidebar");
  sidebar.props.onPage("plugins");
  render();
  assert.ok(find("Plugins"));
  assert.equal(find("Chat"), undefined);
  find("Sidebar").props.onOpenChat("original");
  assert.equal(chat().key, previous.key);
  assert.equal(state[CHAIN], original);
});

test("opening a chat from the sidebar selects it and waits for its messages", () => {
  state[CONVERSATIONS] = [{ id: "replacement", message_count: 1, updated_at: "2026-01-01" }];
  chat().props;
  find("Sidebar").props.onOpenChat("replacement");
  assert.deepEqual(state[SELECTION], { owner: "agent", id: "replacement" });
  render();
  assert.equal(find("Chat"), undefined, "the previous chat must not show under the new selection");
  state[CHAIN] = replacement;
  assert.equal(chat().props.chain, replacement);
});

test("Space opens the library for the current agent and returning keeps the chat", () => {
  const previous = chat();
  find("Sidebar").props.onPage("space");
  render();
  assert.equal(find("SpaceLibrary").props.agentKey, "agent");
  assert.equal(find("Chat"), undefined);
  find("Sidebar").props.onOpenChat("original");
  assert.equal(chat().key, previous.key);
});

test("chat updates keep the sidebar summary current", () => {
  const update = chat().props.onChange;
  update({ ...original, version: 4, messages: [{ id: "m", role: "user", content: "Plan a garden" }] });
  assert.equal(state[CONVERSATIONS][0].title, "Plan a garden");
  assert.equal(state[CONVERSATIONS][0].message_count, 1);
});

test("chat and the agent page open the single persona editor modal, which closes again", () => {
  state[LOADING] = false;
  chat().props.onEditPersona();
  render();
  const editor = find("PersonaEditor");
  assert.equal(editor.props.agentKey, "agent");
  assert.equal(editor.props.agentName, "Agent");
  editor.props.onClose();
  assert.equal(state[EDITING_PERSONA], false);
  render();
  assert.equal(find("PersonaEditor"), undefined);
});

for (const slot of [LOADING, BUSY]) {
  test(`rename is disabled while ${slot === LOADING ? "loading agents" : "busy"}`, () => {
    state[LOADING] = false;
    state[BUSY] = false;
    state[PAGE] = { name: "agent" };
    render();
    assert.equal(find("AgentPage").props.disabled, false);
    state[slot] = true;
    render();
    assert.equal(find("AgentPage").props.disabled, true);
  });
}
