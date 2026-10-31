import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { registerHooks } from "node:module";
import { afterEach, beforeEach, test } from "node:test";
import { runInNewContext } from "node:vm";
import ts from "typescript";
import * as agentRuns from "./agentRuns.ts";
import * as outbox from "./outbox.ts";

const state = [];
const reactUrl = `data:text/javascript,${encodeURIComponent("export const state = []; export const useState = value => [value, next => state.push(next)];")}`;
const componentUrl = new URL("./ChatExport.tsx", import.meta.url).href;
const loader = registerHooks({
  resolve(specifier, context, nextResolve) {
    if (context.parentURL === componentUrl) {
      if (specifier === "react") return { url: reactUrl, shortCircuit: true };
      if (specifier === "./chatExport") return { url: new URL("./chatExport.ts", import.meta.url).href, shortCircuit: true };
    }
    return nextResolve(specifier, context);
  },
  load(url, context, nextLoad) {
    if (url !== componentUrl) return nextLoad(url, context);
    const { outputText } = ts.transpileModule(readFileSync(new URL(url), "utf8"), {
      compilerOptions: { module: ts.ModuleKind.ESNext, jsx: ts.JsxEmit.ReactJSX },
    });
    return { format: "module", source: outputText, shortCircuit: true };
  },
});
const { state: updates } = await import(reactUrl);
const { default: ChatExport } = await import(componentUrl);
loader.deregister();

const originals = new Map(["window", "document", "setTimeout"].map(key => [key, Object.getOwnPropertyDescriptor(globalThis, key)]));
const createUrl = URL.createObjectURL;
const revokeUrl = URL.revokeObjectURL;
let blob, link, release;
beforeEach(() => {
  state.length = 0;
  updates.length = 0;
  link = { click: () => state.push("click"), remove: () => state.push("remove") };
  Object.defineProperty(globalThis, "window", { configurable: true, value: { location: { origin: "https://kadan.example" } } });
  Object.defineProperty(globalThis, "document", { configurable: true, value: {
    createElement: type => { assert.equal(type, "a"); return link; },
    body: { appendChild: element => { assert.equal(element, link); state.push("append"); } },
  } });
  Object.defineProperty(globalThis, "setTimeout", { configurable: true, value: callback => { release = callback; } });
  URL.createObjectURL = value => { blob = value; return "blob:snapshot"; };
  URL.revokeObjectURL = value => state.push(`revoke:${value}`);
});
afterEach(() => {
  for (const [key, descriptor] of originals) {
    if (descriptor) Object.defineProperty(globalThis, key, descriptor);
    else delete globalThis[key];
  }
  URL.createObjectURL = createUrl;
  URL.revokeObjectURL = revokeUrl;
});
const chain = messages => ({
  id: "saved-conversation", agent_key: "haru", title: "Chat", version: 3,
  updated_at: "2026-10-23T12:00:00Z", messages,
});
const saved = { role: "user", content: "Confirmed saved note", created_at: "2026-10-23T12:00:00Z", attachment_ids: [], resource_ids: [], sources: [] };
const button = messages => ChatExport({ chain: chain(messages), agentName: "Haru" }).props.children[0];

test("download control names its format and disables empty or hidden-only history", () => {
  for (const messages of [[], [{ ...saved, role: "tool" }]]) {
    const control = button(messages);
    assert.equal(control.props["aria-label"], "Download saved chat as Markdown");
    assert.equal(control.props.disabled, true);
    control.props.onClick();
    assert.deepEqual(state, []);
    assert.deepEqual(updates, []);
  }
});

test("download handler saves the confirmed snapshot with a stable safe filename and releases the URL", async () => {
  const control = button([saved]);
  assert.equal(control.props.disabled, false);
  control.props.onClick();
  assert.equal(blob.type, "text/markdown;charset=utf-8");
  assert.ok((await blob.text()).includes(saved.content));
  assert.equal(link.download, "kadan-haru-saved-conversation.md");
  assert.equal(link.href, "blob:snapshot");
  assert.deepEqual(state, ["append", "click", "remove"]);
  assert.deepEqual(updates, [""]);
  release();
  assert.equal(state.at(-1), "revoke:blob:snapshot");
});

test("failed download starts report an error and clean up for retry", () => {
  const control = button([saved]);
  link.click = () => { throw new Error("Download failed"); };
  control.props.onClick();
  assert.deepEqual(updates, ["", "Could not start chat download. Try again."]);
  assert.deepEqual(state, ["append", "remove"]);
  release();
  assert.equal(state.at(-1), "revoke:blob:snapshot");
  link.click = () => state.push("click");
  control.props.onClick();
  assert.equal(updates.at(-1), "");
  assert.ok(state.includes("click"));
});

test("Chat passes saved history to export while its draft and optimistic outgoing message remain separate", () => {
  const snapshot = chain([saved]);
  const outgoing = { ...saved, id: "unconfirmed", content: "Still sending" };
  let slot = 0;
  const jsx = (type, props) => ({ type, props });
  const modules = {
    react: {
      Fragment: "Fragment", useEffect() {}, useLayoutEffect() {}, useCallback: callback => callback,
      useRef: current => ({ current }),
      useState: initial => [slot++ === 13 ? outgoing : typeof initial === "function" ? initial() : initial, () => {}],
    },
    "react/jsx-runtime": { jsx, jsxs: jsx },
    "lucide-react": {},
    "./Markdown": { default: "Markdown" },
    "./Composer": { default: "Composer", readiness: () => ({ chat: true, storage: true }) },
    "./Pet": { default: "Pet", useMood: () => ["idle", () => {}] },
    "./Activity": {}, "./CodingIndicator": {},
    "./attachments": { ChatAttachments: class { entries = []; } },
    "./SaveChatResult": {},
    "./RequestReferences": { default: "RequestReferences" },
    "./AgentCard": {}, "./ComputerViewer": {}, "./ChatExport": { default: "ChatExport" },
    "./VoiceCallControl": { useVoiceCall: () => ({ active: false }) },
    "./presence": {}, "./PetContext": { useAgentPet: () => null },
    "./api": {}, "./uuid": {},
    "./agentRuns": agentRuns,
    "./outbox": outbox,
  };
  const exports = {};
  const { outputText } = ts.transpileModule(readFileSync(new URL("./Chat.tsx", import.meta.url), "utf8"), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX },
  });
  runInNewContext(outputText, {
    exports, require: name => { assert.ok(name in modules, name); return modules[name]; },
    sessionStorage: { getItem: key => key === outbox.chatStorageKeys("haru", snapshot.id).draft ? "Unsent draft" : null },
  });
  const tree = exports.default({ agentKey: "haru", agentName: "Haru", chain: snapshot, personas: [], status: null });
  function elements(node) {
    return !node || typeof node !== "object" ? [] : [node, ...[node.props?.children].flat(Infinity).flatMap(elements)];
  }
  const nodes = elements(tree);
  assert.ok(nodes.some(node => node.type === "article" && node.props.className.includes("sending")));
  assert.equal(nodes.find(node => node.type === "Composer").props.value, "Unsent draft");
  const download = nodes.find(node => node.type === "ChatExport");
  assert.equal(download.props.chain, snapshot);
  assert.deepEqual(download.props.chain.messages, [saved]);
});
