import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import ts from "typescript";

// Exercise the component's isolated handlers with inert child/effect boundaries.
const source = ts.transpileModule(readFileSync(new URL("./Composer.tsx", import.meta.url), "utf8"), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, target: ts.ScriptTarget.ES2022 },
}).outputText;
function harness({ chatReady = true, uploadsReady = true } = {}) {
  const elements = [], sends = [], added = [];
  const jsx = (type, props) => { const element = { type, props }; elements.push(element); return element; };
  const mocks = {
    react: { useEffect() {}, useLayoutEffect() {}, useRef: current => ({ current }), useState: initial => [initial, () => {}] },
    "react/jsx-runtime": { jsx, jsxs: jsx },
    "lucide-react": Object.fromEntries(["ArrowUp", "ArrowUpRight", "ChevronDown", "Mic", "NotebookPen", "Paperclip", "Square", "UserRound", "X"].map(key => [key, key])),
    "./AttachmentPicker": { default: "AttachmentPicker" }, "./VoiceCallControl": { default: "VoiceCallControl" },
    "./voiceInput": { microphoneUnavailable: () => null },
  };
  const exports = {};
  new Function("require", "exports", source)(name => {
    assert.ok(name in mocks, `Unexpected dependency ${name}`);
    return mocks[name];
  }, exports);
  const attachments = { ready: uploadsReady, entries: [], add: files => added.push(files) };
  exports.default({ agentName: "Agent", value: "Read this", onChange() {}, onSend: value => sends.push(value), sending: false,
    blocked: false, status: { tsuchi: { state: "ready" }, hikari: { state: chatReady ? "ready" : "offline" } },
    personas: [], persona: "", onPersona() {}, pet: null, voiceCall: { active: false }, onDictating() {}, attachments });
  return { elements, sends, added };
}

test("file selection works in notes-only mode and resets the input for reselecting a file", () => {
  const h = harness({ chatReady: false });
  assert.equal(h.elements.find(item => item.props["aria-label"] === "Attach files").props.disabled, false);
  const selected = new File(["test"], "notes.txt");
  const target = { files: [selected], value: "notes.txt" };
  h.elements.find(item => item.type === "input" && item.props.type === "file").props.onChange({ target });
  assert.deepEqual(h.added, [[selected]]);
  assert.equal(target.value, "");
  h.elements.find(item => item.type === "form").props.onSubmit({ preventDefault() {} });
  assert.deepEqual(h.sends, [false]);
});

test("unresolved uploads disable both send choices and prevent Enter and form submission", () => {
  const h = harness({ uploadsReady: false });
  for (const label of ["Send message", "Save as note without a reply"]) {
    assert.equal(h.elements.find(item => item.props["aria-label"] === label).props.disabled, true);
  }
  h.elements.find(item => item.type === "form").props.onSubmit({ preventDefault() {} });
  h.elements.find(item => item.type === "textarea").props.onKeyDown({ key: "Enter", shiftKey: false, nativeEvent: {}, preventDefault() {} });
  assert.deepEqual(h.sends, []);
});
