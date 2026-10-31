import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { registerHooks } from "node:module";
import { test } from "node:test";
import ts from "typescript";

// Exercise composer decisions and callbacks without mounting a UI or using a microphone.
const modules = {
  react: `export const useEffect = () => {};
    export const useLayoutEffect = () => {};
    export const useRef = current => ({ current });
    export const useState = value => [value, () => {}];`,
  "./AttachmentPicker": "export default function AttachmentPicker() {}",
  "./VoiceCallControl": "export default function VoiceCallControl() {}",
  "./voiceInput": "export const microphoneUnavailable = () => null; export class VoiceInput {}",
};
const composerUrl = new URL("./Composer.tsx", import.meta.url).href;
const loader = registerHooks({
  resolve(specifier, context, nextResolve) {
    if (context.parentURL === composerUrl && specifier in modules) {
      return { url: `data:text/javascript,${encodeURIComponent(modules[specifier])}`, shortCircuit: true };
    }
    return nextResolve(specifier, context);
  },
  load(url, context, nextLoad) {
    if (url !== composerUrl) return nextLoad(url, context);
    const { outputText } = ts.transpileModule(readFileSync(new URL(url), "utf8"), {
      compilerOptions: { module: ts.ModuleKind.ESNext, jsx: ts.JsxEmit.ReactJSX },
    });
    return { format: "module", source: outputText, shortCircuit: true };
  },
});
const { default: Composer } = await import(composerUrl);
loader.deregister();

const status = {
  tsuchi: { state: "ready", message: "Connected", browser_url: "http://localhost:8620" },
  hikari: {
    state: "model_required", model_id: "lfm3-1.7b-bf16",
    message: "Download the LFM3 model", settings_url: "http://localhost:8088/settings",
  },
};
function compose(overrides = {}) {
  return Composer({
    attachments: { ready: true, entries: [] },
    agentName: "Haru", value: "My pending question", onChange() {}, onSend() {},
    sending: false, blocked: false, status, personas: [{ id: "persona", title: "Me" }],
    persona: "persona", onPersona() {}, pet: null, voiceCall: { active: false },
    onDictating() {}, ...overrides,
  });
}
function elements(node) {
  if (!node || typeof node !== "object") return [];
  return [node, ...[node.props?.children].flat(Infinity).flatMap(elements)];
}
const find = (tree, type) => elements(tree).find(node => node.type === type);
const prompt = tree => elements(tree).find(node => node.props?.className === "plan-prompt");

test("missing model shows an explicit separate-tab route to downloads", () => {
  const notice = elements(compose()).find(node => node.props?.className === "attention");
  assert.equal(notice.props.href, status.hikari.settings_url);
  assert.equal(notice.props.target, "_blank");
  assert.equal(notice.props.onClick, undefined);
  assert.ok([notice.props.children].flat(Infinity).includes("Download a model · notes only"));
});

test("missing model leaves the draft editable and preserves explicit note sending", () => {
  const sends = [], edits = [];
  const tree = compose({ onSend: value => sends.push(value), onChange: value => edits.push(value) });
  assert.deepEqual(sends, []);
  assert.deepEqual(edits, []);
  const field = find(tree, "textarea");
  assert.equal(field.props.value, "My pending question");
  assert.equal(field.props.disabled, false);
  const send = elements(tree).find(node => node.props?.className === "send");
  assert.equal(send.props["aria-label"], "Save note");
  assert.equal(send.props.disabled, false);
  find(tree, "form").props.onSubmit({ preventDefault() {} });
  assert.deepEqual(sends, [false]);
});

test("ready model removes the prompt and keeps chat and persona selection", () => {
  const sends = [];
  const tree = compose({
    status: { ...status, hikari: { ...status.hikari, state: "ready" } },
    onSend: value => sends.push(value),
  });
  assert.equal(prompt(tree), undefined);
  assert.equal(find(tree, "textarea").props.value, "My pending question");
  assert.equal(find(tree, "textarea").props.placeholder, "Message Haru");
  assert.equal(find(tree, "select").props.value, "persona");
  find(tree, "form").props.onSubmit({ preventDefault() {} });
  assert.deepEqual(sends, [true]);
});

test("unknown and unavailable states are not presented as no-plan states", () => {
  assert.equal(prompt(compose({ status: null })), undefined);
  for (const [state, label] of [
    ["model_required", "Download a model · notes only"],
    ["unavailable", "Hikari offline · notes only"],
  ]) {
    const tree = compose({ status: { ...status, hikari: { ...status.hikari, state } } });
    assert.equal(prompt(tree), undefined);
    assert.ok(find(tree, "a").props.children.includes(label));
  }
  const tree = compose({ status: { ...status, tsuchi: { ...status.tsuchi, state: "unavailable" } } });
  assert.equal(prompt(tree), undefined);
  assert.ok(find(tree, "a").props.children.includes("Storage offline"));
  assert.equal(find(tree, "textarea").props.disabled, true);
});

test("missing settings link keeps guidance visible without a dead action", () => {
  const tree = compose({ status: { ...status, hikari: { ...status.hikari, settings_url: null } } });
  const notice = elements(tree).find(node => node.props?.className === "attention");
  assert.ok(notice);
  assert.equal(notice.props.href, undefined);
  assert.equal(notice.props["aria-disabled"], true);
});


test("response cancellation has a visible label and explains independent coding cancellation", () => {
  let stopped = 0;
  const tree = compose({ onStop: () => stopped++ });
  const stop = elements(tree).find(node => node.props?.["aria-label"] === "Stop response");
  assert.ok(stop.props.children.includes("Stop response"));
  const help = elements(tree).find(node => node.props?.id === stop.props["aria-describedby"]);
  assert.match(help.props.children, /Coding keeps running; use Stop coding in Coding/);
  stop.props.onClick();
  assert.equal(stopped, 1);
  const pending = elements(compose({ onStop() {}, stopping: true })).find(node => node.props?.["aria-label"] === "Stop response");
  assert.equal(pending.props.disabled, true);
  assert.ok(pending.props.children.includes("Stopping…"));
});


test("file selection in notes-only mode resets the picker without sending", () => {
  const added = [], sends = [];
  const tree = compose({
    attachments: { ready: true, entries: [], add: files => added.push(files) },
    onSend: value => sends.push(value),
  });
  const attach = elements(tree).find(node => node.props?.["aria-label"] === "Attach files");
  assert.equal(attach.props.disabled, false);
  const selected = new File(["test"], "notes.txt");
  const target = { files: [selected], value: "notes.txt" };
  find(tree, "input").props.onChange({ target });
  assert.deepEqual(added, [[selected]]);
  assert.equal(target.value, "");
  assert.deepEqual(sends, []);
  find(tree, "form").props.onSubmit({ preventDefault() {} });
  assert.deepEqual(sends, [false]);
});

test("unresolved uploads block both sending choices and Enter", () => {
  const sends = [];
  const tree = compose({
    attachments: { ready: false, entries: [] },
    status: { ...status, hikari: { state: "ready" } },
    onSend: value => sends.push(value),
  });
  for (const label of ["Send message", "Save as note without a reply"]) {
    assert.equal(elements(tree).find(node => node.props?.["aria-label"] === label).props.disabled, true);
  }
  find(tree, "form").props.onSubmit({ preventDefault() {} });
  find(tree, "textarea").props.onKeyDown({ key: "Enter", shiftKey: false, nativeEvent: {}, preventDefault() {} });
  assert.deepEqual(sends, []);
});
