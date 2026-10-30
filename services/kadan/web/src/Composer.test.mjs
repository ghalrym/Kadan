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
    state: "tier_required", tier: null, model_id: null,
    message: "Choose a tier", settings_url: "http://localhost:8088/settings",
  },
};
function compose(overrides = {}) {
  return Composer({
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

test("no plan shows an explicit explanation and a separate-tab route to selection", () => {
  const notice = prompt(compose());
  assert.equal(notice.props.role, "status");
  assert.equal(find(notice, "strong").props.children, "Select a plan to enable replies");
  assert.match(find(notice, "p").props.children, /keep writing or save a note/);
  const link = find(notice, "a");
  assert.equal(link.props.href, status.hikari.settings_url);
  assert.equal(link.props.target, "_blank");
  assert.equal(link.props.onClick, undefined);
});

test("no-plan prompt leaves the draft editable and preserves explicit note sending", () => {
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

test("ready plan removes the prompt and keeps chat and persona selection", () => {
  const sends = [];
  const tree = compose({
    status: { ...status, hikari: { ...status.hikari, state: "ready", tier: "go" } },
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
  assert.ok(prompt(tree));
  assert.equal(find(prompt(tree), "a"), undefined);
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
