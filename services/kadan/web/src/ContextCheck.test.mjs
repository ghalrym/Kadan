import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { runInNewContext } from "node:vm";
import ts from "typescript";

const source = ts.transpileModule(readFileSync(new URL("./ContextCheck.tsx", import.meta.url), "utf8"), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX },
}).outputText;
function harness() {
  const slots = [], requests = [], effects = [];
  let cursor = 0, tree, resolve, reject;
  const props = { agentKey: "guide", conversationId: "chain", version: 1, task: "Keep my draft", persona: "", references: [], disabled: false };
  const react = {
    useState(initial) { const i = cursor++; if (!(i in slots)) slots[i] = initial; return [slots[i], value => slots[i] = value]; },
    useEffect(effect, dependencies) { const i = cursor++; if (slots[i] !== dependencies[0]) { slots[i] = dependencies[0]; effects.push(effect); } },
    useRef(initial) { const i = cursor++; return slots[i] ??= { current: initial }; },
  };
  const jsx = (type, props) => ({ type, props });
  const exports = {};
  runInNewContext(source, { exports, require: name => ({
    react, "react/jsx-runtime": { jsx, jsxs: jsx },
    "./api": { agentPath: key => `/agents/${key}`, api: (path, body) => {
      requests.push({ path, body }); return new Promise((yes, no) => { resolve = yes; reject = no; });
    } },
  })[name] });
  function render() { cursor = 0; tree = exports.default(props); effects.splice(0).forEach(effect => effect()); }
  const text = (node = tree) => typeof node === "string" ? node : !node || typeof node !== "object" ? "" : [node.props?.children].flat(Infinity).map(child => text(child ?? null)).join(" ");
  function button(node = tree) { if (node?.type === "button") return node; return [node?.props?.children].flat(Infinity).map(child => child && typeof child === "object" ? button(child) : null).find(Boolean); }
  render();
  return { props, requests, render, text, check: () => button().props.onClick(), resolve: value => resolve(value), reject: error => reject(error) };
}
const settle = () => new Promise(resolve => setImmediate(resolve));
const advice = { context_limit: 4096, context_source: "effective", approximate_character_budget: 9216, task_characters: 13, instruction_characters: 20, selected_characters: 0, tool_schema_characters: 500, minimum_exceeds_budget: false, selected_material_risk: false };

test("context check uses the draft without sending or clearing it", async () => {
  const view = harness(); view.check();
  assert.equal(view.requests.length, 1);
  assert.ok(view.requests[0].path.endsWith("/context-advice"));
  assert.equal(view.requests[0].body.content, "Keep my draft");
  view.resolve(advice); await settle(); view.render();
  assert.match(view.text(), /do not guarantee/);
  assert.equal(view.props.task, "Keep my draft");
  view.props.persona = "changed"; view.render();
  assert.doesNotMatch(view.text(), /Context measurements/);
  view.props.persona = ""; view.render();
  assert.doesNotMatch(view.text(), /Context measurements/);
});

test("late context results disappear when selections change", async () => {
  const view = harness(); view.check();
  view.props.workflow = { id: "workflow", version: 2 }; view.render();
  view.resolve(advice); await settle(); view.render();
  assert.doesNotMatch(view.text(), /Context measurements/);
  assert.match(view.text(), /Check workflow context/);
});

test("context errors preserve the draft and allow another manual check", async () => {
  const view = harness(); view.check(); view.reject(new Error("Selected reference changed"));
  await settle(); view.render();
  assert.match(view.text(), /Selected reference changed/);
  assert.equal(view.props.task, "Keep my draft");
  view.props.task = "Edited draft"; view.render();
  assert.doesNotMatch(view.text(), /Selected reference changed/);
  view.check(); assert.equal(view.requests.length, 2);
});
