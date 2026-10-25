import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { runInNewContext } from "node:vm";
import ts from "typescript";

const source = ts.transpileModule(readFileSync(new URL("./ContextReceipt.tsx", import.meta.url), "utf8"), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX },
}).outputText;
const settle = () => new Promise(resolve => setImmediate(resolve));
function receiptView() {
  const slots = [], effects = [], requests = [];
  let cursor = 0, tree;
  const jsx = (type, props) => ({ type, props });
  const react = {
    useState(initial) {
      const index = cursor++;
      if (!(index in slots)) slots[index] = initial;
      return [slots[index], value => { slots[index] = value; }];
    },
    useRef(initial) { return slots[cursor++] ??= { current: initial }; },
    useEffect(effect, dependencies) {
      const index = cursor++;
      const previous = slots[index];
      if (!previous || dependencies.some((value, i) => value !== previous.dependencies[i])) {
        const slot = slots[index] = { dependencies };
        effects.push(() => { previous?.cleanup?.(); slot.cleanup = effect(); });
      }
    },
  };
  const exports = {};
  const modules = {
    react, "react/jsx-runtime": { jsx, jsxs: jsx },
    "./api": {
      agentPath: key => `/agents/${key}`,
      api(path) {
        return new Promise((resolve, reject) => requests.push({ path, resolve, reject }));
      },
    },
  };
  runInNewContext(source, { exports, require: name => modules[name] });
  function render(runId = "request") {
    cursor = 0;
    tree = exports.default({ agentKey: "guide", conversationId: "chat", runId });
    effects.splice(0).forEach(effect => effect());
  }
  function elements(node = tree) {
    if (!node || typeof node !== "object") return [];
    return [node, ...[node.props?.children].flat(Infinity).flatMap(elements)];
  }
  function text(node = tree) {
    if (node === undefined || node === null || typeof node === "boolean") return "";
    if (typeof node !== "object") return String(node);
    return [node.props?.children].flat(Infinity).map(text).join(" ");
  }
  render(); render();
  return {
    render, requests, text,
    click(label) { elements().find(node => node.type === "button" && text(node) === label).props.onClick(); },
    unmount() { slots.forEach(slot => slot?.cleanup?.()); },
  };
}
const result = (stage = "selected", truncated = null) => ({
  run_id: "request", status: "completed",
  selected_references: [{ id: "fact", kind: "memory", title: "Original title", version: 3 }],
  selected_workflow: null, receipt: { references: { stage, truncated }, workflow: null },
});
test("receipt loads only on request and retains previous metadata when refresh fails", async () => {
  const view = receiptView();
  assert.equal(view.requests.length, 0);
  view.click("Context receipt"); view.render();
  assert.equal(view.requests[0].path, "/agents/guide/conversations/chat/runs/request/context");
  view.requests[0].resolve(result()); await settle(); view.render();
  assert.match(view.text(), /Original title/);
  assert.match(view.text(), /Truncation unknown/);
  view.click("Refresh"); view.requests[1].reject(new Error("offline"));
  await settle(); view.render();
  assert.match(view.text(), /Showing previous receipt. Refresh failed:.*offline/);
  assert.match(view.text(), /Original title/);
  assert.equal(view.requests.length, 2);
});
test("newer load wins and a response after unmount is ignored", async () => {
  const view = receiptView();
  view.click("Context receipt"); view.click("Context receipt");
  view.requests[1].resolve(result("model_response_recorded", false)); await settle(); view.render();
  view.requests[0].resolve(result()); await settle(); view.render();
  assert.match(view.text(), /Included in a model call with a recorded response/);
  assert.match(view.text(), /Full snapshot/);
  view.click("Refresh"); view.unmount();
  view.requests[2].resolve(result()); await settle(); view.render();
  assert.match(view.text(), /Included in a model call with a recorded response/);
});
test("tool result distinguishes unconfirmed handoff and unknown truncation", async () => {
  const view = receiptView(); view.click("Context receipt");
  view.requests[0].resolve(result("tool_result_recorded")); await settle(); view.render();
  assert.match(view.text(), /model handoff unconfirmed/);
  assert.match(view.text(), /Truncation unknown/);
  assert.doesNotMatch(view.text(), /Full snapshot/);
});
test("legacy and queued runs disclose unavailable evidence without inferring absence", async () => {
  const view = receiptView(); view.click("Context receipt");
  view.requests[0].resolve({ ...result(), status: "queued", receipt: null }); await settle(); view.render();
  assert.match(view.text(), /request queued or not prepared/);
  assert.match(view.text(), /Original title/);
  assert.doesNotMatch(view.text(), /No references or workflow selected/);
});
test("changing request discards old receipt and ignores its pending refresh", async () => {
  const view = receiptView(); view.click("Context receipt");
  view.render("other"); view.render("other");
  view.requests[0].resolve(result()); await settle(); view.render("other");
  assert.doesNotMatch(view.text(), /Original title/);
  view.click("Context receipt");
  assert.equal(view.requests[1].path, "/agents/guide/conversations/chat/runs/other/context");
});
