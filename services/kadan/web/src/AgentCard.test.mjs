import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { runInNewContext } from "node:vm";
import ts from "typescript";

const source = ts.transpileModule(
  readFileSync(new URL("./AgentCard.tsx", import.meta.url), "utf8"),
  { compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX } },
).outputText;

test("the agent card exposes one ordinary Call action without a trial flag or backend choice", () => {
  const jsx = (type, props) => ({ type, props });
  const modules = {
    react: { useEffect() {}, useLayoutEffect() {}, useRef: current => ({ current }), useState: initial => [initial, () => {}] },
    "react/jsx-runtime": { jsx, jsxs: jsx },
    "lucide-react": {},
    "./Pet": { default: "Pet" },
    "./api": {},
  };
  const exports = {};
  runInNewContext(source, { exports, require: name => modules[name] });
  const calls = [];
  const tree = exports.default({
    agentName: "Guide", sheet: null, mood: "idle", presence: "Ready", onClose() {},
    callDisabled: false, callTitle: "Start a voice call", onCall: (...args) => calls.push(args), onOpenComputer() {}, activity: null,
  });
  const actions = tree.props.children.find(child => child.props?.className === "agent-card-actions");
  const buttons = [actions.props.children].flat();
  assert.equal(buttons.length, 1);
  assert.equal(buttons[0].props.disabled, false);
  assert.equal(buttons[0].props.children[1], "Call");
  buttons[0].props.onClick();
  assert.deepEqual(calls, [[]]);
});
