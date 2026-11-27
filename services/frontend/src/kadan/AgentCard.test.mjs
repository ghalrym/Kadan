import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { runInNewContext } from "node:vm";
import ts from "typescript";

const source = ts.transpileModule(
  readFileSync(new URL("./AgentCard.tsx", import.meta.url), "utf8"),
  { compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX } },
).outputText;

test("the agent card shows Call as disabled Coming soon with Persona directly under it and no Activity section", () => {
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
  const personas = [];
  const tree = exports.default({
    agentName: "Guide", sheet: null, mood: "idle", presence: "Ready", onClose() {},
    personaDisabled: false, onEditPersona: (...args) => personas.push(args),
  });
  const actions = tree.props.children.find(child => child.props?.className === "agent-card-actions");
  const items = [actions.props.children].flat();
  assert.equal(items.length, 2);
  assert.equal(items[0].props.title, "Coming soon");
  assert.equal(items[0].props["data-tooltip"], "Coming soon");
  assert.equal(items[0].props.children.props.disabled, true);
  assert.equal(items[0].props.children.props.children[1], "Call");
  assert.equal(items[1].props.disabled, false);
  assert.equal(items[1].props.children[1], "Persona");
  items[1].props.onClick();
  assert.deepEqual(personas, [[]]);
  assert.equal(tree.props.children.length, 2);
});
