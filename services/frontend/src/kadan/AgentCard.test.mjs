import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { runInNewContext } from "node:vm";
import ts from "typescript";

const source = ts.transpileModule(
  readFileSync(new URL("./AgentCard.tsx", import.meta.url), "utf8"),
  { compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX } },
).outputText;

test("the agent card exposes Call, Rename, Persona, Edit pet, a divider, then Delete agent and no Activity section", () => {
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
  const personas = [];
  const pets = [];
  const renames = [];
  const deletes = [];
  const tree = exports.default({
    agentName: "Guide", sheet: null, mood: "idle", presence: "Ready", onClose() {},
    callDisabled: false, callTitle: "Start a voice call", onCall: (...args) => calls.push(args),
    personaDisabled: false, onEditPersona: (...args) => personas.push(args),
    onEditPet: (...args) => pets.push(args),
    agentDisabled: false, onRename: (...args) => renames.push(args), onDelete: (...args) => deletes.push(args),
  });
  const actions = tree.props.children.find(child => child.props?.className === "agent-card-actions");
  const children = [actions.props.children].flat();
  assert.deepEqual(children.map(child => child.type), ["button", "button", "button", "button", "hr", "button"]);
  const buttons = children.filter(child => child.type === "button");
  assert.deepEqual(buttons.map(button => button.props.children[1]), ["Call", "Rename", "Persona", "Edit pet", "Delete agent"]);
  assert.ok(buttons.every(button => button.props.disabled !== true));
  assert.equal(buttons[4].props.className, "danger-button");
  buttons.forEach(button => button.props.onClick());
  assert.deepEqual(calls, [[]]);
  assert.deepEqual(renames, [[]]);
  assert.deepEqual(personas, [[]]);
  assert.deepEqual(pets, [[]]);
  assert.deepEqual(deletes, [[]]);
  assert.equal(tree.props.children.length, 2);
});
