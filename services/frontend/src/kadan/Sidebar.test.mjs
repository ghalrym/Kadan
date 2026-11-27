import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { runInNewContext } from "node:vm";
import ts from "typescript";

const source = ts.transpileModule(
  readFileSync(new URL("./Sidebar.tsx", import.meta.url), "utf8"),
  { compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX } },
).outputText;

function renderSidebar(overrides = {}) {
  const jsx = (type, props) => ({ type, props });
  const modules = {
    react: {
      useEffect() {},
      useRef: current => ({ current }),
      useState: initial => [typeof initial === "function" ? initial() : initial, () => {}],
    },
    "react/jsx-runtime": { jsx, jsxs: jsx, Fragment: "Fragment" },
    "lucide-react": new Proxy({}, { get: (_, name) => String(name) }),
    "./Pet": { default: "Pet" },
    "./PetContext": { useAgentPet: () => null },
  };
  const exports = {};
  runInNewContext(source, {
    exports,
    require: name => { assert.ok(name in modules, name); return modules[name]; },
  });
  return exports.default({
    agent: { key: "mi", name: "Mi" },
    page: { name: "chat" },
    status: {
      tsuchi: { state: "ready", message: "Connected" },
      hikari: { state: "ready", message: "Ready", model_id: "inkling-2-mini-iq4-nl" },
    },
    loading: false,
    disabled: false,
    closeRef: { current: null },
    onClose() {},
    onRefresh() {},
    onPage() {},
    desktopHost: "http://127.0.0.1:9000",
    onServerSettings() {},
    ...overrides,
  });
}

function elements(node) {
  if (!node || typeof node !== "object") return [];
  return [node, ...[node.props?.children].flat(Infinity).flatMap(child => elements(child ?? null))];
}

test("sidebar shows single Chat navigation without multi-chat controls or Tsuchi/Hikari links", () => {
  const pages = [];
  const tree = renderSidebar({ onPage: name => pages.push(name) });
  const nodes = elements(tree);
  assert.equal(nodes.some(node => node.props?.to === "/tsuchi" || node.props?.to === "/hikari/settings"), false);
  const nav = nodes.find(node => node.type === "nav");
  const navLabels = elements(nav)
    .filter(node => node.type === "button")
    .map(node => [node.props.children].flat(Infinity).filter(item => typeof item === "string").join(""));
  assert.deepEqual(navLabels, ["Chat", "Group Chat", "Space", "Plugins"]);
  elements(nav).find(node => node.type === "button").props.onClick();
  assert.deepEqual(pages, ["chat"]);

  const footer = nodes.find(node => node.type === "footer");
  const footerChildren = [footer.props.children].flat(Infinity).filter(Boolean);
  assert.ok(footerChildren[0].props.className.includes("agent-link"));
  const serverRow = footerChildren[1];
  assert.equal(serverRow.props.className, "sidebar-server-row");
  const rowNodes = elements(serverRow);
  assert.ok(rowNodes.find(node => node.props?.className === "server-settings-trigger"));
  assert.ok(rowNodes.some(node => node.type === "small" && node.props.children === "127.0.0.1:9000"));
});
