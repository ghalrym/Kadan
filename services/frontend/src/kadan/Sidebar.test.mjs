import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { runInNewContext } from "node:vm";
import ts from "typescript";

const workSource = ts.transpileModule(
  readFileSync(new URL("./work.ts", import.meta.url), "utf8"),
  { compilerOptions: { module: ts.ModuleKind.CommonJS } },
).outputText;

const workExports = {};
runInNewContext(workSource, {
  exports: workExports,
  require: name => {
    if (name === "./api") return { api: async () => null };
    throw new Error(`Unexpected import ${name}`);
  },
});

const source = ts.transpileModule(
  readFileSync(new URL("./Sidebar.tsx", import.meta.url), "utf8"),
  { compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX } },
).outputText;

function renderSidebar(overrides = {}, stateOverrides = []) {
  const jsx = (type, props) => ({ type, props });
  let stateIndex = 0;
  const modules = {
    react: {
      useEffect() {},
      useRef: current => ({ current }),
      useState: initial => {
        const idx = stateIndex++;
        const value = idx in stateOverrides
          ? stateOverrides[idx]
          : typeof initial === "function" ? initial() : initial;
        return [value, () => {}];
      },
    },
    "react/jsx-runtime": { jsx, jsxs: jsx, Fragment: "Fragment" },
    "lucide-react": new Proxy({}, { get: (_, name) => String(name) }),
    "./Pet": { default: "Pet" },
    "./PetContext": { useAgentPet: () => null },
    "./work": workExports,
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

function textOf(node) {
  if (typeof node === "string" || typeof node === "number") return String(node);
  if (!node || typeof node !== "object") return "";
  return [node.props?.children].flat(Infinity).map(textOf).join("");
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
  assert.deepEqual(navLabels, ["Chat", "Space", "Plugins"]);
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

test("sidebar Work section groups threads into Default, Working, Snoozed, and collapsible searchable Settled buckets", () => {
  const opened = [];
  const unsettled = [];
  const woken = [];
  const threads = [
    {
      id: "t-default",
      title: "Fix auth",
      bucket: "default",
      run_status: "waiting",
      snoozed_until: null,
      pending_requests: [{ id: "p1", status: "pending" }],
    },
    {
      id: "t-working",
      title: "Refactor API",
      bucket: "working",
      run_status: "running",
      snoozed_until: null,
      pending_requests: [],
    },
    {
      id: "t-snoozed",
      title: "Upgrade deps",
      bucket: "snoozed",
      run_status: "completed",
      snoozed_until: new Date(Date.now() + 3600_000).toISOString(),
      pending_requests: [],
    },
    {
      id: "t-settled",
      title: "Ship v1 docs",
      bucket: "settled",
      run_status: "completed",
      snoozed_until: null,
      pending_requests: [],
    },
  ];

  // Expand Settled bucket via stateOverrides [settledOpen=true, settledQuery=""]
  const tree = renderSidebar(
    {
      workThreads: threads,
      onOpenWorkThread: id => opened.push(id),
      onUnsettleWorkThread: id => unsettled.push(id),
      onWakeWorkThread: id => woken.push(id),
    },
    [true, ""],
  );
  const nodes = elements(tree);
  const buckets = nodes.filter(node => node.props?.["data-bucket"]);
  assert.deepEqual(
    buckets.map(b => b.props["data-bucket"]),
    ["default", "working", "snoozed", "settled"],
  );

  const defaultText = textOf(buckets[0]);
  assert.ok(defaultText.includes("Fix auth"));
  assert.ok(defaultText.includes("Needs input"));

  assert.ok(textOf(buckets[1]).includes("Refactor API"));
  assert.ok(textOf(buckets[2]).includes("Upgrade deps"));
  assert.ok(textOf(buckets[3]).includes("Ship v1 docs"));

  const wakeBtn = elements(buckets[2]).find(n => n.props?.["aria-label"] === "Wake Upgrade deps");
  wakeBtn.props.onClick({ stopPropagation() {} });
  assert.deepEqual(woken, ["t-snoozed"]);

  const unsettleBtn = elements(buckets[3]).find(n => n.props?.["aria-label"] === "Unsettle Ship v1 docs");
  unsettleBtn.props.onClick({ stopPropagation() {} });
  assert.deepEqual(unsettled, ["t-settled"]);

  const openWorking = elements(buckets[1]).find(n => n.props?.className === "chat-row-open");
  openWorking.props.onClick();
  assert.deepEqual(opened, ["t-working"]);
});
