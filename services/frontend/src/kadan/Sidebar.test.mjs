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
    "./PetAvatar": { ParticipantAvatar: "ParticipantAvatar", UserAvatar: "UserAvatar" },
    "./PetContext": { useAgentPet: () => null },
    "./groupChat": { isParticipantActive: p => p.enabled && p.revoked_at === null },
    "./ProviderIcon": { default: "ProviderIcon" },
    "./work": workExports,
  };
  const exports = {};
  runInNewContext(source, {
    exports,
    require: name => { assert.ok(name in modules, name); return modules[name]; },
  });
  return exports.default({
    agent: { key: "mi", name: "Mi" },
    profile: { display_name: "Andrew", pet: null, version: 1 },
    onProfile() {},
    groupChat: null,
    groupChatUnread: false,
    chatUnread: false,
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

test("sidebar opens chat from the agent row at the top of navigation, without multi-chat controls or Tsuchi/Hikari links", () => {
  const pages = [];
  const tree = renderSidebar({ onPage: name => pages.push(name) });
  const nodes = elements(tree);
  assert.equal(nodes.some(node => node.props?.to === "/tsuchi" || node.props?.to === "/hikari/settings"), false);
  const nav = nodes.find(node => node.type === "nav");
  const navLabels = elements(nav)
    .filter(node => node.type === "button")
    .map(textOf);
  assert.deepEqual(navLabels, ["Mi", "Group Chat", "Space", "Plugins"]);
  const agentRow = elements(nav).find(node => node.type === "button");
  assert.ok(agentRow.props.className.includes("agent-link"));
  assert.ok(agentRow.props.className.includes("selected"));
  assert.equal(agentRow.props["aria-current"], "page");
  agentRow.props.onClick();
  assert.deepEqual(pages, ["chat"]);

  const footer = nodes.find(node => node.type === "footer");
  const footerChildren = [footer.props.children].flat(Infinity).filter(Boolean);
  assert.equal(footerChildren.some(child => child.props?.className?.includes?.("agent-link")), false);
  assert.equal(footerChildren[0].props.className, "profile-link");
  assert.equal(textOf(footerChildren[0]), "AndrewYou");
  const serverRow = footerChildren[1];
  assert.equal(serverRow.props.className, "sidebar-server-row");
  const rowNodes = elements(serverRow);
  assert.ok(rowNodes.find(node => typeof node.props?.className === "string" && node.props.className.includes("server-settings-trigger")));
  assert.ok(rowNodes.some(node => node.type === "small" && node.props.children === "127.0.0.1:9000"));
});

const workThreads = [
  {
    id: "t-default",
    project_id: "p1",
    title: "Fix auth",
    harness: "claude",
    bucket: "default",
    run_status: "waiting",
    snoozed_until: null,
    updated_at: new Date().toISOString(),
    pending_requests: [{ id: "p1", status: "pending" }],
  },
  {
    id: "t-working",
    project_id: "p1",
    title: "Refactor API",
    harness: "inkling",
    bucket: "working",
    run_status: "running",
    snoozed_until: null,
    updated_at: new Date().toISOString(),
    pending_requests: [],
  },
  {
    id: "t-snoozed",
    project_id: "p2",
    title: "Upgrade deps",
    harness: "codex",
    bucket: "snoozed",
    run_status: "completed",
    snoozed_until: new Date(Date.now() + 2 * 3600_000).toISOString(),
    updated_at: new Date().toISOString(),
    pending_requests: [],
  },
  {
    id: "t-settled",
    project_id: "p2",
    title: "Ship v1 docs",
    harness: "codex",
    bucket: "settled",
    run_status: "completed",
    snoozed_until: null,
    updated_at: new Date().toISOString(),
    pending_requests: [],
  },
];
const workProjects = [
  { id: "p1", title: "Kadan", root_path: "/repo/kadan" },
  { id: "p2", title: "ghalrym/Docs", root_path: "/repo/docs" },
];

test("sidebar Work list shows active threads with T3-style Working, Snoozed and Settled shelves", () => {
  const opened = [];
  const unsettled = [];
  const woken = [];
  const settled = [];
  // State order: query, shelves, settledLimit, projects
  const tree = renderSidebar(
    {
      workThreads,
      onOpenWorkThread: id => opened.push(id),
      onSettleWorkThread: id => settled.push(id),
      onUnsettleWorkThread: id => unsettled.push(id),
      onWakeWorkThread: id => woken.push(id),
    },
    ["", { working: true, snoozed: true, settled: true }, 25, workProjects],
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
  assert.ok(defaultText.includes("Kadan"));
  assert.ok(defaultText.includes("KA"));

  const shelfToggles = nodes.filter(n => n.props?.className === "work-shelf-toggle");
  assert.deepEqual(shelfToggles.map(n => textOf(n)), ["Working1", "Snoozed1", "Settled1"]);

  assert.ok(textOf(buckets[1]).includes("Refactor API"));
  assert.ok(textOf(buckets[2]).includes("Upgrade deps"));
  assert.ok(textOf(buckets[2]).includes("Snoozed, 2h left"));
  assert.ok(textOf(buckets[3]).includes("Ship v1 docs"));

  const settleBtn = elements(buckets[0]).find(n => n.props?.["aria-label"] === "Settle Fix auth");
  settleBtn.props.onClick({ stopPropagation() {} });
  assert.deepEqual(settled, ["t-default"]);

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

test("sidebar collapses Snoozed and Settled by default, hides empty shelves, and searches every bucket", () => {
  const collapsed = renderSidebar(
    { workThreads: workThreads.filter(t => t.bucket !== "working") },
    ["", { working: true, snoozed: false, settled: false }, 25, workProjects],
  );
  const shelves = elements(collapsed).filter(n => n.props?.className === "work-shelf");
  assert.deepEqual(shelves.map(s => s.props["data-bucket"]), ["snoozed", "settled"]);
  assert.ok(!textOf(collapsed).includes("Upgrade deps"));
  assert.ok(!textOf(collapsed).includes("Ship v1 docs"));

  const searched = renderSidebar(
    { workThreads },
    ["docs", { working: true, snoozed: false, settled: false }, 25, workProjects],
  );
  const text = textOf(searched);
  assert.ok(text.includes("Ship v1 docs"));
  assert.ok(text.includes("Upgrade deps"), "project title matches reveal collapsed shelves");
  assert.ok(!text.includes("Fix auth"));
  assert.ok(text.includes("No threads found"));
});

test("Group Chat row stacks member pets, counts the rest, flags unread messages, and the profile row opens the profile", () => {
  const pages = [];
  const profiles = [];
  const member = (handle, extra = {}) => ({
    handle, kind: "external_agent", agent_key: null, pet: null, enabled: true, revoked_at: null, ...extra,
  });
  const tree = renderSidebar({
    onPage: name => pages.push(name),
    onProfile: () => profiles.push(true),
    groupChatUnread: true,
    chatUnread: true,
    groupChat: {
      latestSeq: 9,
      participants: [
        member("mi", { kind: "kadan_agent", agent_key: "mi" }),
        member("dot"),
        member("sol"),
        member("gone", { revoked_at: "2026-11-27T12:00:00Z", enabled: false }),
      ],
    },
  });
  const nodes = elements(tree);
  const row = nodes.find(node => node.type === "button" && node.props?.className?.includes("group-chat-link"));
  const rowNodes = elements(row);
  assert.equal(rowNodes.filter(node => node.type === "UserAvatar").length, 1);
  assert.deepEqual(rowNodes.filter(node => node.type === "ParticipantAvatar").map(node => node.props.participant.handle), ["mi", "dot"]);
  assert.ok(textOf(row).includes("+1"));
  assert.ok(rowNodes.some(node => node.props?.className === "unread-dot"));
  row.props.onClick();
  assert.deepEqual(pages, ["group-chat"]);

  const agentRow = nodes.find(node => node.type === "button" && node.props?.className?.includes("agent-link"));
  assert.ok(elements(agentRow).some(node => node.props?.className === "unread-dot"));

  nodes.find(node => node.props?.className === "profile-link").props.onClick();
  assert.equal(profiles.length, 1);
});
