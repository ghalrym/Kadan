import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { runInNewContext } from "node:vm";
import ts from "typescript";

function loadTs(filename, mocks) {
  const source = ts.transpileModule(
    readFileSync(new URL(filename, import.meta.url), "utf8"),
    { compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX } },
  ).outputText;
  const exports = {};
  runInNewContext(source, {
    exports,
    require: name => {
      assert.ok(name in mocks, `Unexpected require(${name}) in ${filename}`);
      return mocks[name];
    },
  });
  return exports;
}

const work = loadTs("./work.ts", {
  "./api": { api: async () => null },
});

function elements(node) {
  if (!node || typeof node !== "object") return [];
  return [node, ...[node.props?.children].flat(Infinity).flatMap(child => elements(child ?? null))];
}

function textOf(node) {
  if (typeof node === "string" || typeof node === "number") return String(node);
  if (!node || typeof node !== "object") return "";
  return [node.props?.children].flat(Infinity).map(textOf).join("");
}

test("groupThreadsByBucket filters settled threads by search query and preserves all four buckets", () => {
  const threads = [
    { id: "1", title: "Alpha", harness: "codex", bucket: "default" },
    { id: "2", title: "Beta", harness: "claude", bucket: "working" },
    { id: "3", title: "Gamma", harness: "grok", bucket: "snoozed" },
    { id: "4", title: "Delta migration", harness: "codex", bucket: "settled" },
    { id: "5", title: "Epsilon UI", harness: "antigravity", bucket: "settled" },
  ];
  const grouped = work.groupThreadsByBucket(threads, "delta");
  assert.deepEqual(Array.from(grouped.default, t => t.id), ["1"]);
  assert.deepEqual(Array.from(grouped.working, t => t.id), ["2"]);
  assert.deepEqual(Array.from(grouped.snoozed, t => t.id), ["3"]);
  assert.deepEqual(Array.from(grouped.settled, t => t.id), ["4"]);
});

test("parseWorkHash extracts valid #work/<id> thread identifiers", () => {
  assert.equal(work.parseWorkHash("#work/12345678-1234-1234-1234-1234567890ab"), "12345678-1234-1234-1234-1234567890ab");
  assert.equal(work.parseWorkHash("#other"), null);
  assert.equal(work.parseWorkHash(""), null);
});

test("WorkThreadView renders streamed activity, inline approval card, and active steer/queue/stop controls", () => {
  const jsx = (type, props) => ({ type, props });
  const detail = {
    thread: {
      id: "thr-1",
      project_id: "proj-1",
      title: "Fix API timeout",
      harness: "codex",
      model: "gpt-5.3-codex",
      run_status: "waiting",
      bucket: "default",
      snoozed_until: null,
      pending_requests: [
        {
          id: "req-1",
          kind: "approval",
          title: "Allow pytest?",
          prompt: "Run pytest in the project folder?",
          options: [],
          tool_name: "bash",
          arguments: { command: "pytest" },
          status: "pending",
          response: null,
          created_at: new Date().toISOString(),
          resolved_at: null,
        },
      ],
      version: 2,
    },
    project: { id: "proj-1", title: "Kadan", root_path: "/repo", version: 1 },
    runs: [],
    activity: [
      { id: "a1", kind: "user_message", text: "Fix timeout", data: {} },
      { id: "a2", kind: "assistant_message", text: "Checking request timeout settings.", data: {} },
      { id: "a3", kind: "file_change", text: "", data: { file_path: "services/api.py", change_type: "modified" } },
    ],
  };
  const harnesses = [
    {
      id: "codex",
      display_name: "Codex",
      installed: true,
      supports_steer: true,
      default_model: "gpt-5.3-codex",
      models: [{ id: "gpt-5.3-codex", label: "GPT-5.3 Codex" }],
      auth: { status: "ready" },
    },
  ];
  // State order: detail, harnesses, projects, selectedProjectId, selectedModelKey, loading, busy, error, draft, snoozeOpen, customSnooze, confirmDelete, attachedImages, folderDialogOpen
  const states = [detail, harnesses, [], "", "", false, false, "", "Also add a test", false, "", false, [], false];
  let idx = 0;

  const WorkThreadView = loadTs("./WorkThreadView.tsx", {
    react: {
      useCallback: fn => fn,
      useEffect() {},
      useRef: current => ({ current }),
      useState: initial => {
        const i = idx++;
        const val = i < states.length ? states[i] : (typeof initial === "function" ? initial() : initial);
        return [val, () => {}];
      },
    },
    "react/jsx-runtime": { jsx, jsxs: jsx, Fragment: "Fragment" },
    "lucide-react": new Proxy({}, { get: (_, name) => String(name) }),
    "./ConfirmDialog": { default: "ConfirmDialog" },
    "./FolderPickerDialog": { default: "FolderPickerDialog" },
    "./Markdown": { default: ({ children }) => ({ type: "md", props: { children } }) },
    "./ModelPicker": { default: function ModelPicker() { return null; } },
    "./ProjectPicker": { default: function ProjectPicker() { return null; } },
    "./desktop": { isDesktopApp: () => false, pickDesktopFolder: async () => ({ path: null, cancelled: false, supported: false }) },
    "./work": work,
  }).default;

  const tree = WorkThreadView({ threadId: "thr-1", onChanged() {}, onDeleted() {} });
  const nodes = elements(tree);
  const fullText = textOf(tree);

  assert.ok(fullText.includes("Fix API timeout"));
  assert.ok(fullText.includes("Kadan"));
  assert.ok(fullText.includes("Stop"));
  assert.ok(fullText.includes("Steer"));
  assert.ok(fullText.includes("Queue"));

  const activityRows = nodes.filter(n => typeof n.type === "function" && n.type.name === "ActivityRow");
  assert.equal(activityRows.length, 3);
  const fileRowTree = activityRows[2].type(activityRows[2].props);
  assert.ok(textOf(fileRowTree).includes("services/api.py"));

  const pendingSection = nodes.find(n => typeof n.type === "function" && n.type.name === "PendingCard");
  assert.ok(pendingSection);
  assert.equal(pendingSection.props.request.id, "req-1");
  const pendingTree = pendingSection.type(pendingSection.props);
  assert.ok(textOf(pendingTree).includes("Approve"));
  assert.ok(textOf(pendingTree).includes("Decline"));
});

test("WorkThreadView opens directly to chat with project and model pickers when threadId is null", async () => {
  const jsx = (type, props) => ({ type, props });
  const createdCalls = [];
  const projects = [{ id: "proj-1", title: "Aidoru", root_path: "/home/andrew/Aidoru", version: 1 }];
  const harnesses = [
    {
      id: "inkling",
      display_name: "Inkling",
      installed: true,
      supports_steer: true,
      supports_images: true,
      default_model: "inkling-2-mini-abliterated-iq4-nl",
      models: [{ id: "inkling-2-mini-abliterated-iq4-nl", label: "Inkling 2 Mini" }],
      auth: { status: "ready", login_method: "local", message: "Ready on Kadan Server." },
    },
  ];
  const states = [
    null,
    harnesses,
    projects,
    "proj-1",
    "inkling::inkling-2-mini-abliterated-iq4-nl",
    false,
    false,
    "",
    "Fix the login redirect",
    false,
    "",
    false,
    [],
    false,
  ];
  let idx = 0;

  const WorkThreadView = loadTs("./WorkThreadView.tsx", {
    react: {
      useCallback: fn => fn,
      useEffect() {},
      useRef: current => ({ current }),
      useState: initial => {
        const i = idx++;
        const val = i < states.length ? states[i] : (typeof initial === "function" ? initial() : initial);
        return [val, () => {}];
      },
    },
    "react/jsx-runtime": { jsx, jsxs: jsx, Fragment: "Fragment" },
    "lucide-react": new Proxy({}, { get: (_, name) => String(name) }),
    "./ConfirmDialog": { default: "ConfirmDialog" },
    "./FolderPickerDialog": { default: "FolderPickerDialog" },
    "./Markdown": { default: ({ children }) => ({ type: "md", props: { children } }) },
    "./ModelPicker": { default: function ModelPicker() { return null; } },
    "./ProjectPicker": { default: function ProjectPicker() { return null; } },
    "./desktop": { isDesktopApp: () => false, pickDesktopFolder: async () => ({ path: null, cancelled: false, supported: false }) },
    "./work": {
      ...work,
      createWorkThread: async body => {
        createdCalls.push(body);
        return { id: "thr-new" };
      },
    },
  }).default;

  const createdIds = [];
  const tree = WorkThreadView({
    threadId: null,
    onThreadCreated: id => createdIds.push(id),
    onChanged() {},
    onDeleted() {},
  });
  const nodes = elements(tree);
  const fullText = textOf(tree);

  assert.ok(fullText.includes("What should we build in Aidoru?"));
  assert.ok(!fullText.includes("Task title"));
  const projectPicker = nodes.find(n => typeof n.type === "function" && n.type.name === "ProjectPicker");
  assert.equal(projectPicker.props.value, "proj-1");
  const modelPicker = nodes.find(n => typeof n.type === "function" && n.type.name === "ModelPicker");
  assert.equal(modelPicker.props.harnessId, "inkling");
  assert.equal(modelPicker.props.modelId, "inkling-2-mini-abliterated-iq4-nl");
  const textarea = nodes.find(n => n.type === "textarea");
  assert.ok(textarea);
  textarea.props.onKeyDown({ key: "Enter", shiftKey: false, preventDefault() {} });
  await new Promise(resolve => setTimeout(resolve, 0));

  assert.equal(createdCalls.length, 1);
  assert.equal(createdCalls[0].project_id, "proj-1");
  assert.equal(createdCalls[0].harness, "inkling");
  assert.equal(createdCalls[0].prompt, "Fix the login redirect");
  assert.equal("title" in createdCalls[0], false);
  assert.deepEqual(createdIds, ["thr-new"]);
});

test("harnessAuthLabel renders concise plain-language states", () => {
  assert.equal(work.harnessAuthLabel({ status: "ready", login_method: "local" }, true), "Ready");
  assert.equal(work.harnessAuthLabel({ status: "waiting", login_method: "local" }, true), "Downloading…");
  assert.equal(work.harnessAuthLabel({ status: "unavailable", login_method: "local" }, true), "Download needed");
  assert.equal(work.harnessAuthLabel({ status: "unavailable", login_method: "local" }, false), "Offline");
  assert.equal(work.harnessAuthLabel({ status: "sign_in_required", login_method: "device_code" }, true), "Sign in");
});

test("SettingsPage renders compact Coding assistants list without external sign-in controls for Inkling", () => {
  const jsx = (type, props) => ({ type, props });
  const harnesses = [
    {
      id: "inkling",
      display_name: "Inkling",
      description: "Kadan Server model.",
      installed: true,
      supports_steer: true,
      supports_images: true,
      default_model: "inkling-2-mini-iq4-nl",
      models: [{ id: "inkling-2-mini-iq4-nl", label: "Inkling 2 Mini" }],
      auth: {
        status: "unavailable",
        login_method: "local",
        authorize_url: null,
        user_code: null,
        needs_code: false,
        account_label: null,
        message: "Download Inkling 2 Mini in Kadan Server → Hikari → Settings.",
      },
    },
  ];
  // State order: harnesses, projects, loading, busy, error, expandedSignIn, signInInputs, folderDialogOpen
  const states = [harnesses, [], false, false, "", null, {}, false];
  let idx = 0;

  const SettingsPage = loadTs("./SettingsPage.tsx", {
    react: {
      useCallback: fn => fn,
      useEffect() {},
      useState: initial => {
        const i = idx++;
        const val = i < states.length ? states[i] : (typeof initial === "function" ? initial() : initial);
        return [val, () => {}];
      },
    },
    "react/jsx-runtime": { jsx, jsxs: jsx, Fragment: "Fragment" },
    "lucide-react": new Proxy({}, { get: (_, name) => String(name) }),
    "./FolderPickerDialog": { default: "FolderPickerDialog" },
    "./ProviderIcon": { default: "ProviderIcon" },
    "./desktop": { pickDesktopFolder: async () => ({ path: null, cancelled: false, supported: false }) },
    "./work": work,
  }).default;

  const tree = SettingsPage({
    desktopMode: true,
    desktopHost: "http://192.168.0.19:8000",
    status: null,
    onOpenServerSettings() {},
    onChanged() {},
  });
  const fullText = textOf(tree);

  assert.ok(fullText.includes("Coding assistants"));
  assert.ok(fullText.includes("Inkling"));
  assert.ok(fullText.includes("Download needed"));
  assert.ok(fullText.includes("Hikari → Settings"));
  assert.ok(fullText.includes("Check"));
  assert.ok(!fullText.includes("Sign in"));
  assert.ok(!fullText.includes("Sign out"));
});

test("WorkThreadView enables screenshot attachment control and queues image follow-ups for Inkling threads", async () => {
  const jsx = (type, props) => ({ type, props });
  const sentCalls = [];
  const sampleDataUrl =
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+X2dIAAAAASUVORK5CYII=";
  const detail = {
    thread: {
      id: "thr-ink-active",
      project_id: "proj-1",
      title: "Active Inkling run",
      harness: "inkling",
      model: "inkling-2-mini-iq4-nl",
      run_status: "running",
      bucket: "working",
      snoozed_until: null,
      pending_requests: [],
      version: 2,
    },
    project: { id: "proj-1", title: "Kadan", root_path: "/repo", version: 1 },
    runs: [],
    activity: [
      { id: "u1", kind: "user_message", text: "Check screenshot", data: { image_count: 1 } },
    ],
  };
  const harnesses = [
    {
      id: "inkling",
      display_name: "Inkling",
      installed: true,
      supports_steer: true,
      supports_images: true,
      default_model: "inkling-2-mini-iq4-nl",
      models: [{ id: "inkling-2-mini-iq4-nl", label: "Inkling 2 Mini" }],
      auth: { status: "ready", login_method: "local", message: "Ready on Kadan Server." },
    },
  ];
  const states = [
    detail,
    harnesses,
    [],
    "",
    "",
    false,
    false,
    "",
    "Look at this screenshot next",
    false,
    "",
    false,
    [{ name: "bug.png", dataUrl: sampleDataUrl }],
    false,
  ];
  let idx = 0;

  const WorkThreadView = loadTs("./WorkThreadView.tsx", {
    react: {
      useCallback: fn => fn,
      useEffect() {},
      useRef: current => ({ current }),
      useState: initial => {
        const i = idx++;
        const val = i < states.length ? states[i] : (typeof initial === "function" ? initial() : initial);
        return [val, () => {}];
      },
    },
    "react/jsx-runtime": { jsx, jsxs: jsx, Fragment: "Fragment" },
    "lucide-react": new Proxy({}, { get: (_, name) => String(name) }),
    "./ConfirmDialog": { default: "ConfirmDialog" },
    "./FolderPickerDialog": { default: "FolderPickerDialog" },
    "./Markdown": { default: ({ children }) => ({ type: "md", props: { children } }) },
    "./ModelPicker": { default: function ModelPicker() { return null; } },
    "./ProjectPicker": { default: function ProjectPicker() { return null; } },
    "./desktop": { isDesktopApp: () => false, pickDesktopFolder: async () => ({ path: null, cancelled: false, supported: false }) },
    "./work": {
      ...work,
      getWorkThreadDetail: async () => detail,
      sendWorkMessage: async (id, body) => {
        sentCalls.push({ id, body });
        return { id: "run-2" };
      },
    },
  }).default;

  const tree = WorkThreadView({ threadId: "thr-ink-active", onChanged() {}, onDeleted() {} });
  const nodes = elements(tree);
  const fullText = textOf(tree);

  assert.ok(nodes.some(n => n.props?.["aria-label"] === "Attach screenshots"));
  const steerBtn = nodes.find(n => n.type === "button" && textOf(n).trim() === "Steer");
  assert.ok(steerBtn);
  assert.equal(steerBtn.props.disabled, true);

  const queueBtn = nodes.find(n => n.type === "button" && textOf(n).includes("Queue"));
  assert.ok(queueBtn);
  assert.equal(queueBtn.props.disabled, false);

  const textarea = nodes.find(n => n.type === "textarea");
  assert.ok(textarea);
  textarea.props.onKeyDown({ key: "Enter", shiftKey: false, preventDefault() {} });
  await new Promise(resolve => setTimeout(resolve, 0));

  assert.equal(sentCalls.length, 1);
  assert.equal(sentCalls[0].id, "thr-ink-active");
  assert.equal(sentCalls[0].body.mode, "queue");
  assert.deepEqual(sentCalls[0].body.images, [sampleDataUrl]);
});

test("sidebar helpers format snooze time, timestamps, badges and search matches", () => {
  const now = Date.parse("2026-11-27T12:00:00Z");
  assert.equal(work.snoozeRemaining(new Date(now + 45 * 60000).toISOString(), now), "45m");
  assert.equal(work.snoozeRemaining(new Date(now + 130 * 60000).toISOString(), now), "2h");
  assert.equal(work.snoozeRemaining(new Date(now + 3 * 86400000).toISOString(), now), "3d");
  assert.equal(work.snoozeRemaining(new Date(now - 1000).toISOString(), now), null);
  assert.equal(work.relativeTime(new Date(now - 20000).toISOString(), now), "now");
  assert.equal(work.relativeTime(new Date(now - 3 * 60000).toISOString(), now), "3m");
  assert.equal(work.relativeTime(new Date(now - 5 * 3600000).toISOString(), now), "5h");
  assert.equal(work.relativeTime(new Date(now - 2 * 86400000).toISOString(), now), "2d");
  assert.equal(work.projectInitials("ghalrym/Kadan"), "GK");
  assert.equal(work.projectInitials("Aidoru"), "AI");
  const thread = { title: "Fix login", harness: "codex" };
  assert.equal(work.matchesThreadQuery(thread, "LOGIN"), true);
  assert.equal(work.matchesThreadQuery(thread, "codex"), true);
  assert.equal(work.matchesThreadQuery(thread, "kadan", "ghalrym/Kadan"), true);
  assert.equal(work.matchesThreadQuery(thread, "deploy", "Aidoru"), false);
});

function loadModelPicker(states) {
  const jsx = (type, props) => ({ type, props });
  let idx = 0;
  return loadTs("./ModelPicker.tsx", {
    react: {
      useCallback: fn => fn,
      useLayoutEffect() {},
      useMemo: fn => fn(),
      useRef: current => ({ current }),
      useState: initial => {
        const i = idx++;
        const val = i < states.length ? states[i] : (typeof initial === "function" ? initial() : initial);
        return [val, () => {}];
      },
    },
    "react/jsx-runtime": { jsx, jsxs: jsx, Fragment: "Fragment" },
    "lucide-react": new Proxy({}, { get: (_, name) => String(name) }),
    "./ProviderIcon": { default: "ProviderIcon" },
    "./popover": { usePopover: () => "up" },
    "./work": work,
  });
}

const pickerHarnesses = [
  {
    id: "inkling",
    display_name: "Inkling",
    installed: true,
    default_model: "inkling-2-mini-iq4-nl",
    models: [{ id: "inkling-2-mini-iq4-nl", label: "Inkling 2 Mini", options: { reasoning_effort: ["low", "medium", "high"] } }],
    auth: { status: "ready", login_method: "local", message: "" },
  },
  {
    id: "codex",
    display_name: "Codex",
    installed: true,
    default_model: "gpt-5.3-codex",
    models: [
      { id: "gpt-5.3-codex", label: "GPT-5.3 Codex", options: { effort: ["low", "medium", "high", "xhigh"] } },
      { id: "o3", label: "o3", options: {} },
    ],
    auth: { status: "ready", login_method: "device_code", message: "" },
  },
  {
    id: "grok",
    display_name: "Grok",
    installed: true,
    default_model: "grok-4",
    models: [{ id: "grok-4", label: "Grok 4", options: {} }],
    auth: { status: "sign_in_required", login_method: "api_key", message: "Paste your xAI access key." },
  },
];

test("ModelPicker shows the provider rail, models, reasoning levels and keeps effort on selection", () => {
  // State order: open, rail, query, highlight, favorites, indicatorTop
  const ModelPicker = loadModelPicker([true, "codex", "", 0, new Set(), 40]).default;
  const changes = [];
  const tree = ModelPicker({
    harnesses: pickerHarnesses,
    harnessId: "codex",
    modelId: "gpt-5.3-codex",
    options: { effort: "high" },
    onChange: (...args) => changes.push(args),
    onOpenSettings() {},
  });
  const nodes = elements(tree);
  const text = textOf(tree);

  const trigger = nodes.find(n => n.props?.className?.startsWith?.("model-picker-trigger"));
  assert.ok(textOf(trigger).includes("GPT-5.3 Codex"));
  assert.ok(textOf(trigger).includes("High"));

  const railLabels = nodes.filter(n => n.props?.className?.startsWith?.("model-picker-rail-btn")).map(n => n.props["aria-label"]);
  assert.deepEqual(railLabels, ["Favorites", "Inkling", "Codex", "Grok · Sign in"]);

  const items = nodes.filter(n => n.props?.role === "option");
  assert.deepEqual(items.map(n => textOf(n).replace(/Ctrl\+\d/, "")), ["GPT-5.3 Codex", "o3"]);
  assert.ok(text.includes("Ctrl+1"));

  assert.ok(text.includes("Reasoning"));
  const levels = nodes.filter(n => n.props?.className === "model-picker-effort-item");
  assert.deepEqual(levels.map(n => textOf(n)), ["Low", "MediumDefault", "High", "Extra High"]);
  assert.equal(levels[2].props["aria-pressed"], true);

  // Picking a level selects the highlighted model with that effort.
  levels[3].props.onClick();
  assert.equal(JSON.stringify(changes.at(-1)), JSON.stringify(["codex", "gpt-5.3-codex", { effort: "xhigh" }]));

  // Picking a model without reasoning options clears effort.
  items[1].props.onClick();
  assert.equal(JSON.stringify(changes.at(-1)), JSON.stringify(["codex", "o3", {}]));
});

test("ModelPicker explains unavailable assistants and blocks choosing their models", () => {
  const ModelPicker = loadModelPicker([true, "grok", "", 0, new Set(), null]).default;
  const changes = [];
  const tree = ModelPicker({
    harnesses: pickerHarnesses,
    harnessId: "codex",
    modelId: "gpt-5.3-codex",
    options: {},
    onChange: (...args) => changes.push(args),
    onOpenSettings() {},
  });
  const nodes = elements(tree);
  assert.ok(textOf(tree).includes("Grok isn't ready yet."));
  assert.ok(textOf(tree).includes("Paste your xAI access key."));
  const item = nodes.find(n => n.props?.role === "option");
  assert.equal(item.props["aria-disabled"], true);
  item.props.onClick();
  assert.equal(changes.length, 0);
});

test("ModelPicker search spans every assistant", () => {
  const ModelPicker = loadModelPicker([true, "codex", "grok", 0, new Set(), null]).default;
  const tree = ModelPicker({
    harnesses: pickerHarnesses,
    harnessId: "codex",
    modelId: "gpt-5.3-codex",
    options: {},
    onChange() {},
  });
  const items = elements(tree).filter(n => n.props?.role === "option");
  assert.equal(items.length, 1);
  assert.ok(textOf(items[0]).includes("Grok 4"));
});
