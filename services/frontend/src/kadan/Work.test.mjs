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
  // State order in WorkThreadView: detail, harnesses, loading, busy, error, draft, snoozeOpen, customSnooze, confirmDelete
  const states = [detail, harnesses, false, false, "", "Also add a test", false, "", false];
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
    "./Markdown": { default: ({ children }) => ({ type: "md", props: { children } }) },
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

  // Activity rows include user message, assistant message, and file change.
  const activityRows = nodes.filter(n => typeof n.type === "function" && n.type.name === "ActivityRow");
  assert.equal(activityRows.length, 3);
  const fileRowTree = activityRows[2].type(activityRows[2].props);
  assert.ok(textOf(fileRowTree).includes("services/api.py"));

  // Pending approval card is rendered inline with Approve and Decline buttons.
  const pendingSection = nodes.find(n => typeof n.type === "function" && n.type.name === "PendingCard");
  assert.ok(pendingSection);
  assert.equal(pendingSection.props.request.id, "req-1");
  const pendingTree = pendingSection.type(pendingSection.props);
  assert.ok(textOf(pendingTree).includes("Approve"));
  assert.ok(textOf(pendingTree).includes("Decline"));
});


test("harnessAuthLabel renders plain-language local states for Inkling (Local)", () => {
  assert.equal(
    work.harnessAuthLabel({ status: "ready", login_method: "local" }, true),
    "Ready (Local)",
  );
  assert.equal(
    work.harnessAuthLabel({ status: "waiting", login_method: "local" }, true),
    "Downloading…",
  );
  assert.equal(
    work.harnessAuthLabel({ status: "unavailable", login_method: "local" }, true),
    "Model download needed",
  );
  assert.equal(
    work.harnessAuthLabel({ status: "unavailable", login_method: "local" }, false),
    "Offline",
  );
  // External harnesses preserve sign-in labels.
  assert.equal(
    work.harnessAuthLabel({ status: "sign_in_required", login_method: "device_code" }, true),
    "Sign in needed",
  );
});

test("WorkSetup Harnesses tab shows Inkling (Local) without external sign-in controls", () => {
  const jsx = (type, props) => ({ type, props });
  const harnesses = [
    {
      id: "inkling",
      display_name: "Inkling (Local)",
      description: "Built-in local coding assistant powered by Inkling 2 Mini on this computer.",
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
        message: "Download the standard Inkling 2 Mini model in Kadan Server → Hikari → Settings.",
      },
    },
  ];
  // State order in WorkSetup: tab, projects, harnesses, loading, busy, error, notice, ...
  const states = ["harnesses", [], harnesses, false, false, "", ""];
  let idx = 0;

  const WorkSetup = loadTs("./WorkSetup.tsx", {
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
    "./work": work,
  }).default;

  const tree = WorkSetup({ initialTab: "harnesses", onThreadCreated() {}, onChanged() {} });
  const fullText = textOf(tree);

  assert.ok(fullText.includes("Inkling (Local)"));
  assert.ok(fullText.includes("Model download needed"));
  assert.ok(fullText.includes("Hikari → Settings"));
  assert.ok(fullText.includes("Check status"));
  assert.ok(!fullText.includes("Sign in to Inkling"));
  assert.ok(!fullText.includes("Sign out"));
});

test("WorkThreadView enables screenshot attachment control for Inkling (Local) threads", () => {
  const jsx = (type, props) => ({ type, props });
  const detail = {
    thread: {
      id: "thr-ink",
      project_id: "proj-1",
      title: "Fix button alignment",
      harness: "inkling",
      model: "inkling-2-mini-iq4-nl",
      run_status: "idle",
      bucket: "default",
      snoozed_until: null,
      pending_requests: [],
      version: 1,
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
      display_name: "Inkling (Local)",
      installed: true,
      supports_steer: true,
      supports_images: true,
      default_model: "inkling-2-mini-iq4-nl",
      models: [{ id: "inkling-2-mini-iq4-nl", label: "Inkling 2 Mini" }],
      auth: { status: "ready", login_method: "local", message: "Ready on this computer." },
    },
  ];
  const states = [detail, harnesses, false, false, "", "", false, "", false, []];
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
    "./Markdown": { default: ({ children }) => ({ type: "md", props: { children } }) },
    "./work": work,
  }).default;

  const tree = WorkThreadView({ threadId: "thr-ink", onChanged() {}, onDeleted() {} });
  const nodes = elements(tree);
  const fullText = textOf(tree);

  assert.ok(fullText.includes("Inkling (Local)"));
  assert.ok(fullText.includes("Image"));
  const activityRow = nodes.find(n => typeof n.type === "function" && n.type.name === "ActivityRow");
  assert.ok(activityRow);
  const rowTree = activityRow.type(activityRow.props);
  assert.ok(textOf(rowTree).includes("1 image"));
});


test("WorkThreadView queues image-bearing follow-ups instead of steering while an Inkling run is active", async () => {
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
    activity: [],
  };
  const harnesses = [
    {
      id: "inkling",
      display_name: "Inkling (Local)",
      installed: true,
      supports_steer: true,
      supports_images: true,
      default_model: "inkling-2-mini-iq4-nl",
      models: [{ id: "inkling-2-mini-iq4-nl", label: "Inkling 2 Mini" }],
      auth: { status: "ready", login_method: "local", message: "Ready on this computer." },
    },
  ];
  const states = [
    detail,
    harnesses,
    false,
    false,
    "",
    "Look at this screenshot next",
    false,
    "",
    false,
    [{ name: "bug.png", dataUrl: sampleDataUrl }],
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
    "./Markdown": { default: ({ children }) => ({ type: "md", props: { children } }) },
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
