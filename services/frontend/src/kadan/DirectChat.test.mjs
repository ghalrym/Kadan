import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { runInNewContext } from "node:vm";
import ts from "typescript";

function transpile(file) {
  return ts.transpileModule(
    readFileSync(new URL(file, import.meta.url), "utf8"),
    { compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX } },
  ).outputText;
}

function elements(node) {
  if (!node || typeof node !== "object") return [];
  return [node, ...[node.props?.children].flat(Infinity).flatMap((child) => elements(child ?? null))];
}

function textOf(node) {
  if (typeof node === "string" || typeof node === "number") return String(node);
  if (!node || typeof node !== "object") return "";
  return [node.props?.children].flat(Infinity).map(textOf).join("");
}

const DOT = {
  id: "p-dot",
  handle: "dot",
  display_name: "Dot",
  kind: "external_agent",
  agent_key: null,
  description: "Coding teammate",
  pet: null,
  enabled: true,
  revoked_at: null,
  last_seen_at: "2026-11-28T09:50:00Z",
  version: 1,
  push_status: "connected",
};

function renderDirectChat({ stateValues = [], props = {} } = {}) {
  const jsx = (type, p) => ({ type, props: p });
  const helpers = {};
  runInNewContext(transpile("./groupChat.ts"), { exports: helpers });
  let stateIdx = 0;
  const modules = {
    react: {
      Fragment: "Fragment",
      useCallback: (fn) => fn,
      useEffect() {},
      useLayoutEffect() {},
      useMemo: (fn) => fn(),
      useRef: (current) => ({ current }),
      useState: (initial) => {
        const idx = stateIdx++;
        const value =
          idx in stateValues
            ? stateValues[idx]
            : typeof initial === "function"
              ? initial()
              : initial;
        return [value, () => {}];
      },
    },
    "react/jsx-runtime": { jsx, jsxs: jsx, Fragment: "Fragment" },
    "lucide-react": new Proxy({}, { get: (_, name) => String(name) }),
    "./AgentCard": { default: "AgentCard" },
    "./api": { api: async () => ({}) },
    "./ChatExport": { default: "ChatExport" },
    "./Composer": { readiness: (s) => ({ storage: s?.tsuchi?.state === "ready", chat: s?.tsuchi?.state === "ready" && s?.hikari?.state === "ready" }) },
    "./groupChat": helpers,
    "./Markdown": { default: ({ children }) => ({ type: "Markdown", props: { children } }) },
    "./Pet": { default: "Pet", useMood: (base) => [base, () => {}] },
    "./PetContext": { usePetRefJob: () => null, usePetRefSheet: () => ({ url: "/default.png", version: 2, pixelated: true }) },
    "./PetEditor": { default: "PetEditor" },
    "./uuid": { randomUuid: () => "00000000-0000-0000-0000-000000000099" },
    "./voiceInput": { microphoneUnavailable: () => null, VoiceInput: class {} },
  };
  const exports = {};
  runInNewContext(transpile("./DirectChat.tsx"), {
    exports,
    sessionStorage: { getItem: () => null, setItem() {}, removeItem() {} },
    require: (name) => {
      assert.ok(name in modules, `Unexpected import: ${name}`);
      return modules[name];
    },
  });
  return exports.default({
    participantId: "p-dot",
    agentKey: "kadan",
    status: {
      tsuchi: { state: "ready", message: "" },
      hikari: { state: "ready", message: "", model_id: "inkling-2-mini-iq4-nl" },
    },
    ...props,
  });
}

test("empty DirectChat renders Kadan's home stage with grounded Pet, Edit pet, and Message Dot composer", () => {
  const now = new Date().toISOString();
  const snapshot = {
    room: { id: "dm", title: "Dot", kind: "direct", direct_participant_id: "p-dot", consecutive_agent_turns: 0, max_consecutive_agent_turns: 6, version: 1, created_at: now, updated_at: now },
    participants: [DOT],
    messages: [],
    latest_seq: 0,
    has_more: false,
    turn_cap_reached: false,
    mcp_url: "https://kadan.local/kadan/mcp/group-chat",
  };
  // DirectChat useState slots: 0=snapshot, 1=messages, 2=hasOlder, 3=loading
  const stateValues = [];
  stateValues[0] = snapshot;
  stateValues[1] = [];
  stateValues[3] = false;

  const tree = renderDirectChat({ stateValues });
  const all = elements(tree);
  const text = textOf(tree);

  assert.equal(tree.props.className.trim(), "chat");
  assert.equal(tree.props["aria-label"], "Chat with Dot");
  assert.ok(all.some((node) => node.props?.className === "home-stage"));
  assert.ok(all.some((node) => node.type === "Pet" && node.props?.size === 192 && node.props?.grounded === true));
  assert.ok(text.includes("Only you and Dot can see this chat."));
  assert.ok(all.some((node) => node.type === "button" && textOf(node) === "Edit pet"));
  assert.equal(all.some((node) => node.props?.className === "gc-tag-bar"), false);
  assert.equal(text.includes("Participants & Connections"), false);
  assert.equal(all.find((node) => node.type === "textarea").props.placeholder, "Message Dot");
});

test("DirectChat renders lookout pet, Kadan message bubbles, timestamp breaks, reply refs, receipt, typing indicator, and AgentCard", () => {
  const now = new Date().toISOString();
  const m1 = {
    id: "m-1", room_id: "dm", seq: 1, sender_kind: "external_agent", participant_id: "p-dot", sender_name: "Dot",
    content: "Ready when you are!", handoff_to_participant_ids: [], handoff_suppressed: false,
    reply_to_message_id: null, client_request_id: null, created_at: now,
  };
  const m2 = {
    id: "m-2", room_id: "dm", seq: 2, sender_kind: "user", participant_id: null, sender_name: "You",
    content: "Let's ship it", handoff_to_participant_ids: ["p-dot"], handoff_suppressed: false,
    reply_to_message_id: "m-1", client_request_id: "req-2", created_at: now,
  };
  const snapshot = {
    room: { id: "dm", title: "Dot", kind: "direct", direct_participant_id: "p-dot", consecutive_agent_turns: 0, max_consecutive_agent_turns: 6, version: 1, created_at: now, updated_at: now },
    participants: [DOT],
    messages: [m1, m2],
    latest_seq: 2,
    has_more: false,
    turn_cap_reached: false,
    mcp_url: "https://kadan.local/kadan/mcp/group-chat",
  };
  // 0=snapshot, 1=messages, 3=loading, 11=card
  const stateValues = [];
  stateValues[0] = snapshot;
  stateValues[1] = [m1, m2];
  stateValues[3] = false;
  stateValues[11] = true;

  const tree = renderDirectChat({ stateValues });
  const all = elements(tree);
  const text = textOf(tree);

  assert.ok(all.some((node) => node.props?.className?.includes("lookout working")));
  assert.ok(all.some((node) => node.type === "Pet" && node.props?.size === 96 && node.props?.look === true));
  assert.ok(all.some((node) => node.props?.className === "time-break"));

  const articles = all.filter((node) => node.type === "article");
  assert.equal(articles.length, 2);
  assert.ok(articles[0].props.className.includes("message assistant"));
  assert.ok(articles[1].props.className.includes("message user"));

  const replyRef = all.find((node) => node.props?.className === "message-reply-ref");
  assert.ok(replyRef);
  assert.ok(textOf(replyRef).includes("Dot"));
  assert.ok(textOf(replyRef).includes("Ready when you are!"));

  const receipt = all.find((node) => node.props?.className === "receipt");
  assert.equal(textOf(receipt), "Delivered");

  assert.ok(all.some((node) => node.props?.className === "typing"));
  assert.ok(text.includes("Dot is replying…"));

  const card = all.find((node) => node.type === "AgentCard");
  assert.ok(card);
  assert.equal(card.props.agentName, "Dot");
  assert.equal(card.props.showCall, false);
  assert.equal(typeof card.props.onEditPet, "function");
  assert.equal(card.props.onDelete, undefined);
});
