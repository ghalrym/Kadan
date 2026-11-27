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

function loadHelpers() {
  const exports = {};
  runInNewContext(transpile("./groupChat.ts"), { exports });
  return exports;
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

function renderGroupChat({ stateValues = [], props = {} } = {}) {
  const jsx = (type, p) => ({ type, props: p });
  const helpers = loadHelpers();
  let stateIdx = 0;
  const setters = [];
  const modules = {
    react: {
      useCallback: fn => fn,
      useEffect() {},
      useLayoutEffect() {},
      useRef: current => ({ current }),
      useState: initial => {
        const idx = stateIdx++;
        const value =
          idx in stateValues
            ? stateValues[idx]
            : typeof initial === "function"
              ? initial()
              : initial;
        const setter = () => {};
        setters.push(setter);
        return [value, setter];
      },
    },
    "react/jsx-runtime": { jsx, jsxs: jsx, Fragment: "Fragment" },
    "lucide-react": new Proxy({}, { get: (_, name) => String(name) }),
    "./api": { api: async () => ({}) },
    "./groupChat": helpers,
    "./Markdown": { default: ({ children }) => ({ type: "Markdown", props: { children } }) },
    "./Pet": { default: "Pet" },
    "./PetContext": { useAgentPet: () => null },
    "./uuid": { randomUuid: () => "00000000-0000-0000-0000-000000000099" },
  };
  const exports = {};
  runInNewContext(transpile("./GroupChat.tsx"), {
    exports,
    sessionStorage: { getItem: () => null, setItem() {}, removeItem() {} },
    require: name => {
      assert.ok(name in modules, `Unexpected import: ${name}`);
      return modules[name];
    },
  });
  return exports.default({
    agentKey: "kadan",
    agentName: "Kadan",
    status: {
      tsuchi: { state: "ready", message: "Connected" },
      hikari: { state: "ready", message: "Ready", model_id: "inkling-2-mini-iq4-nl" },
    },
    ...props,
  });
}

const SAMPLE_PARTICIPANTS = [
  {
    id: "p-kadan",
    room_id: "room-1",
    handle: "kadan",
    display_name: "Kadan",
    kind: "kadan_agent",
    agent_key: "kadan",
    description: "Built-in Kadan agent",
    enabled: true,
    revoked_at: null,
    last_seen_at: null,
    cursor_seq: 2,
    version: 1,
    created_at: "2026-11-27T12:00:00Z",
    updated_at: "2026-11-27T12:00:00Z",
    push_status: "built_in",
    push_status_detail: "Wakes automatically when tagged or handed the turn.",
    active_subscriptions: 0,
  },
  {
    id: "p-dot",
    room_id: "room-1",
    handle: "dot",
    display_name: "Dot",
    kind: "external_agent",
    agent_key: null,
    description: "Code reviewer",
    enabled: true,
    revoked_at: null,
    last_seen_at: "2026-11-27T15:59:00Z",
    cursor_seq: 2,
    version: 1,
    created_at: "2026-11-27T12:00:00Z",
    updated_at: "2026-11-27T12:00:00Z",
    push_status: "active",
    push_status_detail: "Push wake-ups are active.",
    active_subscriptions: 1,
  },
  {
    id: "p-paused",
    room_id: "room-1",
    handle: "paused-bot",
    display_name: "Paused Bot",
    kind: "external_agent",
    agent_key: null,
    description: "",
    enabled: false,
    revoked_at: null,
    last_seen_at: null,
    cursor_seq: 0,
    version: 2,
    created_at: "2026-11-27T12:00:00Z",
    updated_at: "2026-11-27T12:00:00Z",
    push_status: "paused",
    push_status_detail: "Paused. Resume to allow replies and push wake-ups.",
    active_subscriptions: 0,
  },
];

const plain = value => JSON.parse(JSON.stringify(value));

test("@mention detection, filtering, insertion, and tagged participant resolution", () => {
  const {
    findActiveMention,
    filterMentionCandidates,
    applyMention,
    resolveTaggedParticipantIds,
  } = loadHelpers();

  const m = findActiveMention("Hey @do", 7);
  assert.deepEqual(plain(m), { start: 4, end: 7, query: "do" });

  // Paused participants are excluded from autocomplete candidates.
  const candidates = filterMentionCandidates(SAMPLE_PARTICIPANTS, "do");
  assert.deepEqual(plain(candidates.map(p => p.handle)), ["dot"]);
  const allActive = filterMentionCandidates(SAMPLE_PARTICIPANTS, "");
  assert.deepEqual(plain(allActive.map(p => p.handle)), ["kadan", "dot"]);

  const applied = applyMention("Hey @do", m, "dot");
  assert.equal(applied.text, "Hey @dot ");
  assert.equal(applied.cursor, 9);

  // Combines chip selection + inline @mentions, deduplicates, and ignores paused handles.
  const tagged = resolveTaggedParticipantIds(
    "Hey @dot and @paused-bot",
    ["p-kadan", "p-dot"],
    SAMPLE_PARTICIPANTS,
  );
  assert.deepEqual(plain(tagged), ["p-kadan", "p-dot"]);
});

test("participantStatusBadge and turnCapNotice return clear plain-language labels", () => {
  const { participantStatusBadge, turnCapNotice } = loadHelpers();

  assert.deepEqual(plain(participantStatusBadge({ push_status: "built_in", last_seen_at: null })), {
    label: "Built-in agent",
    tone: "ok",
  });
  assert.deepEqual(plain(participantStatusBadge({ push_status: "active", last_seen_at: null })), {
    label: "Push wake-ups active",
    tone: "ok",
  });
  assert.deepEqual(plain(participantStatusBadge({ push_status: "paused", last_seen_at: null })), {
    label: "Paused",
    tone: "warn",
  });
  assert.deepEqual(plain(participantStatusBadge({ push_status: "error", last_seen_at: null })), {
    label: "Delivery error",
    tone: "error",
  });
  assert.deepEqual(plain(participantStatusBadge({ push_status: "revoked", last_seen_at: null })), {
    label: "Revoked",
    tone: "muted",
  });
  assert.deepEqual(plain(participantStatusBadge({ push_status: "idle", last_seen_at: null })), {
    label: "Not connected yet",
    tone: "muted",
  });

  assert.equal(
    turnCapNotice({ consecutive_agent_turns: 2, max_consecutive_agent_turns: 6 }),
    null,
  );
  const banner = turnCapNotice({ consecutive_agent_turns: 6, max_consecutive_agent_turns: 6 });
  assert.ok(banner && banner.includes("paused") && banner.includes("Send a message"));
});

test("GroupChat renders messages, handoff chips, turn-cap notice, and Participants drawer with one-time token and revoke controls", () => {
  const snapshot = {
    room: {
      id: "room-1",
      title: "Group Chat",
      consecutive_agent_turns: 6,
      max_consecutive_agent_turns: 6,
      version: 3,
      created_at: "2026-11-27T12:00:00Z",
      updated_at: "2026-11-27T12:10:00Z",
    },
    participants: SAMPLE_PARTICIPANTS,
    messages: [
      {
        id: "m-1",
        room_id: "room-1",
        seq: 1,
        sender_kind: "user",
        participant_id: null,
        sender_name: "You",
        content: "Hey @dot review this",
        handoff_to_participant_ids: ["p-dot"],
        handoff_suppressed: false,
        reply_to_message_id: null,
        client_request_id: null,
        created_at: "2026-11-27T12:01:00Z",
      },
      {
        id: "m-2",
        room_id: "room-1",
        seq: 2,
        sender_kind: "external_agent",
        participant_id: "p-dot",
        sender_name: "Dot",
        content: "Looks good, over to @kadan",
        handoff_to_participant_ids: [],
        handoff_suppressed: true,
        reply_to_message_id: null,
        client_request_id: null,
        created_at: "2026-11-27T12:02:00Z",
      },
    ],
    latest_seq: 2,
    has_more: false,
    turn_cap_reached: true,
    mcp_url: "https://kadan.local/kadan/mcp/group-chat",
  };

  // Match GroupChat.tsx useState slot order:
  // 0: snapshot, 1: messages, 2: hasOlder, 3: loading, 4: loadingOlder,
  // 5: error, 6: notice, 7: drawerOpen, 8: draft, 9: selectedTags,
  // 10: pendingOutbox, 11: sending, 12: mention, 13: mentionIndex, 14: away,
  // 15: newName, 16: newDescription, 17: managing, 18: credential
  const stateValues = [];
  stateValues[0] = snapshot;
  stateValues[1] = snapshot.messages;
  stateValues[3] = false;
  stateValues[7] = true; // drawerOpen
  stateValues[18] = {
    participantName: "Dot",
    handle: "dot",
    mcpUrl: snapshot.mcp_url,
    bearerToken: "kdn_gc_one_time_secret_token",
    rotated: false,
  };

  const tree = renderGroupChat({ stateValues });
  const all = elements(tree);
  const fullText = textOf(tree);

  // Turn-cap notice is displayed clearly.
  assert.ok(fullText.includes("Agent replies are paused"));

  // Messages, sender badges, and handoff/suppressed indicators render.
  const articles = all.filter(node => node.type === "article");
  assert.equal(articles.length, 2);
  assert.ok(textOf(articles[0]).includes("Tagged @dot"));
  assert.ok(textOf(articles[1]).includes("External"));
  assert.ok(textOf(articles[1]).includes("Handoff paused"));

  // Clickable participant chips render for active agents (Kadan + Dot, not Paused Bot).
  const chips = all.filter(node => node.type === "button" && node.props?.className?.includes("gc-chip"));
  assert.equal(chips.length, 2);

  // Participants & Connections drawer shows ChatGPT plugin download, MCP URL, one-time token card, status badges, and Revoke button.
  const drawer = all.find(node => node.type === "aside" && node.props?.className === "group-chat-drawer");
  assert.ok(drawer);
  const drawerText = textOf(drawer);
  assert.ok(drawerText.includes("Connect ChatGPT"));
  assert.ok(drawerText.includes("Download ChatGPT plugin"));
  const downloadLink = all.find(
    node => node.type === "a" && node.props?.href === "/kadan/api/group-chat/chatgpt-plugin.zip",
  );
  assert.ok(downloadLink);
  assert.ok(drawerText.includes("https://kadan.local/kadan/mcp/group-chat"));
  assert.ok(drawerText.includes("kdn_gc_one_time_secret_token"));
  assert.ok(drawerText.includes("will not be shown again"));
  assert.ok(drawerText.includes("Push wake-ups active"));
  assert.ok(drawerText.includes("Create agent token"));
  assert.ok(drawerText.includes("Rotate token"));
  assert.ok(drawerText.includes("Revoke"));
});
