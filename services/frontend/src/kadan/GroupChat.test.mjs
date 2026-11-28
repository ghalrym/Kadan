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

function renderGroupChat({ stateValues = [], props = {}, desktop } = {}) {
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
    "./desktop": desktop ?? {
      isDesktopApp: () => false,
      saveDesktopDownload: async () => ({
        saved: false,
        cancelled: false,
        path: null,
        filename: null,
        supported: false,
      }),
    },
    "./groupChat": helpers,
    "./Markdown": { default: ({ children }) => ({ type: "Markdown", props: { children } }) },
    "./Pet": { default: "Pet" },
    "./pet": { agentHue: () => 200 },
    "./PetAvatar": { ParticipantAvatar: "ParticipantAvatar", UserAvatar: "UserAvatar" },
    "./PetContext": { useAgentPet: () => null, usePetRefJob: () => null },
    "./PetEditor": { default: "PetEditor" },
    "./profile": { USER_SEED: "kadan:user" },
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
    profile: { display_name: "Andrew", pet: null, version: 1 },
    onEditProfile() {},
    onEditAgentPet() {},
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
    label: "Replies when tagged",
    tone: "ok",
  });
  assert.deepEqual(plain(participantStatusBadge({ push_status: "paused", last_seen_at: null })), {
    label: "Paused",
    tone: "warn",
  });
  assert.deepEqual(plain(participantStatusBadge({ push_status: "error", last_seen_at: null })), {
    label: "Can’t reach agent",
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

  // Discord-style rows: each sender block opens with an avatar and name; the user shows their profile name.
  assert.ok(articles.every(node => node.props.className.includes("starts-block")));
  assert.ok(textOf(articles[0]).startsWith("Andrew"));
  assert.ok(elements(articles[0]).some(node => node.type === "UserAvatar"));
  assert.equal(elements(articles[1]).find(node => node.type === "ParticipantAvatar").props.participant.handle, "dot");
  assert.ok(all.some(node => node.props?.role === "separator"));

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
  assert.ok(drawerText.includes("Replies when tagged"));
  assert.ok(drawerText.includes("Create connection key"));
  assert.ok(drawerText.includes("New connection key"));
  assert.ok(drawerText.includes("Revoke"));

  // Every participant who can still change has an Edit pet button, Kadan's agent included.
  const petButtons = elements(drawer).filter(node => node.type === "button" && textOf(node) === "Edit pet");
  assert.equal(petButtons.length, 3);
});

test("groupChatRows groups one sender's nearby messages and labels each day", () => {
  const { groupChatRows, dayLabel } = loadHelpers();
  const message = (id, sender_kind, participant_id, created_at) => ({ id, sender_kind, participant_id, sender_name: id, created_at });
  const now = new Date(2026, 10, 27, 18, 0);
  const at = (day, hour, minute) => new Date(2026, 10, day, hour, minute).toISOString();
  const rows = groupChatRows([
    message("a", "user", null, at(26, 9, 0)),
    message("b", "user", null, at(27, 9, 0)),
    message("c", "user", null, at(27, 9, 5)),
    message("d", "external_agent", "dot", at(27, 9, 6)),
    message("e", "external_agent", "dot", at(27, 9, 20)),
    message("f", "external_agent", "sol", at(27, 9, 21)),
  ], now);
  assert.deepEqual(plain(rows.map(row => [row.message.id, row.startsBlock, row.dayLabel])), [
    ["a", true, "Yesterday"],
    ["b", true, "Today"],
    ["c", false, null],
    ["d", true, null],
    ["e", true, null],
    ["f", true, null],
  ]);
  assert.equal(dayLabel(new Date(2025, 0, 2), now).includes("2025"), true);
});

test("Download ChatGPT plugin uses desktop Save dialog in Kadan Desktop and native link in browsers", async () => {
  const snapshot = {
    room: {
      id: "room-1",
      title: "Group Chat",
      consecutive_agent_turns: 0,
      max_consecutive_agent_turns: 6,
      version: 1,
      created_at: "2026-11-27T12:00:00Z",
      updated_at: "2026-11-27T12:00:00Z",
    },
    participants: SAMPLE_PARTICIPANTS,
    messages: [],
    latest_seq: 0,
    has_more: false,
    turn_cap_reached: false,
    mcp_url: "https://kadan.local/kadan/mcp/group-chat",
  };
  const stateValues = [];
  stateValues[0] = snapshot;
  stateValues[1] = [];
  stateValues[3] = false;
  stateValues[7] = true; // drawerOpen

  const calls = [];
  const desktopTree = renderGroupChat({
    stateValues,
    desktop: {
      isDesktopApp: () => true,
      saveDesktopDownload: async (url, filename) => {
        calls.push({ url, filename });
        return {
          saved: true,
          cancelled: false,
          path: "/tmp/kadan-group-chat-plugin.zip",
          filename: "kadan-group-chat-plugin.zip",
          supported: true,
        };
      },
    },
  });
  const desktopLink = elements(desktopTree).find(
    node => node.type === "a" && node.props?.href === "/kadan/api/group-chat/chatgpt-plugin.zip",
  );
  assert.ok(desktopLink);
  let prevented = false;
  desktopLink.props.onClick({
    preventDefault() {
      prevented = true;
    },
  });
  await new Promise(resolve => setTimeout(resolve, 0));
  assert.equal(prevented, true);
  assert.deepEqual(calls, [
    {
      url: "/kadan/api/group-chat/chatgpt-plugin.zip",
      filename: "kadan-group-chat-plugin.zip",
    },
  ]);

  const browserCalls = [];
  const browserTree = renderGroupChat({
    stateValues,
    desktop: {
      isDesktopApp: () => false,
      saveDesktopDownload: async (url, filename) => {
        browserCalls.push({ url, filename });
        return { saved: false, cancelled: false, path: null, filename: null, supported: false };
      },
    },
  });
  const browserLink = elements(browserTree).find(
    node => node.type === "a" && node.props?.href === "/kadan/api/group-chat/chatgpt-plugin.zip",
  );
  let browserPrevented = false;
  browserLink.props.onClick({
    preventDefault() {
      browserPrevented = true;
    },
  });
  assert.equal(browserPrevented, false);
  assert.equal(browserCalls.length, 0);
});

test("isAwaitingReply follows the latest hand-off until the agent answers or the window passes", () => {
  const { isAwaitingReply, messageSnippet, groupChatRows } = loadHelpers();
  const now = Date.parse("2026-11-27T12:10:00Z");
  const msg = (participant_id, handoff, created_at, handoff_suppressed = false) => ({
    participant_id, handoff_to_participant_ids: handoff, handoff_suppressed, created_at,
  });
  assert.equal(isAwaitingReply([msg(null, ["k"], "2026-11-27T12:08:00Z")], "k", now), true);
  assert.equal(isAwaitingReply([msg(null, ["k"], "2026-11-27T12:08:00Z"), msg("k", [], "2026-11-27T12:09:00Z")], "k", now), false);
  assert.equal(isAwaitingReply([msg(null, ["k"], "2026-11-27T12:00:00Z")], "k", now), false);
  assert.equal(isAwaitingReply([msg("d", ["k"], "2026-11-27T12:09:00Z", true)], "k", now), false);
  assert.equal(isAwaitingReply([msg(null, ["d"], "2026-11-27T12:09:00Z")], "k", now), false);

  assert.equal(messageSnippet("one\n\ntwo   three"), "one two three");
  assert.equal(messageSnippet("x".repeat(100), 10), `${"x".repeat(9)}…`);

  const rows = groupChatRows([
    { id: "a", sender_kind: "user", participant_id: null, sender_name: "You", created_at: "2026-11-27T12:00:00Z" },
    { id: "b", sender_kind: "user", participant_id: null, sender_name: "You", created_at: "2026-11-27T12:01:00Z", reply_to_message_id: "z" },
  ], new Date("2026-11-27T13:00:00Z"));
  assert.equal(rows[1].startsBlock, true);
});

test("replying shows the quoted message, a reply bar, highlighted mentions and the replying indicator", () => {
  const now = new Date().toISOString();
  const original = {
    id: "m-1", room_id: "room-1", seq: 1, sender_kind: "external_agent", participant_id: "p-dot", sender_name: "Dot",
    content: "Here is the plan", handoff_to_participant_ids: [], handoff_suppressed: false,
    reply_to_message_id: null, client_request_id: null, created_at: now,
  };
  const reply = {
    ...original, id: "m-2", seq: 2, sender_kind: "user", participant_id: null, sender_name: "You",
    content: "Thanks @dot, over to @kadan", handoff_to_participant_ids: ["p-kadan"], reply_to_message_id: "m-1",
  };
  const snapshot = {
    room: { id: "room-1", title: "Group Chat", consecutive_agent_turns: 0, max_consecutive_agent_turns: 6, version: 1, created_at: now, updated_at: now },
    participants: SAMPLE_PARTICIPANTS, messages: [original, reply], latest_seq: 2, has_more: false,
    turn_cap_reached: false, mcp_url: "https://kadan.local/kadan/mcp/group-chat",
  };
  const stateValues = [];
  stateValues[0] = snapshot;
  stateValues[1] = snapshot.messages;
  stateValues[3] = false;
  stateValues[27] = original; // replyTo
  const tree = renderGroupChat({ stateValues });
  const all = elements(tree);

  const quote = all.find(node => node.props?.className === "gc-reply-ref");
  assert.ok(textOf(quote).includes("Dot"));
  assert.ok(textOf(quote).includes("Here is the plan"));

  const bar = all.find(node => node.props?.className === "gc-replying-to");
  assert.ok(textOf(bar).includes("Replying to Dot"));

  const markdown = all.filter(node => node.props?.mentions);
  assert.equal(markdown.length, 2);
  assert.deepEqual([...markdown[1].props.mentions].sort(), ["dot", "kadan", "paused-bot"]);

  const replying = all.find(node => node.props?.className === "gc-replying");
  assert.ok(textOf(replying).includes("Kadan is replying"));

  assert.ok(all.some(node => node.props?.["aria-label"] === "Reply to Dot"));
});

test("sortDirectChats puts the most recent private chats first, then the rest by name", () => {
  const { sortDirectChats } = loadHelpers();
  const chat = (name, last_message_at = null) => ({ participant: { display_name: name }, last_message_at });
  assert.deepEqual(
    plain(sortDirectChats([chat("Zed"), chat("Ann"), chat("Old", "2026-11-26T10:00:00Z"), chat("New", "2026-11-27T10:00:00Z")]))
      .map(item => item.participant.display_name),
    ["New", "Old", "Ann", "Zed"],
  );
});

test("a private chat shows only its agent, always asks it to reply, and hides room controls", () => {
  const now = new Date().toISOString();
  const snapshot = {
    room: { id: "dm", title: "Dot", kind: "direct", direct_participant_id: "p-dot", consecutive_agent_turns: 0, max_consecutive_agent_turns: 6, version: 1, created_at: now, updated_at: now },
    participants: [SAMPLE_PARTICIPANTS[1]], messages: [], latest_seq: 0, has_more: false,
    turn_cap_reached: false, mcp_url: "https://kadan.local/kadan/mcp/group-chat",
  };
  const stateValues = [];
  stateValues[0] = snapshot;
  stateValues[3] = false;
  const tree = renderGroupChat({ stateValues, props: { directParticipantId: "p-dot" } });
  const all = elements(tree);
  const text = textOf(tree);

  assert.equal(tree.props["aria-label"], "Chat with Dot");
  assert.ok(text.includes("Your private chat with Dot"));
  assert.ok(text.includes("Dot will be asked to reply"));
  assert.equal(all.some(node => node.props?.className === "gc-tag-bar"), false);
  assert.equal(text.includes("Participants & Connections"), false);
  assert.equal(all.find(node => node.type === "textarea").props.placeholder, "Message Dot");
  assert.ok(all.some(node => node.type === "button" && textOf(node) === "Edit pet"));
});
