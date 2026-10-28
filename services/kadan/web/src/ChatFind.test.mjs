import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { runInNewContext } from "node:vm";
import ts from "typescript";
import * as agentRuns from "./agentRuns.ts";
import * as outbox from "./outbox.ts";

const source = ts.transpileModule(
  readFileSync(new URL("./Chat.tsx", import.meta.url), "utf8"),
  { compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX } },
).outputText;
const message = (id, content, role = "user") => ({
  id, content, role, created_at: "2026-10-23T10:00:00Z", resource_ids: [], attachment_ids: [], sources: [],
});
const conversation = (messages, id = "conversation") => ({ id, version: 1, messages });

// Exercise Chat handlers and effects with in-memory hooks and service/DOM boundaries.
function chat(chain, storageValues = {}) {
  const slots = [], effects = [], layouts = [], scrolls = [], focuses = [], timers = [];
  let viewportNode, scrollTop = 1500;
  const viewport = {
    scrollHeight: 2000, clientHeight: 500,
    get scrollTop() { return scrollTop; },
    set scrollTop(value) {
      scrolls.push({ node: viewportNode, top: value });
      scrollTop = Math.min(value, this.scrollHeight - this.clientHeight);
    },
  };
  let resolveSnapshot;
  const storage = new Map(Object.entries(storageValues));
  let cursor = 0, tree, dirty = false;
  const effect = queue => (callback, dependencies) => {
    const index = cursor++;
    if (!slots[index] || !dependencies || dependencies.some((value, i) => value !== slots[index][i])) {
      slots[index] = dependencies;
      queue.push(callback);
    }
  };
  const react = {
    Fragment: "Fragment",
    useState(initial) {
      const index = cursor++;
      if (!(index in slots)) slots[index] = typeof initial === "function" ? initial() : initial;
      return [slots[index], value => {
        const next = typeof value === "function" ? value(slots[index]) : value;
        if (next !== slots[index]) { slots[index] = next; dirty = true; }
      }];
    },
    useRef(initial) { const index = cursor++; return slots[index] ??= { current: initial }; },
    useCallback: callback => callback,
    useEffect: effect(effects),
    useLayoutEffect: effect(layouts),
  };
  const jsx = (type, props) => ({ type, props });
  const reactToMood = () => {};
  const modules = {
    react,
    "react/jsx-runtime": { jsx, jsxs: jsx },
    "lucide-react": {},
    "./Markdown": { default: "Markdown" },
    "./Activity": { default: "Activity" },
    "./Composer": { default: "Composer", readiness: () => ({ chat: true, storage: true }) },
    "./Pet": { default: "Pet", useMood: () => ["idle", reactToMood] },
    "./AgentCard": { default: "AgentCard" },
    "./ComputerViewer": { default: "ComputerViewer" },
    "./VoiceCallControl": { useVoiceCall: () => ({}) },
    "./presence": { activeLabel: () => "idle" },
    "./PetContext": { useAgentPet: () => null },
    "./api": { agentPath: key => `/agents/${key}`, api: () => new Promise(resolve => { resolveSnapshot = resolve; }) },
    "./agentRuns": agentRuns,
    "./uuid": { randomUuid: () => "outgoing" },
    "./outbox": outbox,
  };
  const exports = {};
  runInNewContext(source, {
    exports, require: name => { assert.ok(name in modules, name); return modules[name]; },
    sessionStorage: {
      getItem: key => storage.get(key) ?? null,
      setItem: (key, value) => storage.set(key, value),
      removeItem: key => storage.delete(key),
    },
    setTimeout: callback => timers.push(callback), clearTimeout() {}, setInterval: () => 0, clearInterval() {},
  });
  const props = {
    agentKey: "guide", agentName: "Guide", chain, personas: [], status: null, entrance: "idle",
    onChange() {}, onBusy() {}, onEditPet() {}, blocked: false, refreshStatus: async () => {},
  };
  function all(predicate, node = tree) {
    if (!node || typeof node !== "object") return [];
    return [...(predicate(node) ? [node] : []),
      ...[node.props?.children].flat(Infinity).flatMap(child => all(predicate, child ?? null))];
  }
  function render() {
    let passes = 0;
    do {
      dirty = false;
      cursor = 0;
      tree = exports.default(props);
      for (const node of all(node => node.props?.ref)) {
        const element = node.props.className === "messages" ? viewport : {
          scrollIntoView(options) {
            scrolls.push({ node, options });
          },
          focus: () => focuses.push(node),
        };
        if (element === viewport) {
          viewportNode = node;
          element.focus = () => focuses.push(node);
        }
        if (typeof node.props.ref === "function") node.props.ref(element);
        else node.props.ref.current = element;
      }
      layouts.splice(0).forEach(callback => callback());
      effects.splice(0).forEach(callback => callback());
      assert.ok(++passes < 10, "render settled");
    } while (dirty);
  }
  const named = name => all(node => node.props?.["aria-label"] === name)[0];
  function keyDown(target, event) {
    function path(node) {
      if (!node || typeof node !== "object") return null;
      if (node === target) return [node];
      for (const child of [node.props?.children].flat(Infinity)) {
        const found = path(child);
        if (found) return [...found, node];
      }
      return null;
    }
    const lineage = path(tree);
    assert.ok(lineage, "keydown target belongs to the current tree");
    lineage.forEach(node => node.props?.onKeyDown?.(event));
  }
  const field = () => all(node => node.type === "input")[0];
  const status = () => all(node => node.props?.className === "find-count")[0].props.children;
  function search(query) {
    if (!field()) { named("Find in conversation").props.onClick(); render(); }
    field().props.onChange({ target: { value: query } });
    render();
  }
  render();
  return {
    all, render, named, keyDown, field, status, search, scrolls, focuses, storage, props, viewport,
    scroll(distance) {
      viewport.scrollTop = viewport.scrollHeight - viewport.clientHeight - distance;
      named("Conversation messages").props.onScroll({ currentTarget: viewport });
      render();
    },
    async updateRuns(runs) {
      if (timers.length) timers.shift()();
      resolveSnapshot({ runs, images: [], conversation: props.chain });
      await new Promise(resolve => setImmediate(resolve));
      render();
    },
  };
}

const selected = view => view.all(node => node.type === "article" && node.props.className.includes("find-match"));

test("find reveals a saved message outside the newest 200 and scrolls to it", () => {
  const view = chat(conversation([message("old", "An earlier needle"), ...Array.from({ length: 201 }, (_, i) => message(`m${i}`, "Recent"))]));
  assert.equal(view.all(node => node.type === "article").length, 200);
  view.scrolls.length = 0;
  view.search("NEEDLE");
  assert.equal(view.status(), "1 of 1 messages");
  assert.equal(view.all(node => node.type === "article").length, 202);
  assert.equal(selected(view)[0].props["aria-label"], "Matching message 1 of 1");
  assert.equal(view.scrolls.at(-1).node.type, "article");
  assert.equal(view.scrolls.at(-1).options.block, "center");
  assert.ok(view.scrolls.every(scroll => scroll.node.type === "article"));
});

test("find counts messages once, treats punctuation literally and excludes retired tool records", () => {
  const view = chat(conversation([
    message("first", "Use [a] twice: [a]"), message("second", "Another [A]"),
    message("plain", "a"), message("tool", "[a]", "tool"),
  ]));
  view.search("[a]");
  assert.equal(view.status(), "1 of 2 messages");
  view.all(node => node.type === "form")[0].props.onSubmit({ preventDefault() {} });
  view.render();
  assert.equal(view.status(), "2 of 2 messages");
  view.all(node => node.type === "form")[0].props.onSubmit({ preventDefault() {} });
  view.render();
  assert.equal(view.status(), "1 of 2 messages");
  view.field().props.onKeyDown({ key: "Enter", shiftKey: true, preventDefault() {} });
  view.render();
  assert.equal(view.status(), "2 of 2 messages");
  view.named("Previous matching message").props.onClick();
  view.render();
  assert.equal(view.status(), "1 of 2 messages");
});

test("empty and missing queries give honest state without moving the conversation", () => {
  const view = chat(conversation([message("one", "Saved text")]));
  view.scrolls.length = 0;
  view.search("   ");
  assert.equal(view.field().props.autoFocus, true);
  assert.equal(view.status(), "Saved text only");
  assert.equal(view.named("Next matching message").props.disabled, true);
  assert.equal(selected(view).length, 0);
  view.search("missing");
  assert.equal(view.status(), "No matching messages");
  assert.equal(view.named("Previous matching message").props.disabled, true);
  assert.equal(view.scrolls.length, 0);
});

test("Escape closes find, restores its trigger focus and preserves the draft and recovery", () => {
  const pending = JSON.stringify({ id: "pending", content: "Unsent needle", generate: false, persona: "" });
  const view = chat(conversation([message("saved", "Saved message")]), {
    "kadan.draft.guide": "Draft needle", "kadan.draft.guide.pending": pending,
  });
  view.search("needle");
  assert.equal(view.status(), "No matching messages");
  view.keyDown(view.field(), { key: "Escape", preventDefault() {} });
  view.render();
  assert.equal(view.field(), undefined);
  assert.equal(view.focuses.length, 1);
  assert.equal(view.storage.get("kadan.draft.guide"), "Draft needle");
  assert.equal(view.storage.get("kadan.draft.guide.pending"), pending);
  assert.equal(view.all(node => node.type === "Composer")[0].props.value, "Draft needle");
  view.search("");
  assert.equal(view.status(), "Saved text only");
});

test("Escape from Close find clears search and restores the trigger focus", () => {
  const view = chat(conversation([message("saved", "needle")]));
  view.search("needle");
  let prevented = 0;
  view.keyDown(view.named("Close find"), { key: "Escape", preventDefault() { prevented++; } });
  view.render();
  assert.equal(prevented, 1);
  assert.equal(view.field(), undefined);
  assert.equal(selected(view).length, 0);
  assert.equal(view.focuses.length, 1);
  assert.equal(view.named("Find in conversation").props["aria-expanded"], false);
});

test("updates keep the selected message and do not pull a search to the newest reply", () => {
  const view = chat(conversation([message("first", "needle"), message("second", "needle")]));
  view.search("needle");
  view.all(node => node.type === "form")[0].props.onSubmit({ preventDefault() {} });
  view.render();
  view.scrolls.length = 0;
  view.props.chain = { ...view.props.chain, version: 2, messages: [...view.props.chain.messages, message("third", "needle")] };
  view.render();
  assert.equal(view.status(), "2 of 3 messages");
  assert.equal(selected(view)[0].props["aria-label"], "Matching message 2 of 3");
  assert.equal(view.scrolls.length, 0);
});

test("a replacement conversation clears search and its revealed history", () => {
  const view = chat(conversation([message("first", "needle")]));
  view.search("needle");
  view.props.chain = conversation(Array.from({ length: 205 }, (_, i) => message(`new${i}`, "needle")), "replacement");
  view.render();
  assert.equal(view.field(), undefined);
  assert.equal(selected(view).length, 0);
  assert.equal(view.all(node => node.type === "article").length, 200);
  view.search("");
  assert.equal(view.status(), "Saved text only");
});

test("optimistic outgoing text is excluded until storage confirms it", () => {
  const view = chat(conversation([message("saved", "Already saved")]), { "kadan.draft.guide": "Outgoing needle" });
  view.search("needle");
  view.all(node => node.type === "Composer")[0].props.onSend(false);
  view.render();
  assert.equal(view.status(), "No matching messages");
  assert.equal(view.all(node => node.type === "article").length, 2);
  assert.equal(selected(view).length, 0);
});

test("closing find leaves revealed messages and the scroll position alone", () => {
  const view = chat(conversation([message("old", "needle"), ...Array.from({ length: 201 }, (_, i) => message(`m${i}`, "Recent"))]));
  view.search("needle");
  view.scrolls.length = 0;
  view.named("Close find").props.onClick();
  view.render();
  assert.equal(view.all(node => node.type === "article").length, 202);
  assert.equal(view.scrolls.length, 0);
  assert.equal(selected(view).length, 0);
});

test("an emptied conversation keeps Close find available", () => {
  const view = chat(conversation([message("saved", "needle")]));
  view.search("needle");
  view.props.chain = { ...view.props.chain, version: 2, messages: [] };
  view.render();
  assert.equal(view.status(), "No matching messages");
  assert.equal(view.named("Close find").props.disabled, false);
  view.named("Close find").props.onClick();
  view.render();
  assert.equal(view.named("Find in conversation").props.disabled, true);
});

const jump = view => view.all(node => node.props?.className === "jump-latest")[0];
function appendMessage(view, text = "A new reply") {
  view.props.chain = {
    ...view.props.chain, version: view.props.chain.version + 1,
    messages: [...view.props.chain.messages, message(`new-${view.props.chain.version}`, text, "assistant")],
  };
  view.viewport.scrollHeight += 100;
  view.render();
}

test("opening a conversation starts at latest without a jump control", () => {
  const view = chat(conversation([message("saved", "Saved text")]));
  assert.equal(view.scrolls.at(-1).top, view.viewport.scrollHeight);
  assert.equal(view.viewport.scrollTop, view.viewport.scrollHeight - view.viewport.clientHeight);
  assert.equal(jump(view), undefined);
});

test("reading earlier messages keeps the position and visible history when replies and typing arrive", async () => {
  const view = chat(conversation(Array.from({ length: 201 }, (_, i) => message(`m${i}`, `Message ${i}`))));
  view.scroll(500);
  const position = view.viewport.scrollTop;
  view.scrolls.length = 0;
  appendMessage(view);
  assert.equal(view.viewport.scrollTop, position);
  assert.equal(view.scrolls.length, 0);
  assert.equal(view.all(node => node.type === "article").length, 201);
  assert.equal(view.all(node => node.type === "Markdown")[0].props.children, "Message 1");
  assert.ok(jump(view));
  await view.updateRuns([{ id: "running", conversation_id: view.props.chain.id, status: "running" }]);
  assert.equal(view.scrolls.length, 0);
  assert.equal(view.viewport.scrollTop, position);
});

test("messages and typing follow when the reader is near latest", async () => {
  const view = chat(conversation([message("saved", "Saved text")]));
  view.scroll(40);
  view.scrolls.length = 0;
  appendMessage(view);
  assert.equal(view.scrolls.at(-1).top, view.viewport.scrollHeight);
  assert.equal(view.viewport.scrollTop, view.viewport.scrollHeight - view.viewport.clientHeight);
  assert.equal(jump(view), undefined);
  view.scrolls.length = 0;
  await view.updateRuns([{ id: "running", conversation_id: view.props.chain.id, status: "running" }]);
  assert.equal(view.scrolls.at(-1).top, view.viewport.scrollHeight);
});

test("scrolling back near latest resumes following", () => {
  const view = chat(conversation([message("saved", "Saved text")]));
  view.scroll(500);
  assert.ok(jump(view));
  view.scroll(20);
  assert.equal(jump(view), undefined);
  view.scrolls.length = 0;
  appendMessage(view);
  assert.equal(view.scrolls.at(-1).top, view.viewport.scrollHeight);
});

test("Jump to latest returns to the bottom, resumes following and gives focus to the message region", () => {
  const view = chat(conversation([message("saved", "Saved text")]));
  view.scroll(500);
  jump(view).props.onClick();
  view.render();
  assert.equal(jump(view), undefined);
  assert.equal(view.viewport.scrollTop, view.viewport.scrollHeight - view.viewport.clientHeight);
  assert.equal(view.focuses.at(-1).props["aria-label"], "Conversation messages");
  view.scrolls.length = 0;
  appendMessage(view);
  assert.equal(view.scrolls.at(-1).top, view.viewport.scrollHeight);
});

test("Jump explicitly closes Find and clears its selected match before following latest", () => {
  const view = chat(conversation([message("saved", "needle")]));
  view.search("needle");
  assert.equal(jump(view).props.children.at(-1), "Close find and jump to latest");
  view.scrolls.length = 0;
  jump(view).props.onClick();
  view.render();
  assert.equal(view.field(), undefined);
  assert.equal(selected(view).length, 0);
  assert.equal(jump(view), undefined);
  assert.ok(view.scrolls.every(scroll => scroll.top === view.viewport.scrollHeight));
  appendMessage(view, "Another needle");
  assert.equal(selected(view).length, 0);
  assert.equal(view.scrolls.at(-1).top, view.viewport.scrollHeight);
});

test("closing Find at an earlier match keeps follow paused for incoming replies", () => {
  const view = chat(conversation([message("saved", "needle")]));
  view.search("needle");
  view.named("Close find").props.onClick();
  view.render();
  view.scrolls.length = 0;
  appendMessage(view);
  assert.equal(view.scrolls.length, 0);
  assert.ok(jump(view));
});

test("sending one's own message returns to latest while an active Find query keeps its position", () => {
  for (const searching of [false, true]) {
    const view = chat(conversation([message("saved", "needle")]));
    view.scroll(500);
    if (searching) view.search("needle");
    view.all(node => node.type === "Composer")[0].props.onChange("My new message");
    view.render();
    view.scrolls.length = 0;
    view.all(node => node.type === "Composer")[0].props.onSend(false);
    view.render();
    if (searching) {
      assert.equal(view.field().props.value, "needle");
      assert.equal(selected(view).length, 1);
      assert.equal(view.scrolls.length, 0);
      assert.ok(jump(view));
    } else {
      assert.equal(view.scrolls.at(-1).top, view.viewport.scrollHeight);
      assert.equal(jump(view), undefined);
    }
  }
});

test("changing conversation resets search and follow even when the message count stays the same", () => {
  const view = chat(conversation([message("saved", "needle")]));
  view.scroll(500);
  view.search("needle");
  view.scrolls.length = 0;
  view.props.chain = conversation([message("other", "New conversation")], "replacement");
  view.render();
  assert.equal(view.field(), undefined);
  assert.equal(jump(view), undefined);
  assert.equal(view.scrolls.at(-1).top, view.viewport.scrollHeight);
  view.scrolls.length = 0;
  appendMessage(view);
  assert.equal(view.scrolls.at(-1).top, view.viewport.scrollHeight);
});
