import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { runInNewContext } from "node:vm";
import ts from "typescript";
import { newResource, singularLabels } from "./types.ts";

const source = ts.transpileModule(
  readFileSync(new URL("./ResourceEditor.tsx", import.meta.url), "utf8"),
  { compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX } },
).outputText;
const plain = value => JSON.parse(JSON.stringify(value));
const saved = (kind = "memory", patch = {}) => ({
  ...newResource(kind),
  title: "Research",
  markdown: "Saved content",
  parent_id: "parent-space",
  tags: ["research"],
  sources: [{ label: "Reference", url: "https://example.com/reference" }],
  attachment_ids: ["attached-file"],
  id: "original",
  agent_key: "guide",
  version: 3,
  created_at: "2026-10-23T10:00:00Z",
  updated_at: "2026-10-23T11:00:00Z",
  deleted_at: null,
  ...patch,
});
const slide = (id, patch = {}) => ({
  id, layout: "image", title: "A slide", body: "Slide text", notes: "Speaker notes",
  image_asset_id: "shared-image", ...patch,
});
const text = node => {
  if (typeof node === "string" || typeof node === "number") return String(node);
  if (!node || typeof node !== "object") return "";
  return [node.props?.children].flat(Infinity).map(text).join("");
};

// Call the component and its handlers directly; hooks, children and requests are mocked.
function editor(initial) {
  const slots = [], requests = [], confirmations = [];
  let cursor = 0, tree, changed = 0, closed = 0, confirmResult = false;
  let response = () => { assert.fail("Unexpected request"); };
  const react = {
    useId: () => "resource-form",
    useState(initialValue) {
      const index = cursor++;
      if (!(index in slots)) slots[index] = typeof initialValue === "function" ? initialValue() : initialValue;
      return [slots[index], value => {
        slots[index] = typeof value === "function" ? value(slots[index]) : value;
      }];
    },
  };
  const jsx = (type, props, key) => ({ type, props, key });
  const modules = {
    react,
    "react/jsx-runtime": { jsx, jsxs: jsx },
    "lucide-react": {},
    "./uuid": { randomUuid: () => "new-message" },
    "./types": { newResource, singularLabels },
    "./components": Object.fromEntries([
      "Dialog", "ErrorNotice", "Loading", "Markdown", "SelectItems", "SourceLinks", "SourcesField", "TagsField",
    ].map(name => [name, name])),
    "./DeckEditor": { DeckEditor: "DeckEditor", SlidePreview: "SlidePreview" },
    "./ConversationPanel": {
      ConversationPanel: "ConversationPanel",
      emptyMessageDraft: () => ({ role: "user", content: "", attachment_ids: [], resource_ids: [], sources: [] }),
    },
    "./api": {
      agentPath: key => `/api/agents/${encodeURIComponent(key)}`,
      assetUrl: (key, id) => `/api/agents/${key}/assets/${id}/content`,
      formatDate: value => value,
      json: (method, body) => ({ method, ...(body === undefined ? {} : { body: JSON.stringify(body) }) }),
      async request(path, options) {
        requests.push({ path, ...options });
        return response(path, options);
      },
      useLoad: (_load, dependencies) => ({
        data: dependencies.length === 1
          ? { resources: [saved("space", { id: "parent-space" })], assets: [] }
          : [{ version: 1, created_at: "2026-10-23T09:00:00Z", snapshot: saved() }],
        loading: false,
        error: "",
      }),
    },
  };
  const exports = {};
  runInNewContext(source, {
    exports, structuredClone, Error,
    require: name => { assert.ok(name in modules, name); return modules[name]; },
    window: { confirm(message) { confirmations.push(message); return confirmResult; } },
  });
  function render() {
    cursor = 0;
    tree = exports.ResourceEditor({
      agentKey: "guide", initial,
      onChanged: () => changed++, onClose: () => closed++,
    });
  }
  function findAll(predicate, node = tree) {
    if (!node || typeof node !== "object") return [];
    return [
      ...(predicate(node) ? [node] : []),
      ...[node.props?.children].flat(Infinity).flatMap(child => findAll(predicate, child ?? null)),
    ];
  }
  const element = type => {
    const matches = findAll(node => node.type === type);
    assert.equal(matches.length, 1, type);
    return matches[0];
  };
  const buttons = label => findAll(node => node.type === "button" && text(node) === label);
  const button = label => {
    const matches = buttons(label);
    assert.equal(matches.length, 1, label);
    return matches[0];
  };
  function click(label) {
    const control = button(label);
    assert.ok(!control.props.disabled, label);
    control.props.onClick();
    render();
  }
  function title(value) {
    const input = element("input");
    if (value === undefined) return input.props.value;
    input.props.onChange({ target: { value } });
    render();
  }
  function submit() {
    element("form").props.onSubmit({ preventDefault() {} });
    render();
  }
  async function settle() {
    await new Promise(resolve => setImmediate(resolve));
    render();
  }
  render();
  return {
    render, findAll, element, buttons, button, click, title, submit, settle, requests, confirmations,
    respond: handler => { response = handler; },
    confirm: value => { confirmResult = value; },
    changed: () => changed, closed: () => closed,
  };
}

for (const kind of ["persona", "memory", "space", "deck"]) {
  test(`${kind} offers one secondary copy action on Content and Details only`, () => {
    const view = editor(saved(kind));
    assert.equal(view.buttons("Make a copy").length, 1);
    assert.equal(view.button("Make a copy").props.type, "button");
    assert.notEqual(view.button("Make a copy").props.className, "primary");
    view.click("Details");
    assert.equal(view.buttons("Make a copy").length, 1);
    view.click("History");
    assert.equal(view.buttons("Make a copy").length, 0);
    view.click("Content");
    view.click("Make a copy");
    assert.equal(view.buttons("Make a copy").length, 0);
    assert.equal(view.element("Dialog").props.title, `New ${singularLabels[kind]}`);
    assert.equal(view.title(), "Copy of Research");
    assert.equal(view.button("History").props.disabled, true);
    assert.equal(view.buttons("Trash").length, 0);
    assert.equal(view.button("Create").props.disabled, false);
    assert.equal(view.requests.length, 0);
    assert.equal(view.changed(), 0);
    assert.equal(view.closed(), 0);
  });

  test(`${kind} does not offer copying in Trash or a new draft`, () => {
    for (const initial of [kind, saved(kind, { deleted_at: "2026-10-23T12:00:00Z" })]) {
      const view = editor(initial);
      assert.equal(view.buttons("Make a copy").length, 0);
      view.click("Details");
      assert.equal(view.buttons("Make a copy").length, 0);
    }
  });
}

test("copy uses the current edits and creates a sibling through the same-namespace POST", async () => {
  const original = saved("persona");
  const snapshot = structuredClone(original);
  const view = editor(original);
  view.title("A variation");
  view.element("textarea").props.onChange({ target: { value: "Edited instructions" } });
  view.render();
  view.click("Details");
  view.element("select").props.onChange({ target: { value: "edited-parent" } });
  view.element("TagsField").props.onChange(["edited-tag"]);
  view.element("SourcesField").props.onChange([{ label: "Edited source", url: "https://example.com/edited" }]);
  view.element("SelectItems").props.onChange(["edited-file"]);
  view.render();
  const formKey = view.element("fieldset").key;
  view.click("Make a copy");
  assert.notEqual(view.element("fieldset").key, formKey);
  assert.equal(view.title(), "Copy of A variation");
  assert.equal(view.element("textarea").props.value, "Edited instructions");
  assert.equal(view.requests.length, 0);
  view.respond((_path, options) => saved("persona", {
    ...JSON.parse(options.body), id: "copy", version: 1,
  }));
  view.submit();
  await view.settle();
  assert.equal(view.requests.length, 1);
  const { path, method, body } = view.requests[0];
  assert.equal(path, "/api/agents/guide/resources");
  assert.equal(method, "POST");
  assert.deepEqual(JSON.parse(body), {
    ...newResource("persona"),
    title: "Copy of A variation", markdown: "Edited instructions", parent_id: "edited-parent",
    tags: ["edited-tag"], sources: [{ label: "Edited source", url: "https://example.com/edited" }],
    attachment_ids: ["edited-file"],
  });
  assert.deepEqual(original, snapshot);
  assert.equal(view.changed(), 1);
  assert.equal(view.closed(), 0);
  assert.equal(view.button("History").props.disabled, false);
});

test("deck copy deeply isolates nested draft values while retaining slide and file IDs", async () => {
  const original = saved("deck", {
    deck: { ...newResource("deck").deck, slides: [slide("first"), slide("second")] },
  });
  const snapshot = structuredClone(original);
  const view = editor(original);
  const editedDeck = structuredClone(original.deck);
  editedDeck.slides[1].notes = "Unsaved notes";
  view.element("DeckEditor").props.onChange(editedDeck);
  view.render();
  view.click("Make a copy");
  const copy = view.element("DeckEditor").props.deck;
  assert.deepEqual(copy, editedDeck);
  assert.notEqual(copy, editedDeck);
  assert.notEqual(copy.slides, editedDeck.slides);
  assert.notEqual(copy.slides[1], editedDeck.slides[1]);
  copy.slides[1].notes = "Copy notes";
  assert.equal(editedDeck.slides[1].notes, "Unsaved notes");
  view.click("Details");
  const tags = view.element("TagsField").props.value;
  const sources = view.element("SourcesField").props.value;
  const attachments = view.element("SelectItems").props.selected;
  assert.notEqual(tags, original.tags);
  assert.notEqual(sources, original.sources);
  assert.notEqual(sources[0], original.sources[0]);
  assert.notEqual(attachments, original.attachment_ids);
  tags.push("copy-tag");
  sources[0].label = "Copy reference";
  attachments.push("another-file");
  view.respond((_path, options) => saved("deck", { ...JSON.parse(options.body), id: "copy", version: 1 }));
  view.submit();
  await view.settle();
  const body = JSON.parse(view.requests[0].body);
  assert.deepEqual(body.deck.slides.map(item => item.id), ["first", "second"]);
  assert.deepEqual(body.deck.slides.map(item => item.image_asset_id), ["shared-image", "shared-image"]);
  assert.deepEqual(body.attachment_ids, ["attached-file", "another-file"]);
  assert.deepEqual(original, snapshot);
});

test("copy bounds the title and leaves blank titles requiring a useful name", () => {
  const long = editor(saved("memory", { title: "a".repeat(500) }));
  long.click("Make a copy");
  assert.equal(long.title(), `Copy of ${"a".repeat(292)}`);
  assert.equal(long.element("input").props.maxLength, 300);
  for (const title of ["", "   \t"]) {
    const view = editor(saved());
    view.title(title);
    view.click("Make a copy");
    assert.equal(view.title(), "");
    assert.equal(view.button("Create").props.disabled, true);
    view.title("Useful name");
    assert.equal(view.button("Create").props.disabled, false);
  }
});

for (const [boundary, title, expected] of [
  ["split emoji", `${"a".repeat(291)}😀`, `Copy of ${"a".repeat(291)}`],
  ["intact emoji", `${"a".repeat(290)}😀`, `Copy of ${"a".repeat(290)}😀`],
  ["emoji-rich original", "😀".repeat(150), `Copy of ${"😀".repeat(146)}`],
]) {
  test(`copy keeps a well-formed title within 300 UTF-16 units at the ${boundary} boundary`, async () => {
    const view = editor(saved("memory", { title }));
    view.click("Make a copy");
    const copiedTitle = view.title();
    assert.equal(copiedTitle.isWellFormed(), true);
    assert.ok(copiedTitle.length <= 300);
    assert.equal(copiedTitle, expected);
    assert.equal(view.element("input").props.maxLength, 300);
    assert.equal(view.requests.length, 0);
    view.respond((_path, options) => saved("memory", { ...JSON.parse(options.body), id: "copy", version: 1 }));
    view.submit();
    await view.settle();
    assert.equal(view.requests.length, 1);
    const { path, method, body } = view.requests[0];
    assert.equal(path, "/api/agents/guide/resources");
    assert.equal(method, "POST");
    assert.equal(JSON.parse(body).title, copiedTitle);
  });
}

test("copy returns to content editing and preserves the dirty-close warning", () => {
  const view = editor(saved());
  view.click("Preview");
  view.click("Details");
  view.click("Make a copy");
  assert.equal(view.element("textarea").props["aria-label"], "Markdown content");
  assert.equal(view.button("Content").props["aria-selected"], true);
  assert.equal(view.findAll(node => node.props?.className === "footer-status").map(text)[0], "Unsaved changes");
  view.element("Dialog").props.onClose();
  assert.deepEqual(view.confirmations, ["Discard unsaved changes?"]);
  assert.equal(view.closed(), 0);
  view.confirm(true);
  view.element("Dialog").props.onClose();
  assert.equal(view.closed(), 1);
});

test("Space notice stays visible through edits and failure, then clears after creation", async () => {
  const view = editor(saved("space", { tags: ["project"] }));
  view.click("History");
  view.click("Details");
  view.click("Make a copy");
  const notice = () => view.findAll(node => node.type === "p" && node.props.role === "status");
  assert.match(text(notice()[0]), /children are not included/);
  assert.match(text(notice()[0]), /Project tags are kept/);
  assert.equal(view.findAll(node => node.type === "details").length, 0);
  view.title("Copied project document");
  assert.equal(notice().length, 1);
  view.respond(() => { throw new Error("Restore attached assets before linking them"); });
  view.submit();
  await view.settle();
  assert.equal(notice().length, 1);
  assert.equal(view.title(), "Copied project document");
  assert.equal(view.element("ErrorNotice").props.message, "Restore attached assets before linking them");
  assert.equal(view.button("History").props.disabled, true);
  assert.equal(view.changed(), 0);
  view.respond((_path, options) => saved("space", { ...JSON.parse(options.body), id: "copy", version: 1 }));
  view.submit();
  await view.settle();
  assert.equal(notice().length, 0);
  assert.equal(view.element("ErrorNotice").props.message, "");
  assert.equal(view.changed(), 1);
  for (const request of view.requests) {
    assert.equal(request.path, "/api/agents/guide/resources");
    assert.equal(request.method, "POST");
    assert.deepEqual(JSON.parse(request.body).tags, ["project"]);
    assert.equal(JSON.parse(request.body).parent_id, "parent-space");
  }
});

test("pending save disables copying; copying afterwards clears stale errors and notices", async () => {
  const view = editor(saved());
  let finish;
  view.respond(() => new Promise(resolve => { finish = resolve; }));
  view.submit();
  assert.equal(view.button("Make a copy").props.disabled, true);
  view.button("Make a copy").props.onClick();
  view.render();
  assert.equal(view.buttons("Create").length, 0);
  assert.equal(view.requests.length, 1);
  finish(saved("memory", { version: 4 }));
  await view.settle();
  assert.equal(view.button("Make a copy").props.disabled, false);
  assert.match(text(view.findAll(node => node.props?.className === "footer-status")[0]), /Saved/);
  view.respond(() => { throw new Error("Save failed"); });
  view.submit();
  await view.settle();
  assert.equal(view.element("ErrorNotice").props.message, "Save failed");
  view.click("Make a copy");
  assert.equal(view.element("ErrorNotice").props.message, "");
  assert.equal(text(view.findAll(node => node.props?.className === "footer-status")[0]), "Unsaved changes");
  assert.equal(view.requests.length, 2);
  assert.equal(view.changed(), 1);
});

test("conversation copy stays unavailable, including during a pending message", async () => {
  const view = editor(saved("conversation"));
  assert.equal(view.buttons("Make a copy").length, 0);
  view.click("Details");
  assert.equal(view.buttons("Make a copy").length, 0);
  view.click("Content");
  view.element("ConversationPanel").props.onChange({
    role: "user", content: "An unsent message", attachment_ids: [], resource_ids: [], sources: [],
  });
  view.render();
  let finish;
  view.respond(() => new Promise(resolve => { finish = resolve; }));
  const pending = view.element("ConversationPanel").props.onAppend();
  view.render();
  assert.equal(view.element("ConversationPanel").props.busy, true);
  assert.equal(view.buttons("Make a copy").length, 0);
  finish(saved("conversation", { version: 4 }));
  await pending;
  view.render();
  assert.equal(view.buttons("Make a copy").length, 0);
  assert.deepEqual(plain(view.element("ConversationPanel").props.draft), {
    role: "user", content: "", attachment_ids: [], resource_ids: [], sources: [],
  });
});
