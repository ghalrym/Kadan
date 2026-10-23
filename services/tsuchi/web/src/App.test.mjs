import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { runInNewContext } from "node:vm";
import ts from "typescript";

const sources = Object.fromEntries(
  ["App", "ResourceEditor", "AssetEditor", "types"].map(name => [name,
    ts.transpileModule(readFileSync(new URL(`./${name}.${name === "types" ? "ts" : "tsx"}`, import.meta.url), "utf8"), {
      compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX },
    }).outputText,
  ]),
);
const agent = { key: "guide", name: "Guide" };
const memory = {
  id: "memory", agent_key: agent.key, kind: "memory", title: "Saved memory",
  markdown: "Saved text", tags: [], sources: [], attachment_ids: [],
  parent_id: null, deck: null, messages: [], version: 1,
};
const asset = {
  id: "asset", agent_key: agent.key, title: "Saved attachment", filename: "note.txt",
  media_type: "text/plain", is_image: false, tags: [], sources: [], version: 1,
};
const tick = () => new Promise(resolve => setImmediate(resolve));
function elements(node) {
  if (!node || typeof node !== "object") return [];
  return [node, ...[node.props?.children].flat(Infinity).flatMap(elements)];
}
function find(tree, predicate) {
  const node = elements(tree).find(predicate);
  assert.ok(node, "Expected element in current component output");
  return node;
}

// Direct component/callback units: no DOM, browser, renderer, or live service.
function app(search = "") {
  let scope;
  const requests = [], confirmations = [];
  let discard = false;
  const react = {
    useState(initial) {
      const owner = scope, index = owner.cursor++;
      if (!(index in owner.slots))
        owner.slots[index] = typeof initial === "function" ? initial() : initial;
      return [owner.slots[index], value => {
        owner.writes++;
        owner.slots[index] = typeof value === "function" ? value(owner.slots[index]) : value;
      }];
    },
    useRef(initial) {
      const index = scope.cursor++;
      return scope.slots[index] ??= { current: initial };
    },
    useId: () => "editor-form",
    useEffect(effect, dependencies) {
      const owner = scope, index = owner.cursor++;
      const previous = owner.slots[index];
      if (!previous || dependencies.some((value, i) => value !== previous.dependencies[i])) {
        owner.slots[index] = { dependencies, cleanup: previous?.cleanup };
        owner.effects.push(() => {
          owner.slots[index].cleanup?.();
          owner.slots[index].cleanup = effect();
        });
      }
    },
  };
  function hooks() {
    return {
      slots: [], effects: [], cursor: 0, writes: 0,
      run(component, props) {
        scope = this;
        this.cursor = 0;
        const tree = component(props);
        this.effects.splice(0).forEach(effect => effect());
        return tree;
      },
      cleanup() {
        this.slots.forEach(slot => slot?.cleanup?.());
      },
    };
  }
  const jsx = (type, props) => ({ type, props });
  const modules = {
    react,
    "react/jsx-runtime": { jsx, jsxs: jsx },
    "lucide-react": {},
    "./AgentForm": { AgentForm: "AgentForm" },
    "./components": Object.fromEntries([
      "Dialog", "ErrorNotice", "Loading", "Markdown", "SelectItems", "SourceLinks", "SourcesField", "TagsField",
    ].map(name => [name, name])),
    "./LibraryPages": { AssetsPage: "AssetsPage", ResourcesPage: "ResourcesPage", TrashPage: "TrashPage" },
    "./OverviewPages": { OverviewPage: "OverviewPage", ActivityPage: "ActivityPage" },
    "./DeckEditor": { DeckEditor: "DeckEditor", SlidePreview: "SlidePreview" },
    "./ConversationPanel": {
      ConversationPanel: "ConversationPanel",
      emptyMessageDraft: () => ({ role: "user", content: "", attachment_ids: [], resource_ids: [], sources: [] }),
    },
    "./uuid": { randomUuid: () => "message-id" },
    "./api": {
      agentPath: key => `/api/agents/${key}`,
      assetUrl: () => "asset-content",
      formatBytes: String,
      formatDate: String,
      json: (method, body) => ({ method, body }),
      useLoad: () => ({ data: scope === root ? [agent] : null, loading: false, error: "" }),
      request: (path, options) => new Promise((resolve, reject) => requests.push({ path, options, resolve, reject })),
    },
  };
  for (const name of ["types", "ResourceEditor", "AssetEditor", "App"]) {
    const exports = {};
    runInNewContext(sources[name], {
      exports,
      require: name => { assert.ok(name in modules, name); return modules[name]; },
      Error, URLSearchParams,
      window: {
        location: { search }, setInterval: () => 0, clearInterval() {},
        confirm: message => { confirmations.push(message); return discard; },
      },
      localStorage: { getItem: () => agent.key },
    });
    modules[`./${name}`] = exports;
  }
  const root = hooks(), namespace = hooks();
  const namespaceNode = find(root.run(modules["./App"].App), node => node.props?.agent === agent);
  let tree;
  function render() {
    tree = namespace.run(namespaceNode.type, namespaceNode.props);
  }
  function node(type) {
    return find(tree, node => node.type === type);
  }
  function navigate(label) {
    find(tree, node => node.type === "button" && [node.props.children].flat().includes(label)).props.onClick();
    render();
  }
  function editorNode(type) {
    const current = elements(tree).find(node =>
      [modules["./ResourceEditor"].ResourceEditor, modules["./AssetEditor"].AssetEditor].includes(node.type));
    assert.equal(current?.type, modules[`./${type}Editor`][`${type}Editor`], `${type}Editor must remain selected`);
    return current;
  }
  function editor(type) {
    const owner = hooks();
    let output;
    const initial = editorNode(type).props.initial;
    function renderEditor() {
      const current = editorNode(type);
      assert.equal(current.props.initial, initial, "Newer editor must remain selected");
      output = owner.run(current.type, current.props);
    }
    function titleField() {
      return find(output, node => node.type === "input" && node.props.required && !node.props.type);
    }
    renderEditor();
    return {
      render: renderEditor,
      edit(value) { titleField().props.onChange({ target: { value } }); renderEditor(); },
      title: () => titleField().props.value,
      close() { find(output, node => node.type === "Dialog").props.onClose(); render(); },
      save() { find(output, node => node.type === "form").props.onSubmit({ preventDefault() {} }); renderEditor(); },
    };
  }
  render();
  return {
    render, node, navigate, editor, editorNode, requests, confirmations,
    discard(value) { discard = value; },
    hasEditor: () => elements(tree).some(node => [modules["./ResourceEditor"].ResourceEditor, modules["./AssetEditor"].AssetEditor].includes(node.type)),
    error: () => node("ErrorNotice").props.message,
    search(value) {
      find(tree, node => node.props?.["aria-label"] === "Search all resources").props.onChange({ target: { value } });
      render();
      find(tree, node => node.type === "form").props.onSubmit({ preventDefault() {} });
      render();
    },
    cleanup() { namespace.cleanup(); return namespace.writes; },
    writes: () => namespace.writes,
  };
}

test("late asset read cannot replace a newer memory editor or its draft after navigation", async () => {
  const view = app();
  view.navigate("Attachments");
  const slow = view.node("AssetsPage").props.onOpen(asset);
  view.navigate("Memories");
  const fast = view.node("ResourcesPage").props.onOpen(memory);
  assert.equal(view.requests[0].path, "/api/agents/guide/assets/asset");
  assert.equal(view.requests[1].path, "/api/agents/guide/resources/memory");
  view.requests[1].resolve(memory);
  await fast;
  view.render();
  const draft = view.editor("Resource");
  draft.edit("My unsaved memory");
  view.requests[0].resolve(asset);
  await slow;
  view.render();
  draft.render();
  assert.equal(draft.title(), "My unsaved memory");
  assert.equal(view.confirmations.length, 0);
});

test("late resource read cannot replace a newer asset draft without intervening navigation", async () => {
  const view = app();
  view.navigate("Trash");
  const slow = view.node("TrashPage").props.onResource(memory);
  const fast = view.node("TrashPage").props.onAsset(asset);
  view.requests[1].resolve(asset);
  await fast;
  view.render();
  const draft = view.editor("Asset");
  draft.edit("My unsaved metadata");
  view.requests[0].resolve(memory);
  await slow;
  view.render();
  draft.render();
  assert.equal(draft.title(), "My unsaved metadata");
  assert.equal(view.confirmations.length, 0);
});

for (const [type, page, label, record, initial] of [
  ["Resource", "ResourcesPage", "Memories", memory, "memory"],
  ["Asset", "AssetsPage", "Images", { ...asset, is_image: true }, null],
]) {
  test(`slow ${type} read cannot replace a new ${type} draft`, async () => {
    const view = app();
    view.navigate(label);
    const slow = view.node(page).props.onOpen(record);
    view.node(page).props.onNew("memory");
    view.render();
    assert.equal(view.editorNode(type).props.initial, initial);
    if (type === "Asset") assert.equal(view.editorNode(type).props.imagesOnly, true);
    const draft = view.editor(type);
    draft.edit("New unsaved content");
    view.requests[0].resolve(record);
    await slow;
    view.render();
    draft.render();
    assert.equal(draft.title(), "New unsaved content");
  });

  test(`${type} accepted close invalidates reads and preserves dirty and pending close guards`, async () => {
    const view = app();
    view.navigate(label);
    const open = view.node(page).props.onOpen(record);
    view.requests[0].resolve(record);
    await open;
    view.render();
    const draft = view.editor(type);
    draft.edit("Unsaved draft");
    draft.close();
    assert.equal(view.hasEditor(), true);
    assert.deepEqual(view.confirmations, ["Discard unsaved changes?"]);
    const slow = view.node(page).props.onOpen({ ...record, id: "slow" });
    draft.save();
    view.discard(true);
    draft.close();
    assert.equal(view.hasEditor(), true, "Pending save must block close");
    assert.equal(view.confirmations.length, 1);
    view.requests[2].resolve({ ...record, title: "Unsaved draft", version: 2 });
    await tick();
    draft.render();
    draft.edit("Another unsaved edit");
    draft.close();
    assert.equal(view.hasEditor(), false);
    assert.equal(view.confirmations.length, 2);
    view.requests[1].resolve(record);
    await slow;
    view.render();
    assert.equal(view.hasEditor(), false);
  });

  test(`current ${type} read opens successfully and reports current failures`, async () => {
    const view = app();
    view.navigate(label);
    const open = view.node(page).props.onOpen(record);
    view.requests[0].resolve(record);
    await open;
    view.render();
    assert.equal(view.editorNode(type).props.initial, record);
    const failed = view.node(page).props.onOpen(record);
    view.requests[1].reject(new Error("Current read failed"));
    await failed;
    view.render();
    assert.equal(view.error(), "Current read failed");
    const fallback = view.node(page).props.onOpen(record);
    view.render();
    assert.equal(view.error(), "");
    view.requests[2].reject(null);
    await fallback;
    view.render();
    assert.equal(view.error(), `Unable to open ${type.toLowerCase()}.`);
  });
}

for (const action of ["navigation", "search"]) {
  for (const fails of [false, true]) {
    for (const origin of ["asset", "deep-link"]) {
      test(`${action} invalidates a pending ${origin} read's ${fails ? "failure" : "success"}`, async () => {
        const view = app(origin === "deep-link" ? "?agent=guide&resource=memory" : "");
        let pending;
        if (origin === "asset") {
          view.navigate("Attachments");
          pending = view.node("AssetsPage").props.onOpen(asset);
        }
        if (action === "search") view.search("saved memory");
        else view.navigate("Memories");
        if (fails) view.requests[0].reject(new Error("Obsolete failure"));
        else view.requests[0].resolve(origin === "asset" ? asset : memory);
        await pending;
        await tick();
        view.render();
        assert.equal(view.hasEditor(), false);
        assert.equal(view.error(), "");
        assert.equal(view.node("ResourcesPage").props.kind, action === "search" ? undefined : "memory");
      });
    }
  }
}

for (const first of ["Resource", "Asset"]) {
  test(`stale ${first} failure cannot replace the current cross-type error`, async () => {
    const view = app();
    view.navigate("Trash");
    const second = first === "Resource" ? "Asset" : "Resource";
    const slow = view.node("TrashPage").props[`on${first}`](first === "Resource" ? memory : asset);
    const current = view.node("TrashPage").props[`on${second}`](second === "Resource" ? memory : asset);
    view.requests[1].reject(new Error("Current error"));
    await current;
    view.requests[0].reject(new Error("Stale error"));
    await slow;
    view.render();
    assert.equal(view.error(), "Current error");
  });
}

for (const fails of [false, true]) {
  test(`initial deep-link ${fails ? "failure" : "success"} cannot override a later asset draft or current error`, async () => {
    const view = app("?agent=guide&resource=deep%2Flink");
    assert.equal(view.requests[0].path, "/api/agents/guide/resources/deep%2Flink");
    view.navigate("Attachments");
    const open = view.node("AssetsPage").props.onOpen(asset);
    view.requests[1].resolve(asset);
    await open;
    view.render();
    const draft = view.editor("Asset");
    draft.edit("Keep this draft");
    if (fails) {
      const current = view.node("AssetsPage").props.onOpen(asset);
      view.requests[2].reject(new Error("Current asset error"));
      await current;
      view.requests[0].reject(new Error("Obsolete deep-link error"));
    } else view.requests[0].resolve(memory);
    await tick();
    view.render();
    draft.render();
    assert.equal(draft.title(), "Keep this draft");
    assert.equal(view.error(), fails ? "Current asset error" : "");
  });

  test(`current deep-link ${fails ? "failure remains visible" : "success opens the resource"}`, async () => {
    const view = app("?agent=guide&resource=memory");
    if (fails) view.requests[0].reject(new Error("Deep-link error"));
    else view.requests[0].resolve(memory);
    await tick();
    view.render();
    if (fails) assert.equal(view.error(), "Deep-link error");
    else assert.equal(view.editorNode("Resource").props.initial, memory);
  });

  for (const origin of ["Resource", "Asset", "deep-link"]) {
    test(`namespace cleanup prevents pending ${origin} ${fails ? "failure" : "success"} from publishing`, async () => {
      const view = app(origin === "deep-link" ? "?agent=guide&resource=memory" : "");
      let pending;
      if (origin !== "deep-link") {
        view.navigate("Trash");
        pending = view.node("TrashPage").props[`on${origin}`](origin === "Resource" ? memory : asset);
      }
      const writes = view.cleanup();
      if (fails) view.requests[0].reject(new Error("Old namespace error"));
      else view.requests[0].resolve(origin === "Asset" ? asset : memory);
      await pending;
      await tick();
      assert.equal(view.writes(), writes, "Disposed namespace must receive no state updates");
    });
  }
}
