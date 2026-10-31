import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { runInNewContext } from "node:vm";
import ts from "typescript";

const source = ts.transpileModule(readFileSync(new URL("./SaveChatResult.tsx", import.meta.url), "utf8"), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX },
}).outputText;

function editor(disabled = false) {
  const slots = [], calls = [];
  let cursor = 0, tree, resolve, reject;
  const jsx = (type, props) => ({ type, props });
  const exports = {};
  const modules = {
    react: {
      useState(initial) {
        const index = cursor++;
        if (!(index in slots)) slots[index] = typeof initial === "function" ? initial() : initial;
        return [slots[index], value => { slots[index] = value; }];
      },
      useRef(value) { const index = cursor++; return slots[index] ??= { current: value }; },
    },
    "react/jsx-runtime": { jsx, jsxs: jsx },
    "./api": {
      agentPath: key => `/agents/${key}`,
      api(path, body) {
        calls.push({ path, body });
        return new Promise((yes, no) => { resolve = yes; reject = no; });
      },
    },
  };
  runInNewContext(source, { exports, require: name => modules[name] });
  function render() {
    cursor = 0;
    tree = exports.default({ agentKey: "guide", conversationId: "chat", disabled,
      message: { id: "reply", role: "assistant", content: "# Useful result\nDetails" },
    });
    return tree;
  }
  function find(node, predicate) {
    if (!node || typeof node !== "object") return null;
    if (predicate(node)) return node;
    for (const child of [node.props?.children].flat(Infinity)) {
      const found = find(child, predicate);
      if (found) return found;
    }
    return null;
  }
  render();
  find(tree, node => node.type === "button").props.onClick();
  render();
  return {
    calls, render,
    find: predicate => find(tree, predicate),
    submit: () => find(tree, node => node.type === "form").props.onSubmit({ preventDefault() {} }),
    resolve: value => resolve(value), reject: error => reject(error),
    setDisabled(value) { disabled = value; render(); },
  };
}

const text = node => JSON.stringify(node);

test("save uses persisted reply identity and confirms only after storage acknowledgement", async () => {
  const view = editor();
  assert.equal(view.find(node => node.type === "input").props.value, "Useful result");
  const pending = view.submit();
  view.render();
  assert.equal(view.find(node => node.type === "button" && node.props.type === "submit").props.disabled, true);
  assert.equal(view.find(node => node.props?.role === "status"), null);
  assert.equal(view.calls[0].path, "/agents/guide/conversations/chat/messages/reply/document");
  assert.equal(view.calls[0].body.title, "Useful result");
  view.resolve({ id: "document", title: "Stored title" });
  await pending;
  assert.match(text(view.render()), /Saved in Workspace/);
  assert.match(text(view.render()), /Stored title/);
  assert.equal(view.find(node => node.type === "form"), null);
});

test("rapid submits cannot create concurrent requests", async () => {
  const view = editor();
  const pending = view.submit();
  await view.submit();
  assert.equal(view.calls.length, 1);
  view.resolve({ id: "document", title: "Title" });
  await pending;
});

test("uncertain save preserves the title and retry uses the same source identity", async () => {
  const view = editor();
  view.find(node => node.type === "input").props.onChange({ target: { value: "My result" } });
  view.render();
  const pending = view.submit();
  view.reject(new Error("Storage unavailable"));
  await pending;
  view.render();
  assert.match(text(view.find(node => node.props?.role === "alert")), /not confirmed/);
  assert.equal(view.find(node => node.type === "input").props.value, "My result");
  const retry = view.submit();
  assert.equal(view.calls[0].path, view.calls[1].path);
  assert.equal(view.calls[1].body.title, "My result");
  view.resolve({ id: "same-document", title: "Original saved title" });
  await retry;
  assert.match(text(view.render()), /Original saved title/);
});

test("storage unavailability and a blank title do not submit", async () => {
  const view = editor(true);
  await view.submit();
  assert.equal(view.calls.length, 0);
  view.setDisabled(false);
  view.find(node => node.type === "input").props.onChange({ target: { value: "  " } });
  view.render();
  await view.submit();
  assert.equal(view.calls.length, 0);
});
