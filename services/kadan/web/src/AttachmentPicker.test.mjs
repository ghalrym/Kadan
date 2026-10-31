import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { runInNewContext } from "node:vm";
import ts from "typescript";

const source = ts.transpileModule(readFileSync(new URL("./AttachmentPicker.tsx", import.meta.url), "utf8"), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX },
}).outputText;

function picker(disabled = false) {
  const calls = [], errors = [];
  const jsx = (type, props) => ({ type, props });
  const modules = {
    react: { useState: initial => [initial, value => errors.push(value)] },
    "react/jsx-runtime": { jsx, jsxs: jsx },
    "lucide-react": { X: "X" },
  };
  const exports = {};
  runInNewContext(source, { exports, require: name => modules[name] });
  const attachments = {
    entries: [
      { id: "uncertain", name: "report.pdf", size: 3, status: "error", error: "Lost response" },
      { id: "missing", name: "notes.txt", size: 4, status: "reselect" },
    ],
    retry: id => calls.push(["retry", id]),
    remove: id => calls.push(["remove", id]),
    reselect(id, file) {
      if (file.name !== "notes.txt") throw new Error("Choose the original file");
      calls.push(["reselect", id, file]);
    },
  };
  const tree = exports.default({ attachments, disabled });
  function elements(node = tree) {
    if (!node || typeof node !== "object") return [];
    return [node, ...[node.props?.children].flat(Infinity).flatMap(child => elements(child ?? null))];
  }
  return { calls, errors, elements: elements() };
}

test("retry and removal target the selected draft entry without deleting storage", () => {
  const view = picker();
  view.elements.find(node => node.props?.children === "Retry").props.onClick();
  view.elements.find(node => node.props?.["aria-label"] === "Remove report.pdf").props.onClick();
  assert.deepEqual(view.calls, [["retry", "uncertain"], ["remove", "uncertain"]]);
});

test("reselection reports an invalid original and resets the input on either outcome", () => {
  const view = picker();
  const input = view.elements.find(node => node.type === "input");
  const target = { files: [new File(["test"], "other.txt")], value: "other.txt" };
  input.props.onChange({ target });
  assert.deepEqual(view.errors, ["Choose the original file"]);
  assert.equal(target.value, "");
  target.files = [new File(["test"], "notes.txt")];
  input.props.onChange({ target });
  assert.deepEqual(view.calls, [["reselect", "missing", target.files[0]]]);
  assert.equal(target.value, "");
});

test("blocked picker disables retry, reselection and removal", () => {
  const controls = picker(true).elements.filter(node => ["button", "input"].includes(node.type));
  assert.equal(controls.length, 4);
  assert.ok(controls.every(node => node.props.disabled));
});
