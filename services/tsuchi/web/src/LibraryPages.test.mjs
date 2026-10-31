import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { test } from "node:test";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import ts from "typescript";

const require = createRequire(import.meta.url);

// Node's type stripping does not handle TSX. Compile with the existing compiler
// so these units exercise the page with only its data-loading boundary replaced.
function compile(path, imports = {}) {
  const source = readFileSync(new URL(path, import.meta.url), "utf8");
  const { outputText } = ts.transpileModule(source, {
    compilerOptions: {
      module: ts.ModuleKind.CommonJS,
      jsx: ts.JsxEmit.ReactJSX,
      target: ts.ScriptTarget.ES2022,
    },
  });
  const exports = {};
  new Function("require", "exports", outputText)(
    (name) => imports[name] ?? require(name),
    exports,
  );
  return exports;
}

const components = compile("./components.tsx");
const types = compile("./types.ts");

function render(state) {
  const { ResourcesPage } = compile("./LibraryPages.tsx", {
    "./api": { useLoad: () => state },
    "./components": components,
    "./types": types,
  });
  return renderToStaticMarkup(
    createElement(ResourcesPage, {
      agentKey: "test-agent",
      kind: "space",
      refresh: 0,
      onOpen: () => {},
      onNew: () => {},
    }),
  );
}

test("Spaces shows loading without the empty-library action", () => {
  const html = render({ data: null, loading: true, error: "" });
  assert.match(html, /aria-label="Loading"/);
  assert.doesNotMatch(html, /No spaces yet|Create Space|role="alert"/);
});

test("a failed Spaces load shows its error without claiming the library is empty", () => {
  const html = render({
    data: null,
    loading: false,
    error: "Unable to load data.",
  });
  assert.match(html, /role="alert"/);
  assert.match(html, /Unable to load data\./);
  assert.doesNotMatch(html, /No spaces yet|Create Space|class="space-tree"/);
});

test("a successfully loaded empty library offers to create a Space", () => {
  const html = render({ data: [], loading: false, error: "" });
  assert.match(html, /No spaces yet\./);
  assert.match(html, /Create Space/);
  assert.doesNotMatch(html, /role="alert"|aria-label="Loading"/);
});

test("a successfully loaded library without Spaces offers to create a Space", () => {
  const html = render({
    data: [{ id: "memory", kind: "memory", parent_id: null }],
    loading: false,
    error: "",
  });
  assert.match(html, /No spaces yet\./);
  assert.match(html, /Create Space/);
  assert.doesNotMatch(html, /class="space-tree"/);
});

test("a successfully loaded Spaces tree renders its Space and child resources", () => {
  const html = render({
    data: [
      { id: "space", kind: "space", title: "Research", parent_id: null },
      { id: "memory", kind: "memory", title: "Notes", parent_id: "space" },
    ],
    loading: false,
    error: "",
  });
  assert.match(html, /class="space-tree"/);
  assert.match(html, /Research/);
  assert.match(html, /1 children/);
  assert.match(html, /Notes/);
  assert.doesNotMatch(html, /No spaces yet|Create Space|role="alert"/);
});
