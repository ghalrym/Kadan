import assert from "node:assert/strict";
import { test } from "node:test";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { readFileSync } from "node:fs";

const render = (text) => renderToStaticMarkup(createElement(ReactMarkdown, { remarkPlugins: [remarkGfm] }, text));

test("chat Markdown enables GitHub-style tables, task lists and strikethrough", () => {
  const source = readFileSync(new URL("./Markdown.tsx", import.meta.url), "utf8");
  assert.match(source, /import remarkGfm from "remark-gfm"/);
  assert.match(source, /markdownPlugins/);
  assert.match(source, /className="markdown-table"/);
});

test("an agent reply with a table directly under its intro renders as a table", () => {
  const html = render([
    "All nine sprite animations (rows are zero-based):",
    "| Animation | Atlas row | Frames / cycle |",
    "|---|---|---|",
    "| `idle` | 0 | 6 frames, 1,100ms |",
    "| `running-right` | 1 | 8 frames, 1,060ms |",
  ].join("\n"));
  assert.match(html, /<p>All nine sprite animations \(rows are zero-based\):<\/p>/);
  assert.match(html, /<table>/);
  assert.match(html, /<th>Animation<\/th>/);
  assert.match(html, /<td><code>running-right<\/code><\/td>/);
  assert.doesNotMatch(html, /\|---/);
});

test("task lists, strikethrough and bare links render", () => {
  const html = render("- [x] done\n- [ ] todo\n\n~~old~~ see https://example.com");
  assert.match(html, /type="checkbox"/);
  assert.match(html, /<del>old<\/del>/);
  assert.match(html, /<a href="https:\/\/example.com">/);
});

test("known @handles become highlighted mentions outside code, unknown ones stay plain", async () => {
  const ts = (await import("typescript")).default;
  const { runInNewContext } = await import("node:vm");
  const exports = {};
  runInNewContext(
    ts.transpileModule(readFileSync(new URL("./mentions.ts", import.meta.url), "utf8"), {
      compilerOptions: { module: ts.ModuleKind.CommonJS },
    }).outputText,
    { exports },
  );
  const html = renderToStaticMarkup(createElement(
    ReactMarkdown,
    { remarkPlugins: [remarkGfm, exports.remarkMentions(new Set(["dot", "kadan"]))] },
    "Hey @Dot and @nobody, ask @kadan. Not `@dot` or [@dot](https://x.test) or me@dot.com",
  ));
  assert.match(html, /Hey <span class="mention">@Dot<\/span> and @nobody, ask <span class="mention">@kadan<\/span>\./);
  assert.match(html, /<code>@dot<\/code>/);
  assert.match(html, /<a href="https:\/\/x.test">@dot<\/a>/);
  assert.doesNotMatch(html, /me<span/);
});
