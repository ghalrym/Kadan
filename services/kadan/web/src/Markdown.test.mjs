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
  assert.match(source, /remarkPlugins=\{markdownPlugins\}/);
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
