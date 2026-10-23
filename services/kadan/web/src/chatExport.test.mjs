import assert from "node:assert/strict";
import { test } from "node:test";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import ReactMarkdown from "react-markdown";
import { conversationMarkdown } from "./chatExport.ts";

const message = (role, content, extra = {}) => ({
  id: content, role, content, created_at: "2026-10-23T10:11:12Z",
  attachment_ids: [], resource_ids: [], sources: [], ...extra,
});
const chain = (messages) => ({
  id: "conversation-id", agent_key: "haru", title: "A saved chat", version: 7,
  updated_at: "2026-10-23T11:00:00Z", messages,
  markdown: "Hidden resource text", attachment_ids: ["not-a-message-attachment"],
});

test("exports all saved visible messages in order, with quoted Markdown and saved timestamps", () => {
  const messages = Array.from({ length: 205 }, (_, index) => message(index % 2 ? "assistant" : "user", `Message ${index}`));
  messages[1].content = "```js\nconst greeting = 'こんにちは';\n```\n\n**Saved text**";
  messages.push(message("system", "Visible system message"));
  const result = conversationMarkdown(chain(messages), "Haru", "https://kadan.example");
  assert.match(result, /^# Chat with Haru\n/);
  assert.match(result, /Conversation: A saved chat \(conversation-id\)/);
  assert.match(result, /Saved snapshot: 2026-10-23T11:00:00Z · version 7/);
  assert.match(result, /## User · 2026-10-23T10:11:12Z\n\n> Message 0/);
  assert.match(result, /## Assistant · 2026-10-23T10:11:12Z/);
  assert.ok(result.includes("> ```js\n> const greeting = 'こんにちは';\n> ```\n> \n> **Saved text**"));
  assert.ok(result.indexOf("Message 0") < result.indexOf("Message 204"));
  assert.match(result, /## System · 2026-10-23T10:11:12Z\n\n> Visible system message/);
  assert.equal(result.match(/^## /gm).length, 206);
});

test("omits retired tool history and fields outside saved message text", () => {
  const result = conversationMarkdown(chain([
    message("user", "Saved note"),
    message("tool", "Hidden tool result", { attachment_ids: ["hidden-tool-file"] }),
  ]), "Haru", "https://kadan.example");
  assert.ok(result.includes("Saved note"));
  for (const hidden of ["Hidden tool result", "hidden-tool-file", "Hidden resource text", "not-a-message-attachment"]) {
    assert.ok(!result.includes(hidden));
  }
});

test("exports attachment links and resource IDs without reading file bytes or metadata", () => {
  const result = conversationMarkdown(chain([message("assistant", "Here is the file.", {
    attachment_ids: ["file-one", "file-two"], resource_ids: ["persona-id"],
  })]), "Haru", "https://kadan.example:8443");
  assert.ok(result.includes("Attachment: [file-one](<https://kadan.example:8443/api/agents/haru/assets/file-one/content?download=true>)"));
  assert.ok(result.indexOf("file-one") < result.indexOf("file-two"));
  assert.ok(result.includes("Resource: persona-id"));
  assert.ok(result.includes("file contents are not included"));
  assert.ok(!result.includes("!["));
});

test("keeps supported source links, escapes labels and omits unsupported link payloads", () => {
  const result = conversationMarkdown(chain([message("assistant", "Saved", {
    sources: [
      { label: "Document [draft]\nNext", url: "https://example.com/a>b" },
      { label: "Recording", url: "data:audio/wav;base64,PRIVATE_BYTES" },
      { label: "Local file", url: "file:///private/file" },
    ],
  })]), "Haru\n# Another heading", "https://kadan.example");
  assert.ok(result.startsWith("# Chat with Haru \\# Another heading\n"));
  assert.ok(result.includes("Source: [Document \\[draft\\] Next](<https://example.com/a%3Eb>)"));
  assert.ok(result.includes("Source: Recording (link unavailable)"));
  assert.ok(!result.includes("PRIVATE_BYTES"));
  assert.ok(!result.includes("file:///private/file"));
});

for (const fence of ["```", "~~~"]) {
  test(`an unclosed ${fence} fence stays inside its message when the snapshot renders`, () => {
    const result = conversationMarkdown(chain([
      message("user", `Before code\n\n${fence}js\nunfinished()\n\nstillCode()`, { attachment_ids: ["first-file"] }),
      message("assistant", "**Next reply**", { attachment_ids: ["next-file"] }),
    ]), "Haru", "https://kadan.example");
    const html = renderToStaticMarkup(createElement(ReactMarkdown, { children: result }));
    assert.match(html, /<h2>User · 2026-10-23T10:11:12Z<\/h2>/);
    assert.match(html, /<h2>Assistant · 2026-10-23T10:11:12Z<\/h2>/);
    assert.match(html, /<strong>Next reply<\/strong>/);
    assert.match(html, /<code class="language-js">unfinished\(\)\n\nstillCode\(\)\n<\/code>/);
    for (const id of ["first-file", "next-file"]) {
      assert.ok(html.includes(`<a href="https://kadan.example/api/agents/haru/assets/${id}/content?download=true">${id}</a>`));
    }
  });
}
