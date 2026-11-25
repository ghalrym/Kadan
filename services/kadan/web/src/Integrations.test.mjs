import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import ts from "typescript";
import { runInNewContext } from "node:vm";

function loadHelpers() {
  const source = ts.transpileModule(
    readFileSync(new URL("./integrations.ts", import.meta.url), "utf8"),
    { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 } },
  ).outputText;
  const exports = {};
  runInNewContext(source, { exports });
  return exports;
}

test("statusLabel reflects paused, connected, error, and disconnected states", () => {
  const { statusLabel } = loadHelpers();
  assert.equal(statusLabel({ enabled: false, status: "connected" }), "Paused");
  assert.equal(statusLabel({ enabled: true, status: "connected" }), "Connected");
  assert.equal(statusLabel({ enabled: true, status: "error" }), "Needs attention");
  assert.equal(statusLabel({ enabled: true, status: "disconnected" }), "Not connected");
});

test("toggleGroupTools pins only discovered tools and cleanly unpins a group", () => {
  const { toggleGroupTools } = loadHelpers();
  const available = [
    { name: "t3_thread_launch" },
    { name: "pull_request_list" },
    { name: "ci_logs" },
  ];
  const enabled = toggleGroupTools(
    ["t3_thread_launch"],
    available,
    ["pull_request_list", "pull_request_read", "ci_logs"],
    true,
  );
  assert.deepEqual(enabled, ["t3_thread_launch", "pull_request_list", "ci_logs"]);

  const disabled = toggleGroupTools(
    enabled,
    available,
    ["pull_request_list", "pull_request_read", "ci_logs"],
    false,
  );
  assert.deepEqual(disabled, ["t3_thread_launch"]);
});
