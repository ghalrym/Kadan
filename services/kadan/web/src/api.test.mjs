import assert from "node:assert/strict";
import { afterEach, test } from "node:test";
import { api, ApiError } from "./api.ts";

const originalFetch = globalThis.fetch;
afterEach(() => { globalThis.fetch = originalFetch; });

test("successful assignment with no content does not attempt to parse JSON", async () => {
  let request;
  globalThis.fetch = async (...args) => {
    request = args;
    return { ok: true, status: 204, json: () => { throw new Error("No JSON body"); } };
  };
  const assignment = { runner_id: "runner-2", expected_runner_id: "runner-1" };
  assert.equal(await api("/agents/ember/runner", assignment), undefined);
  assert.equal(request[0], "/api/agents/ember/runner");
  assert.equal(request[1].method, "POST");
  assert.deepEqual(JSON.parse(request[1].body), assignment);
});

test("JSON responses and assignment conflict errors retain their existing behavior", async () => {
  const rows = [{ runner_id: "runner-1" }];
  globalThis.fetch = async () => ({ ok: true, status: 200, json: async () => rows });
  assert.equal(await api("/runners"), rows);
  globalThis.fetch = async () => ({ ok: false, status: 409, json: async () => ({ detail: "Agent assignment changed" }) });
  await assert.rejects(api("/agents/ember/runner", {}), error => error instanceof ApiError
    && error.status === 409 && error.message === "Agent assignment changed");
});
