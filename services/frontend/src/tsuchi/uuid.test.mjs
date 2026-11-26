import assert from "node:assert/strict";
import { test } from "node:test";
import { randomUuid } from "./uuid.ts";

test("creates UUIDv4 without the secure-context randomUUID API", (context) => {
  const original = Object.getOwnPropertyDescriptor(globalThis, "crypto");
  context.after(() => Object.defineProperty(globalThis, "crypto", original));
  Object.defineProperty(globalThis, "crypto", {
    configurable: true,
    value: { getRandomValues: (bytes) => bytes.fill(255) },
  });
  assert.equal(randomUuid(), "ffffffff-ffff-4fff-bfff-ffffffffffff");
});
