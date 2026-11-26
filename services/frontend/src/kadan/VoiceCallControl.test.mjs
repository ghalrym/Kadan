import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { registerHooks } from "node:module";
import { afterEach, beforeEach, test } from "node:test";
import ts from "typescript";

// Run the hook with local state and no mounted component or browser.
const reactUrl = `data:text/javascript,${encodeURIComponent(`
  export const state = [];
  export const useEffect = () => {};
  export const useRef = current => ({ current });
  export function useState(value) {
    const index = state.push(value) - 1;
    return [value, next => { state[index] = next; }];
  }
`)}`;
const controlUrl = new URL("./VoiceCallControl.tsx", import.meta.url).href;
const loader = registerHooks({
  resolve(specifier, context, nextResolve) {
    if (context.parentURL === controlUrl) {
      if (specifier === "react") return { url: reactUrl, shortCircuit: true };
      if (specifier === "./voiceCall") return { url: new URL("./voiceCall.ts", import.meta.url).href, shortCircuit: true };
    }
    return nextResolve(specifier, context);
  },
  load(url, context, nextLoad) {
    if (url !== controlUrl) return nextLoad(url, context);
    const { outputText } = ts.transpileModule(readFileSync(new URL(url), "utf8"), {
      compilerOptions: { module: ts.ModuleKind.ESNext, jsx: ts.JsxEmit.ReactJSX },
    });
    return { format: "module", source: outputText, shortCircuit: true };
  },
});
const { state } = await import(reactUrl);
const { useVoiceCall } = await import(controlUrl);
loader.deregister();

const originals = new Map(["window", "navigator"].map(key => [key, Object.getOwnPropertyDescriptor(globalThis, key)]));
beforeEach(() => {
  state.length = 0;
  Object.defineProperty(globalThis, "window", { configurable: true, value: { isSecureContext: true } });
  Object.defineProperty(globalThis, "navigator", { configurable: true, value: { mediaDevices: {} } });
});
afterEach(() => {
  for (const [key, descriptor] of originals) {
    if (descriptor) Object.defineProperty(globalThis, key, descriptor);
    else delete globalThis[key];
  }
});

test("starting an unavailable call reports the session error for the composer alert", () => {
  const voiceCall = useVoiceCall("test", null, false);
  assert.equal(voiceCall.unavailable, "This browser does not support voice calls.");
  voiceCall.start();
  assert.deepEqual(state, [null, false, voiceCall.unavailable]);
});

test("a disabled call does not start a session or report an availability error", () => {
  useVoiceCall("test", null, true).start();
  assert.deepEqual(state, [null, false, ""]);
});
