import assert from "node:assert/strict";
import { test } from "node:test";
import { ComputerControl } from "./computerControl.ts";

class Keyboard {
  held = new Map();
  grabbed = false;
  constructor(onkeyevent) { this.onkeyevent = onkeyevent; }
  key(keysym, code, down) {
    if (!this.grabbed) return;
    if (down) this.held.set(code, keysym);
    else this.held.delete(code);
    this.onkeyevent(keysym, code, down);
  }
  ungrab() {
    this.grabbed = false;
    for (const [code, keysym] of this.held) this.onkeyevent(keysym, code, false);
    this.held.clear();
  }
}

// Only the verified noVNC 1.7.0 ordering and public key dispatch; no browser or transport.
class Client extends EventTarget {
  state = "connecting";
  watching = false;
  events = [];
  keyboard = new Keyboard(this.handleKeyEvent.bind(this));
  get viewOnly() { return this.watching; }
  set viewOnly(value) {
    this.watching = value;
    this.events.push(["viewOnly", value]);
    if (this.state === "connecting" || this.state === "connected") {
      if (value) this.keyboard.ungrab();
      else this.keyboard.grabbed = true;
    }
  }
  handleKeyEvent(keysym, code, down) { this.sendKey(keysym, code, down); }
  sendKey(keysym, code, down) {
    if (this.state !== "connected" || this.viewOnly) return;
    if (down === undefined) {
      this.sendKey(keysym, code, true);
      this.sendKey(keysym, code, false);
      return;
    }
    this.events.push(["key", keysym, code, down]);
  }
  connect() { this.state = "connected"; this.dispatchEvent(new Event("connect")); }
  focus() { this.events.push(["focus"]); }
  disconnect() {
    this.events.push(["disconnect"]);
    this.state = "disconnecting";
    this.keyboard.ungrab();
    this.state = "disconnected";
    this.dispatchEvent(new Event("disconnect"));
  }
}

function controlling() {
  const client = new Client();
  const control = new ComputerControl(client);
  client.connect();
  control.setControl(true);
  client.events = [];
  return { client, control };
}

test("baseline: viewOnly drops ungrab's ordinary-key release without a window blur", (t) => {
  const client = new Client();
  client.connect();
  client.viewOnly = false;
  client.events = [];
  client.keyboard.key(97, "KeyA", true);
  client.viewOnly = true;
  assert.deepEqual(client.events, [["key", 97, "KeyA", true], ["viewOnly", true]]);
  assert.equal(client.keyboard.held.size, 0);
  t.diagnostic(`before: ${JSON.stringify(client.events)}`);
});

test("Stop controlling releases the owned ordinary key before viewOnly ungrabs", (t) => {
  const { client, control } = controlling();
  client.keyboard.key(97, "KeyA", true);
  control.setControl(false);
  assert.deepEqual(client.events, [
    ["key", 97, "KeyA", true], ["key", 97, "KeyA", false], ["viewOnly", true],
  ]);
  assert.equal(client.keyboard.held.size, 0);
  t.diagnostic(`after: ${JSON.stringify(client.events)}`);
});

test("normal close/unmount releases before Watching and disconnecting", () => {
  const { client, control } = controlling();
  client.keyboard.key(97, "KeyA", true);
  control.disconnect();
  assert.deepEqual(client.events, [
    ["key", 97, "KeyA", true], ["key", 97, "KeyA", false],
    ["viewOnly", true], ["disconnect"],
  ]);
});

test("presses, repeats and natural releases pass through without extra cleanup releases", () => {
  const { client, control } = controlling();
  client.keyboard.key(97, "KeyA", true);
  client.keyboard.key(97, "KeyA", true);
  client.keyboard.key(97, "KeyA", false);
  control.setControl(false);
  assert.deepEqual(client.events, [
    ["key", 97, "KeyA", true], ["key", 97, "KeyA", true],
    ["key", 97, "KeyA", false], ["viewOnly", true],
  ]);
});

test("repeated downs require only one cleanup release per held key", () => {
  const { client, control } = controlling();
  client.keyboard.key(97, "KeyA", true);
  client.keyboard.key(97, "KeyA", true);
  client.keyboard.key(98, "KeyB", true);
  client.events = [];
  control.setControl(false);
  control.setControl(false);
  assert.deepEqual(client.events, [
    ["key", 97, "KeyA", false], ["key", 98, "KeyB", false],
    ["viewOnly", true], ["viewOnly", true],
  ]);
});

test("omitted-down pulses preserve recursive public dispatch and leave no held key", () => {
  const { client, control } = controlling();
  client.sendKey(97, "KeyA");
  client.sendKey(98, "KeyB", true);
  client.sendKey(98, "KeyB");
  control.setControl(false);
  assert.deepEqual(client.events, [
    ["key", 97, "KeyA", true], ["key", 97, "KeyA", false],
    ["key", 98, "KeyB", true], ["key", 98, "KeyB", true],
    ["key", 98, "KeyB", false], ["viewOnly", true],
  ]);
});

test("keys with unknown physical codes are tracked separately by keysym", () => {
  const { client, control } = controlling();
  client.sendKey(97, null, true);
  client.sendKey(98, null, true);
  client.events = [];
  control.setControl(false);
  assert.deepEqual(client.events, [
    ["key", 97, null, false], ["key", 98, null, false], ["viewOnly", true],
  ]);
});

test("new views start Watching", () => {
  const client = new Client();
  new ComputerControl(client);
  client.connect();
  assert.equal(client.viewOnly, true);
  assert.equal(client.keyboard.grabbed, false);
});

test("calls rejected before connection are not owned after connecting", () => {
  const client = new Client();
  const control = new ComputerControl(client);
  control.setControl(true);
  client.sendKey(97, "KeyA", true);
  client.sendKey(98, "KeyB");
  client.connect();
  client.events = [];
  control.setControl(false);
  assert.deepEqual(client.events, [["viewOnly", true]]);
});

test("calls rejected while Watching are not owned on taking control", () => {
  const { client, control } = controlling();
  control.setControl(false);
  client.sendKey(97, "KeyA");
  client.sendKey(98, "KeyB", true);
  control.setControl(true);
  client.events = [];
  control.setControl(false);
  assert.deepEqual(client.events, [["viewOnly", true]]);
});

test("an unexpected disconnect discards ownership without claiming remote cleanup", () => {
  const { client, control } = controlling();
  client.keyboard.key(97, "KeyA", true);
  client.disconnect();
  client.events = [];
  control.disconnect();
  assert.deepEqual(client.events, [["viewOnly", true], ["disconnect"]]);
});

test("stopping one viewer releases only its own keys", () => {
  const first = controlling();
  const second = controlling();
  first.client.keyboard.key(97, "KeyA", true);
  second.client.keyboard.key(98, "KeyB", true);
  first.control.setControl(false);
  assert.deepEqual(first.client.events, [
    ["key", 97, "KeyA", true], ["key", 97, "KeyA", false], ["viewOnly", true],
  ]);
  assert.deepEqual(second.client.events, [["key", 98, "KeyB", true]]);
});
