import assert from "node:assert/strict";
import { afterEach, beforeEach, test } from "node:test";
import { CallActivity, CallPlayback, VoiceCallSession, callUnavailable } from "./voiceCall.ts";

const flush = () => new Promise(resolve => setImmediate(resolve));
const originals = new Map(["window", "navigator", "AudioWorkletNode", "AudioContext", "WebSocket"].map(key => [key, Object.getOwnPropertyDescriptor(globalThis, key)]));
const install = (key, value) => Object.defineProperty(globalThis, key, { configurable: true, writable: true, value });
let track, capture, capturedTracks, context, socket, sources, states, errors;
function fakeContext() {
  return {
    state: "running", sampleRate: 16000, currentTime: 0, destination: {}, closed: false,
    audioWorklet: { addModule: async () => {} },
    resume: async () => {}, close() { this.closed = true; return Promise.resolve(); },
    createMediaStreamSource: stream => ({ connect() { capturedTracks = stream.getAudioTracks().map(track => track.enabled); }, disconnect() {} }),
    decodeAudioData: async () => ({ duration: 0.1 }),
    createBufferSource() {
      const source = { buffer: null, starts: [], stops: 0, onended: null, connect() {}, disconnect() {}, start(time) { this.starts.push(time); }, stop() { this.stops++; } };
      sources.push(source);
      return source;
    },
  };
}
function sample(value = 0.1) { return new Float32Array(320).fill(value); }
function frame(value) { capture.port.onmessage?.({ data: sample(value) }); }
function speech() { for (let i = 0; i < 5; i++) frame(0.1); }
function silence() { for (let i = 0; i < 31; i++) frame(0); }
function message(event) { socket.onmessage({ data: JSON.stringify(event) }); }
function session() { return new VoiceCallSession(state => states.push(state), error => errors.push(error)); }
async function connected() {
  const call = session();
  await call.start("test", "selected-persona");
  socket.readyState = 1;
  socket.onopen();
  message({ type: "ready" });
  return call;
}

beforeEach(() => {
  sources = []; states = []; errors = []; capturedTracks = [];
  track = { enabled: true, stopped: false, onended: null, stop() { this.stopped = true; } };
  install("window", { isSecureContext: true, location: { href: "https://kadan.local/" } });
  install("navigator", { mediaDevices: { getUserMedia: async () => ({ getTracks: () => [track], getAudioTracks: () => [track] }) } });
  install("AudioContext", function () { context = fakeContext(); return context; });
  install("AudioWorkletNode", class {
    port = { onmessage: null, close() {} };
    constructor() { capture = this; }
    connect() {} disconnect() {}
  });
  install("WebSocket", class {
    static OPEN = 1;
    readyState = 0; sent = []; closed = false;
    constructor(url) { this.url = String(url); socket = this; }
    send(data) { this.sent.push(JSON.parse(data)); }
    close() { this.closed = true; }
  });
});
afterEach(() => {
  for (const [key, descriptor] of originals) {
    if (descriptor) Object.defineProperty(globalThis, key, descriptor);
    else delete globalThis[key];
  }
});

test("VAD keeps pre-roll, detects onset once and ends at silence with bounded recording", () => {
  const vad = new CallActivity(16000);
  for (let i = 0; i < 100; i++) assert.equal(vad.push(sample(0)).ended, undefined);
  let starts = 0, utterance;
  for (let i = 0; i < 5; i++) starts += Number(vad.push(sample()).started);
  for (let i = 0; i < 31; i++) utterance = vad.push(sample(0)).ended ?? utterance;
  assert.equal(starts, 1);
  assert.ok(utterance.length <= 16000);
  const loud = new CallActivity(16000);
  let ended;
  for (let i = 0; i < 1002; i++) ended = loud.push(sample()).ended ?? ended;
  assert.ok(ended.length <= 320320);
});

test("stopping during audio decode prevents a stale source from ever starting", async () => {
  const audio = fakeContext();
  let resolve;
  audio.decodeAudioData = () => new Promise(done => { resolve = done; });
  const player = new CallPlayback(audio);
  const pushing = player.push("AAAA", () => assert.fail("Stale acknowledgement"));
  await flush();
  player.stop();
  resolve({ duration: 1 });
  await pushing;
  assert.equal(sources.length, 0);
  assert.equal(player.playing, false);
});

test("stopping playback cancels all scheduled segments and invalidates queued ended callbacks", async () => {
  const player = new CallPlayback(fakeContext());
  let acknowledgements = 0;
  await player.push("AAAA", () => acknowledgements++);
  await player.push("AAAA", () => acknowledgements++);
  const staleEnded = sources[0].onended;
  assert.ok(sources[1].starts[0] > sources[0].starts[0]);
  player.stop();
  staleEnded();
  assert.deepEqual(sources.map(source => source.stops), [1, 1]);
  assert.equal(acknowledgements, 0);
});

test("capture continues during playback and onset cancels old audio before sending new speech", async () => {
  const call = await connected();
  assert.equal(socket.url, "wss://kadan.local/api/agents/test/calls");
  assert.deepEqual(socket.sent[0], { type: "start", persona_id: "selected-persona" });
  speech(); silence();
  assert.deepEqual(socket.sent.map(event => event.type), ["start", "interrupt", "utterance"]);
  assert.equal(socket.sent[2].turn, 1);
  message({ type: "audio", turn: 1, segment: 0, audio: "AAAA" });
  await flush();
  assert.equal(sources.length, 1);
  speech();
  assert.equal(sources[0].stops, 1);
  assert.equal(socket.sent.at(-1).type, "interrupt");
  assert.equal(socket.sent.at(-1).turn, 2);
  message({ type: "audio", turn: 1, segment: 1, audio: "AAAA" });
  await flush();
  assert.equal(sources.length, 1);
  call.end();
  assert.equal(track.stopped, true);
  assert.equal(context.closed, true);
  assert.equal(socket.closed, true);
});

test("mute discards a partial utterance and keeps received playback usable", async () => {
  const call = await connected();
  speech();
  call.setMuted(true);
  assert.equal(track.enabled, false);
  silence();
  assert.equal(socket.sent.filter(event => event.type === "utterance").length, 0);
  message({ type: "audio", turn: 1, segment: 0, audio: "AAAA" });
  await flush();
  assert.equal(sources.length, 1);
  call.setMuted(false);
  assert.equal(track.enabled, true);
  speech(); silence();
  assert.equal(socket.sent.at(-1).type, "utterance");
  assert.equal(socket.sent.at(-1).turn, 2);
  call.end();
});

for (const { name, mutes, enabled } of [
  { name: "muting before microphone permission disables arriving tracks before capture connects", mutes: [true], enabled: false },
  { name: "unmuted startup keeps arriving tracks enabled when capture connects", mutes: [], enabled: true },
  { name: "muting then unmuting before microphone permission keeps arriving tracks enabled", mutes: [true, false], enabled: true },
]) {
  test(name, async () => {
    let permit;
    navigator.mediaDevices.getUserMedia = () => new Promise(done => { permit = done; });
    const tracks = [track, { ...track }];
    const call = session();
    const starting = call.start("Hi");
    await flush();
    assert.deepEqual(states, [{ status: "Connecting…", muted: false }]);
    for (const muted of mutes) call.setMuted(muted);
    permit({ getTracks: () => tracks, getAudioTracks: () => tracks });
    await starting;
    try {
      assert.deepEqual(capturedTracks, [enabled, enabled]);
      assert.deepEqual(tracks.map(track => track.enabled), [enabled, enabled]);
      assert.deepEqual(states, [
        { status: "Connecting…", muted: false },
        ...mutes.map(muted => ({ status: muted ? "Muted" : "Connecting…", muted })),
      ]);
      assert.deepEqual(errors, []);
    } finally { call.end(); }
  });
}

test("only completed playback is acknowledged; done waits for queued audio", async () => {
  const call = await connected();
  speech(); silence();
  message({ type: "audio", turn: 1, segment: 0, audio: "AAAA" });
  message({ type: "done", turn: 1 });
  assert.equal(states.at(-1).status, "Speaking");
  assert.equal(socket.sent.filter(event => event.type === "played").length, 0);
  await flush();
  sources[0].onended();
  assert.deepEqual(socket.sent.at(-1), { type: "played", turn: 1, segment: 0 });
  assert.equal(states.at(-1).status, "Listening");
  call.end();
});

test("ending before permission resolves releases a late microphone without opening a call", async () => {
  let permit;
  navigator.mediaDevices.getUserMedia = () => new Promise(done => { permit = done; });
  socket = null;
  const call = session();
  const starting = call.start("test", null);
  await flush();
  call.end();
  permit({ getTracks: () => [track], getAudioTracks: () => [track] });
  await starting;
  assert.equal(track.stopped, true);
  assert.equal(socket, null);
  assert.equal(errors.length, 1);
});

test("ending before worklet loading resolves cannot attach capture afterwards", async () => {
  let finishModule;
  install("AudioContext", function () { context = fakeContext(); context.audioWorklet.addModule = () => new Promise(done => { finishModule = done; }); return context; });
  capture = null;
  const call = session();
  const starting = call.start("test", null);
  await flush();
  call.end();
  finishModule();
  await starting;
  assert.equal(capture, null);
  assert.equal(track.stopped, true);
});

test("upstream errors release audio and mic and ignore late messages", async () => {
  const call = await connected();
  const receive = socket.onmessage;
  message({ type: "error", message: "Install the tier's speech model." });
  receive({ data: JSON.stringify({ type: "audio", turn: 0, audio: "AAAA" }) });
  await flush();
  assert.equal(errors[0], "Install the tier's speech model.");
  assert.equal(sources.length, 0);
  assert.equal(track.stopped, true);
  assert.equal(context.closed, true);
  call.end();
  assert.equal(errors.length, 1);
});

test("browser-trusted HTTP origins can start calls over an unencrypted WebSocket", async () => {
  window.location.href = "http://kadan.local/";
  assert.equal(callUnavailable(), null);
  const call = await connected();
  assert.equal(socket.url, "ws://kadan.local/api/agents/test/calls");
  call.end();
});

test("untrusted origins and denied permissions return useful errors", async () => {
  window.isSecureContext = false;
  assert.match(callUnavailable(), /HTTPS or localhost/);
  window.isSecureContext = true;
  navigator.mediaDevices.getUserMedia = async () => { throw new Error("Permission denied"); };
  await session().start("test", null);
  assert.equal(errors[0], "Permission denied");
  assert.equal(context.closed, true);
});


test("starting without an additional persona sends an explicit null selection", async () => {
  const call = session();
  await call.start("test", null);
  socket.readyState = 1;
  socket.onopen();
  assert.deepEqual(socket.sent[0], { type: "start", persona_id: null });
  call.end();
});


test("LFM3 call selection is explicit and uses the same session transport", async () => {
  const call = session();
  await call.start("test", null, "lfm3");
  socket.readyState = WebSocket.OPEN;
  socket.onopen();
  assert.deepEqual(socket.sent[0], { type: "start", persona_id: null, backend: "lfm3" });
  call.end();
});
