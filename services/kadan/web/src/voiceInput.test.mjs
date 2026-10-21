import assert from "node:assert/strict";
import { afterEach, beforeEach, test } from "node:test";
import { microphoneUnavailable, recognitionWav, VoiceInput } from "./voiceInput.ts";

let recorder, tracks, fetches, closedContexts, audio;
const originals = new Map(["window", "navigator", "MediaRecorder", "AudioContext", "fetch"].map(key => [key, Object.getOwnPropertyDescriptor(globalThis, key)]));
const install = (key, value) => Object.defineProperty(globalThis, key, { configurable: true, writable: true, value });
const flush = () => new Promise(resolve => setImmediate(resolve));

beforeEach(() => {
  tracks = [{ stopCount: 0, onended: null, stop() { this.stopCount++; } }];
  fetches = []; closedContexts = 0; audio = null;
  install("window", { isSecureContext: true });
  install("navigator", { mediaDevices: { getUserMedia: async () => ({ getTracks: () => tracks, getAudioTracks: () => tracks }) } });
  install("MediaRecorder", class {
    state = "inactive";
    constructor() { recorder = this; }
    start() { this.state = "recording"; }
    stop() {
      this.state = "inactive";
      queueMicrotask(() => { this.ondataavailable({ data: new Blob(["recorded audio"]) }); this.onstop(); });
    }
  });
  install("AudioContext", class {
    async decodeAudioData() { return { sampleRate: 48000, getChannelData: () => new Float32Array(48) }; }
    async close() { closedContexts++; }
  });
  install("fetch", async (path, options) => {
    fetches.push({ path, options });
    audio = JSON.parse(options.body).audio;
    return { ok: true, json: async () => ({ text: "  Hello agent  " }) };
  });
});
afterEach(() => {
  for (const [key, descriptor] of originals) {
    if (descriptor) Object.defineProperty(globalThis, key, descriptor);
    else delete globalThis[key];
  }
});

test("WAV transport is mono 16-kHz PCM with clipped samples and a twenty-second bound", () => {
  const bytes = Buffer.from(recognitionWav(new Float32Array([-2, 0, 2]), 16000).split(",")[1], "base64");
  assert.equal(bytes.toString("ascii", 0, 4), "RIFF");
  assert.equal(bytes.readUInt16LE(22), 1);
  assert.equal(bytes.readUInt32LE(24), 16000);
  assert.equal(bytes.readUInt16LE(34), 16);
  assert.deepEqual([bytes.readInt16LE(44), bytes.readInt16LE(46), bytes.readInt16LE(48)], [-32768, 0, 32767]);
  const bounded = Buffer.from(recognitionWav(new Float32Array(16000 * 21), 16000).split(",")[1], "base64");
  assert.equal(bounded.readUInt32LE(40), 16000 * 20 * 2);
});

test("finishing one recording transcribes once and returns text for review without sending a chat", async () => {
  const texts = [], states = [], done = [];
  const session = new VoiceInput(text => texts.push(text), state => states.push(state), error => done.push(error));
  await session.start();
  assert.equal(recorder.state, "recording");
  assert.deepEqual(fetches, []);
  session.finish();
  await flush();
  assert.deepEqual(texts, ["Hello agent"]);
  assert.deepEqual(states, ["requesting", "recording", "transcribing"]);
  assert.deepEqual(done, [undefined]);
  assert.equal(fetches.length, 1);
  assert.equal(fetches[0].path, "/api/transcriptions");
  assert.equal(Buffer.from(audio.split(",")[1], "base64").readUInt32LE(40), 16 * 2);
  assert.ok(tracks[0].stopCount > 0);
  assert.equal(closedContexts, 1);
});

test("cancelling before microphone permission returns releases the late stream", async () => {
  let permit;
  navigator.mediaDevices.getUserMedia = () => new Promise(resolve => { permit = resolve; });
  const texts = [], errors = [];
  const session = new VoiceInput(text => texts.push(text), () => {}, error => errors.push(error));
  const starting = session.start();
  session.cancel();
  permit({ getTracks: () => tracks, getAudioTracks: () => tracks });
  await starting;
  assert.equal(tracks[0].stopCount, 1);
  assert.deepEqual([texts, errors, fetches], [[], [], []]);
});

test("discarding an active recording releases the microphone and never transcribes", async () => {
  const session = new VoiceInput(() => assert.fail("Discarded text"), () => {}, () => assert.fail("Discarded result"));
  await session.start();
  session.cancel();
  await flush();
  assert.equal(recorder.state, "inactive");
  assert.equal(tracks[0].stopCount, 1);
  assert.deepEqual(fetches, []);
});

test("cancelling an in-flight transcription aborts it and discards a late result", async () => {
  let resolve, signal;
  install("fetch", (_path, options) => { signal = options.signal; return new Promise(done => { resolve = done; }); });
  const texts = [], errors = [];
  const session = new VoiceInput(text => texts.push(text), () => {}, error => errors.push(error));
  await session.start(); session.finish(); await flush();
  session.cancel();
  assert.equal(signal.aborted, true);
  resolve({ ok: true, json: async () => ({ text: "Late result" }) });
  await flush();
  assert.deepEqual([texts, errors], [[], []]);
  assert.equal(closedContexts, 1);
});

test("permission failures and empty recognition leave text unchanged and report an error", async () => {
  const errors = [];
  const denied = new VoiceInput(() => assert.fail("Permission denied"), () => {}, error => errors.push(error));
  navigator.mediaDevices.getUserMedia = async () => { throw new Error("Permission denied"); };
  await denied.start();
  assert.deepEqual(errors, ["Permission denied"]);
  assert.deepEqual(fetches, []);
  window.isSecureContext = false;
  assert.match(microphoneUnavailable(), /HTTPS or localhost/);
});

test("empty transcription is reported without adding draft text", async () => {
  install("fetch", async () => ({ ok: true, json: async () => ({ text: " " }) }));
  const errors = [];
  const session = new VoiceInput(() => assert.fail("Empty transcript"), () => {}, error => errors.push(error));
  await session.start(); session.finish(); await flush();
  assert.deepEqual(errors, ["No speech was recognized. Try again."]);
  assert.equal(closedContexts, 1);
});
