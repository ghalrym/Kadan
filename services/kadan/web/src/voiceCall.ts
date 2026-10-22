import { recognitionWav } from "./voiceInput.ts";

export function callUnavailable(): string | null {
  if (!window.isSecureContext) return "Calls require HTTPS or localhost.";
  if (!navigator.mediaDevices?.getUserMedia || typeof AudioWorkletNode === "undefined" || typeof AudioContext === "undefined")
    return "This browser does not support voice calls.";
  return null;
}

/** Energy VAD adapted from PR307: 300ms pre-roll, 60ms onset, 600ms silence. */
export class CallActivity {
  private frames: Float32Array[] = [];
  private active = false;
  private hot = 0;
  private quiet = 0;
  private duration = 0;
  readonly sampleRate: number;
  constructor(sampleRate: number) { this.sampleRate = sampleRate; }

  push(frame: Float32Array): { started: boolean; ended?: Float32Array } {
    const seconds = frame.length / this.sampleRate;
    const level = Math.sqrt(frame.reduce((sum, x) => sum + x * x, 0) / frame.length);
    const voiced = level >= 0.02;
    this.hot = voiced ? this.hot + seconds : 0;
    this.quiet = voiced ? 0 : this.quiet + seconds;
    this.frames.push(frame);
    let started = false;
    if (!this.active && this.hot >= 0.06) {
      this.active = true;
      this.duration = this.frames.reduce((sum, item) => sum + item.length, 0) / this.sampleRate;
      started = true;
    } else if (this.active) this.duration += seconds;
    if (!this.active) {
      while (this.frames.length > Math.ceil(0.3 / seconds)) this.frames.shift();
    } else if (this.quiet >= 0.6 || this.duration >= 20) {
      const ended = new Float32Array(this.frames.reduce((sum, item) => sum + item.length, 0));
      let offset = 0;
      for (const samples of this.frames) { ended.set(samples, offset); offset += samples.length; }
      this.frames = []; this.active = false; this.hot = 0; this.quiet = 0; this.duration = 0;
      return { started, ended };
    }
    return { started };
  }
}

/** Every decode, scheduled source and ended callback belongs to one turn. */
export class CallPlayback {
  private context: AudioContext;
  private generation = 0;
  private nextTime = 0;
  private sources = new Set<AudioBufferSourceNode>();
  private decoding: Promise<void> = Promise.resolve();
  private queued = 0;
  constructor(context: AudioContext) { this.context = context; }
  get playing() { return this.queued > 0; }

  push(encoded: string, ended: () => void): Promise<void> {
    const generation = this.generation;
    this.queued++;
    const pending = this.decoding.then(async () => {
      if (generation !== this.generation) return;
      const bytes = Uint8Array.from(atob(encoded), (value) => value.charCodeAt(0));
      const buffer = await this.context.decodeAudioData(bytes.buffer);
      if (generation !== this.generation) return;
      const source = this.context.createBufferSource();
      source.buffer = buffer;
      source.connect(this.context.destination);
      this.sources.add(source);
      source.onended = () => {
        this.sources.delete(source); source.disconnect();
        if (generation !== this.generation) return;
        this.queued--;
        ended();
      };
      const start = Math.max(this.context.currentTime + 0.02, this.nextTime);
      this.nextTime = start + buffer.duration;
      source.start(start);
    });
    this.decoding = pending;
    return pending;
  }

  stop(): void {
    this.generation++;
    for (const source of this.sources) { source.onended = null; source.stop(); source.disconnect(); }
    this.sources.clear(); this.nextTime = 0; this.queued = 0;
    this.decoding = Promise.resolve();
  }
}

export type CallState = { status: string; muted: boolean };

export class VoiceCallSession {
  private closed = false;
  private ready = false;
  private turn = 0;
  private muted = false;
  private phase = "Connecting…";
  private replyDone = false;
  private stream: MediaStream | null = null;
  private context: AudioContext | null = null;
  private capture: AudioWorkletNode | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private player: CallPlayback | null = null;
  private activity: CallActivity | null = null;
  private socket: WebSocket | null = null;
  private timer: ReturnType<typeof setTimeout> | undefined;
  private update: (state: CallState) => void;
  private done: (error?: string) => void;
  constructor(update: (state: CallState) => void, done: (error?: string) => void) {
    this.update = update; this.done = done;
  }
  private state(status: string) {
    this.phase = status;
    if (!this.closed) this.update({ status: this.muted ? "Muted" : status, muted: this.muted });
  }
  private send(event: object) {
    if (!this.closed && this.socket?.readyState === WebSocket.OPEN) this.socket.send(JSON.stringify(event));
  }

  async start(agentKey: string, personaId: string | null): Promise<void> {
    try {
      const unavailable = callUnavailable();
      if (unavailable) throw new Error(unavailable);
      this.state("Connecting…");
      // Create and resume inside the start gesture, before permission/network waits.
      this.context = new AudioContext();
      await this.context.resume();
      if (this.closed) return;
      const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true } });
      if (this.closed) { stream.getTracks().forEach((track) => track.stop()); return; }
      this.stream = stream;
      stream.getAudioTracks()[0].onended = () => this.end("The microphone disconnected. Start a new call.");
      this.activity = new CallActivity(this.context.sampleRate);
      this.player = new CallPlayback(this.context);
      await this.context.audioWorklet.addModule(new URL("./voiceCapture.worklet.js", import.meta.url));
      if (this.closed) return;
      this.capture = new AudioWorkletNode(this.context, "call-capture");
      this.capture.port.onmessage = (event: MessageEvent<Float32Array>) => this.frame(event.data);
      this.source = this.context.createMediaStreamSource(stream);
      this.source.connect(this.capture);
      this.capture.connect(this.context.destination);
      this.context.onstatechange = () => {
        if (!this.closed && this.context?.state !== "running") this.end("Audio was suspended. Start a new call.");
      };
      const url = new URL(`/api/agents/${encodeURIComponent(agentKey)}/calls`, window.location.href);
      url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
      this.socket = new WebSocket(url);
      this.socket.onopen = () => this.send({ type: "start", persona_id: personaId });
      this.socket.onmessage = (event) => this.message(event.data);
      this.socket.onerror = () => this.end("Could not connect the call. Try again.");
      this.socket.onclose = () => this.end("The call disconnected. Start a new call.");
      this.timer = setTimeout(() => this.end("Call connection timed out. Try again."), 20000);
    } catch (error) {
      this.end(error instanceof Error ? error.message : "Could not start the call.");
    }
  }

  private frame(samples: Float32Array) {
    if (this.closed || !this.ready || this.muted) return;
    const event = this.activity!.push(samples);
    if (event.started) {
      this.turn++;
      this.replyDone = false;
      this.player!.stop();
      this.send({ type: "interrupt", turn: this.turn });
      this.state("Listening…");
    }
    if (event.ended) {
      this.send({ type: "utterance", turn: this.turn, audio: recognitionWav(event.ended, this.context!.sampleRate) });
      this.state("Transcribing…");
    }
  }

  private message(data: string) {
    if (this.closed) return;
    try {
      const event = JSON.parse(data);
      if (event.type === "ready") { clearTimeout(this.timer); this.ready = true; this.state("Listening"); return; }
      if (event.type === "error") { this.end(event.message); return; }
      if (event.turn !== this.turn) return;
      if (event.type === "transcript" || event.type === "text") this.state(this.player?.playing ? "Speaking" : "Replying…");
      if (event.type === "audio") {
        const turn = this.turn;
        this.state("Speaking");
        void this.player!.push(event.audio, () => {
          if (this.closed || turn !== this.turn) return;
          this.send({ type: "played", turn, segment: event.segment });
          if (this.replyDone && !this.player!.playing) this.state("Listening");
        }).catch((error) => {
          if (!this.closed && turn === this.turn) this.end(`Audio playback failed: ${error instanceof Error ? error.message : "Start a new call."}`);
        });
      }
      if (event.type === "done") { this.replyDone = true; if (!this.player!.playing) this.state("Listening"); }
    } catch { this.end("The call returned an invalid message. Try again."); }
  }

  setMuted(muted: boolean) {
    this.muted = muted;
    this.stream?.getAudioTracks().forEach((track) => { track.enabled = !muted; });
    if (this.context) this.activity = new CallActivity(this.context.sampleRate);
    this.state(this.phase);
  }

  end(error?: string): void {
    if (this.closed) return;
    this.send({ type: "end" });
    this.closed = true; this.ready = false; this.turn++;
    clearTimeout(this.timer);
    this.player?.stop();
    if (this.capture) { this.capture.port.onmessage = null; this.capture.disconnect(); this.capture.port.close(); }
    this.source?.disconnect();
    this.stream?.getTracks().forEach((track) => { track.onended = null; track.stop(); });
    if (this.context) { this.context.onstatechange = null; void this.context.close().catch(() => {}); }
    this.socket?.close();
    this.done(error);
  }
}
