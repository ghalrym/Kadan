export type VoiceState = "requesting" | "recording" | "transcribing";

/** Reuse PR307's bounded 16-kHz WAV transport, without its voice-call session. */
export function recognitionWav(samples: Float32Array, sampleRate: number): string {
  const ratio = sampleRate / 16000;
  const count = Math.min(320000, Math.floor(samples.length / ratio));
  const bytes = new Uint8Array(44 + count * 2);
  const view = new DataView(bytes.buffer);
  const word = (offset: number, text: string) => {
    for (let i = 0; i < text.length; i++) bytes[offset + i] = text.charCodeAt(i);
  };
  word(0, "RIFF"); view.setUint32(4, bytes.length - 8, true); word(8, "WAVEfmt ");
  view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
  view.setUint32(24, 16000, true); view.setUint32(28, 32000, true);
  view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  word(36, "data"); view.setUint32(40, count * 2, true);
  for (let i = 0; i < count; i++) {
    const start = i * ratio, end = (i + 1) * ratio;
    let value = 0;
    for (let j = Math.floor(start); j < Math.ceil(end); j++)
      value += samples[j] * (Math.min(end, j + 1) - Math.max(start, j));
    value = Math.max(-1, Math.min(1, value / ratio));
    view.setInt16(44 + i * 2, Math.round(value * (value < 0 ? 32768 : 32767)), true);
  }
  let binary = "";
  for (let i = 0; i < bytes.length; i += 8192)
    binary += String.fromCharCode(...bytes.subarray(i, i + 8192));
  return `data:audio/wav;base64,${btoa(binary)}`;
}

export function microphoneUnavailable(): string | null {
  if (!window.isSecureContext) return "Microphone input requires HTTPS or localhost.";
  if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined" || typeof AudioContext === "undefined")
    return "This browser does not support microphone recording.";
  return null;
}

/** One recording owns its microphone and request until completion or cancellation. */
export class VoiceInput {
  private stream: MediaStream | null = null;
  private recorder: MediaRecorder | null = null;
  private context: AudioContext | null = null;
  private timer: ReturnType<typeof setTimeout> | undefined;
  private controller = new AbortController();
  private closed = false;
  private chunks: Blob[] = [];
  private onText: (text: string) => void;
  private onState: (state: VoiceState) => void;
  private onDone: (error?: string) => void;

  constructor(onText: (text: string) => void, onState: (state: VoiceState) => void, onDone: (error?: string) => void) {
    this.onText = onText;
    this.onState = onState;
    this.onDone = onDone;
  }

  async start(): Promise<void> {
    try {
      const unavailable = microphoneUnavailable();
      if (unavailable) throw new Error(unavailable);
      this.onState("requesting");
      const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true } });
      if (this.closed) { stream.getTracks().forEach((track) => track.stop()); return; }
      this.stream = stream;
      this.recorder = new MediaRecorder(stream);
      this.recorder.ondataavailable = (event) => { if (!this.closed && event.data.size) this.chunks.push(event.data); };
      this.recorder.onerror = () => this.done("Microphone recording failed. Try again.");
      this.recorder.onstop = () => { if (!this.closed) void this.transcribe(); };
      stream.getAudioTracks()[0].onended = () => this.done("The microphone disconnected. Try again.");
      this.recorder.start();
      this.onState("recording");
      this.timer = setTimeout(() => this.finish(), 20000);
    } catch (error) {
      this.done(error instanceof Error ? error.message : "Could not start the microphone.");
    }
  }

  finish(): void {
    if (this.recorder?.state === "recording") this.recorder.stop();
  }

  cancel(): void {
    this.closed = true;
    clearTimeout(this.timer);
    this.controller.abort();
    if (this.recorder?.state === "recording") this.recorder.stop();
    this.stream?.getTracks().forEach((track) => { track.onended = null; track.stop(); });
    if (this.context) void this.context.close().catch(() => {});
  }

  private done(error?: string): void {
    if (this.closed) return;
    this.cancel();
    this.onDone(error);
  }

  private async transcribe(): Promise<void> {
    clearTimeout(this.timer);
    this.stream?.getTracks().forEach((track) => { track.onended = null; track.stop(); });
    this.onState("transcribing");
    try {
      this.context = new AudioContext();
      const recording = await new Blob(this.chunks).arrayBuffer();
      if (this.closed) return;
      const audio = await this.context.decodeAudioData(recording);
      if (this.closed) return;
      const response = await fetch("/api/transcriptions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ audio: recognitionWav(audio.getChannelData(0), audio.sampleRate) }),
        signal: this.controller.signal,
      });
      const result = await response.json();
      if (this.closed) return;
      if (!response.ok) throw new Error(typeof result.detail === "string" ? result.detail : "Transcription failed. Try again.");
      if (typeof result.text !== "string" || !result.text.trim()) throw new Error("No speech was recognized. Try again.");
      this.onText(result.text.trim());
      this.done();
    } catch (error) {
      this.done(error instanceof Error ? error.message : "Transcription failed. Try again.");
    }
  }
}
