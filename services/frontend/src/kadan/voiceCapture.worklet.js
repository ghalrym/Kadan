// Keep capture running during playback. Silent output prevents mic monitoring.
class CallCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.frame = new Float32Array(Math.round(sampleRate * 0.02));
    this.offset = 0;
  }
  process(inputs) {
    const input = inputs[0]?.[0];
    if (input) for (const sample of input) {
      this.frame[this.offset++] = sample;
      if (this.offset === this.frame.length) {
        this.port.postMessage(this.frame, [this.frame.buffer]);
        this.frame = new Float32Array(Math.round(sampleRate * 0.02));
        this.offset = 0;
      }
    }
    return true;
  }
}
registerProcessor("call-capture", CallCapture);
