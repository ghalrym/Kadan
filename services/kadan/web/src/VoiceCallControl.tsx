import { useEffect, useRef, useState } from "react";
import { Mic, MicOff, Phone, PhoneOff } from "lucide-react";
import { callUnavailable, VoiceCallSession, type CallState } from "./voiceCall";

export type VoiceCall = ReturnType<typeof useVoiceCall>;

/** Owns the call session so it outlives the agent card that starts it. */
export function useVoiceCall(instruction: string, disabled: boolean) {
  const session = useRef<VoiceCallSession | null>(null);
  const [call, setCall] = useState<CallState | null>(null);
  const [active, setActive] = useState(false);
  const [error, setError] = useState("");
  useEffect(() => () => session.current?.end(), []);
  useEffect(() => { if (disabled) session.current?.end(); }, [disabled]);

  function start() {
    if (disabled || session.current) return;
    setError("");
    const next = new VoiceCallSession(setCall, (message) => {
      session.current = null; setCall(null); setActive(false); setError(message ?? "");
    });
    session.current = next;
    setActive(true);
    void next.start(instruction);
  }

  return {
    call,
    active,
    error,
    unavailable: callUnavailable(),
    start,
    end: () => session.current?.end(),
    setMuted: (muted: boolean) => session.current?.setMuted(muted),
  };
}

export default function VoiceCallControl({ voiceCall }: { voiceCall: VoiceCall }) {
  const { call, error } = voiceCall;
  return <>
    {error && <p role="alert">{error}</p>}
    {call && <div className="call-controls">
      <span role="status"><Phone size={14} />{call.status}</span>
      <button type="button" className="chip" aria-label={call.muted ? "Unmute microphone" : "Mute microphone"} aria-pressed={call.muted} onClick={() => voiceCall.setMuted(!call.muted)}>
        {call.muted ? <MicOff size={14} /> : <Mic size={14} />}{call.muted ? "Unmute" : "Mute"}
      </button>
      <button type="button" className="chip end-call" onClick={voiceCall.end}><PhoneOff size={14} />End call</button>
    </div>}
  </>;
}
