import { useEffect, useRef, useState } from "react";
import { Mic, MicOff, Phone, PhoneOff } from "lucide-react";
import { callUnavailable, VoiceCallSession, type CallState } from "./voiceCall";

export default function VoiceCallControl({ instruction, disabled, onActive }: {
  instruction: string;
  disabled: boolean;
  onActive: (active: boolean) => void;
}) {
  const session = useRef<VoiceCallSession | null>(null);
  const [call, setCall] = useState<CallState | null>(null);
  const [error, setError] = useState("");
  useEffect(() => () => session.current?.end(), []);
  useEffect(() => { if (disabled) session.current?.end(); }, [disabled]);
  const unavailable = callUnavailable();

  function start() {
    setError("");
    const next = new VoiceCallSession(setCall, (message) => {
      session.current = null; setCall(null); onActive(false); setError(message ?? "");
    });
    session.current = next;
    onActive(true);
    void next.start(instruction);
  }

  return <>
    {error && <p role="alert">{error}</p>}
    <div className="call-controls">
      {call ? <>
        <span role="status"><Phone size={14} />{call.status}</span>
        <button type="button" className="chip" aria-label={call.muted ? "Unmute microphone" : "Mute microphone"} aria-pressed={call.muted} onClick={() => session.current?.setMuted(!call.muted)}>
          {call.muted ? <MicOff size={14} /> : <Mic size={14} />}{call.muted ? "Unmute" : "Mute"}
        </button>
        <button type="button" className="chip end-call" onClick={() => session.current?.end()}><PhoneOff size={14} />End call</button>
      </> : <button type="button" className="chip" disabled={disabled || !!unavailable} title={unavailable ?? "Start a voice call"} onClick={start}><Phone size={14} />Start call</button>}
    </div>
  </>;
}
