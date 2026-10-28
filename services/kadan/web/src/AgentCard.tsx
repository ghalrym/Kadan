import { useEffect, useRef, useState } from "react";
import { Monitor, Phone } from "lucide-react";
import Pet from "./Pet";
import { api } from "./api";
import type { PetSheet, PetState } from "./pet";
import type { ComputerStatus } from "./types";

const COMPUTER_STATES: Record<ComputerStatus["state"], string> = {
  ready: "Ready",
  unavailable: "Unavailable",
  not_configured: "Not set up",
};
/** The thumbnail is a still; the live view is one click away. */
const REFRESH_MS = 5000;

interface Props {
  agentName: string;
  sheet: PetSheet;
  mood: PetState;
  presence: string;
  onClose: () => void;
  callDisabled: boolean;
  callTitle: string;
  onCall: () => void;
  onOpenComputer: () => void;
}

/** The agent's contact card, opened from its pet: presence, calling and its computer. */
export default function AgentCard({ agentName, sheet, mood, presence, onClose, callDisabled, callTitle, onCall, onOpenComputer }: Props) {
  const card = useRef<HTMLDivElement>(null);
  const [computer, setComputer] = useState<ComputerStatus | null>(null);
  const [frame, setFrame] = useState(0);
  const [thumbnail, setThumbnail] = useState(false);

  useEffect(() => {
    card.current?.focus();
    const outside = (event: PointerEvent) => {
      const target = event.target as Node;
      // The pet button toggles the card itself.
      if (!card.current?.parentElement?.contains(target)) onClose();
    };
    const escape = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    document.addEventListener("pointerdown", outside);
    document.addEventListener("keydown", escape);
    return () => {
      document.removeEventListener("pointerdown", outside);
      document.removeEventListener("keydown", escape);
    };
  }, [onClose]);
  useEffect(() => {
    let closed = false;
    async function refresh() {
      try {
        const value = await api<ComputerStatus>("/computer");
        if (!closed) {
          setComputer(value);
          setFrame((count) => count + 1);
        }
      } catch (error) {
        if (!closed) setComputer({ state: "unavailable", message: (error as Error).message });
      }
    }
    void refresh();
    const timer = setInterval(() => void refresh(), REFRESH_MS);
    return () => { closed = true; clearInterval(timer); };
  }, []);

  const ready = computer?.state === "ready";
  return (
    <div ref={card} className="agent-card" role="dialog" aria-label={agentName} tabIndex={-1}>
      <header className="agent-card-head">
        <Pet sheet={sheet} state={mood} size={96} look />
        <div>
          <h2>{agentName}</h2>
          <p>{presence}</p>
        </div>
      </header>
      <div className="agent-card-actions">
        <button disabled={callDisabled} title={callTitle} onClick={() => onCall()}>
          <Phone size={18} />
          Call
        </button>
      </div>
      <section className="agent-card-computers" aria-labelledby="agent-card-computers">
        <h3 id="agent-card-computers">Computers</h3>
        <button className="computer-row" disabled={!ready} onClick={onOpenComputer}
          title={computer?.message ?? "Checking"}>
          <span className={`computer-icon ${computer?.state ?? ""}`} aria-hidden="true">
            <Monitor size={20} />
            <i />
          </span>
          <span className="computer-text">
            <span>{agentName}’s computer</span>
            <span className="computer-state">{computer ? COMPUTER_STATES[computer.state] : "Checking…"}</span>
          </span>
          <span className="computer-thumb" aria-hidden="true">
            {ready && <img src={`/api/computer/screenshot?frame=${frame}`} alt=""
              className={thumbnail ? "loaded" : ""}
              onLoad={() => setThumbnail(true)} onError={() => setThumbnail(false)} />}
          </span>
        </button>
      </section>
    </div>
  );
}
