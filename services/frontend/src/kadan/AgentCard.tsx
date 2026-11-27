import { useEffect, useLayoutEffect, useRef } from "react";
import { Phone, UserRound } from "lucide-react";
import Pet from "./Pet";
import type { PetSheet, PetState } from "./pet";

/** Space kept between the card and the bottom of the window. */
const VIEWPORT_MARGIN = 16;

interface Props {
  agentName: string;
  sheet: PetSheet;
  mood: PetState;
  presence: string;
  onClose: () => void;
  personaDisabled: boolean;
  onEditPersona: () => void;
}

/** The agent's contact card, opened from its pet: presence, calling and persona editing. */
export default function AgentCard({
  agentName,
  sheet,
  mood,
  presence,
  onClose,
  personaDisabled,
  onEditPersona,
}: Props) {
  const card = useRef<HTMLDivElement>(null);

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
  useLayoutEffect(() => {
    // The card hangs below its pet, so its room depends on where the pet sits.
    const fit = () => {
      const element = card.current;
      if (!element) return;
      const top = element.getBoundingClientRect().top;
      element.style.maxHeight = `${Math.max(0, window.innerHeight - top - VIEWPORT_MARGIN)}px`;
    };
    fit();
    window.addEventListener("resize", fit);
    return () => window.removeEventListener("resize", fit);
  }, []);

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
        <span className="agent-card-coming-soon" title="Coming soon" data-tooltip="Coming soon">
          <button type="button" disabled aria-label="Call (Coming soon)">
            <Phone size={18} />
            Call
          </button>
        </span>
        <button
          disabled={personaDisabled}
          title={`Edit ${agentName}’s persona`}
          onClick={() => onEditPersona()}
        >
          <UserRound size={18} />
          Persona
        </button>
      </div>
    </div>
  );
}
