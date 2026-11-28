import { useEffect, useLayoutEffect, useRef } from "react";
import { Palette, Pencil, Phone, Trash2, UserRound } from "lucide-react";
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
  callDisabled: boolean;
  callTitle: string;
  onCall: () => void;
  personaDisabled: boolean;
  onEditPersona: () => void;
  onEditPet: () => void;
  agentDisabled: boolean;
  onRename: () => void;
  onDelete: () => void;
}

/** The agent's contact card, opened from its pet: presence, calling, renaming, persona and pet editing, and deletion. */
export default function AgentCard({
  agentName,
  sheet,
  mood,
  presence,
  onClose,
  callDisabled,
  callTitle,
  onCall,
  personaDisabled,
  onEditPersona,
  onEditPet,
  agentDisabled,
  onRename,
  onDelete,
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
        <button disabled={callDisabled} title={callTitle} onClick={() => onCall()}>
          <Phone size={18} />
          Call
        </button>
        <button disabled={agentDisabled} title={`Rename ${agentName}`} onClick={() => onRename()}>
          <Pencil size={18} />
          Rename
        </button>
        <button
          disabled={personaDisabled}
          title={`Edit ${agentName}’s persona`}
          onClick={() => onEditPersona()}
        >
          <UserRound size={18} />
          Persona
        </button>
        <button title={`Edit ${agentName}’s pet`} onClick={() => onEditPet()}>
          <Palette size={18} />
          Edit pet
        </button>
        <hr className="agent-card-divider" />
        <button className="danger-button" disabled={agentDisabled} title={`Delete ${agentName}`} onClick={() => onDelete()}>
          <Trash2 size={18} />
          Delete agent
        </button>
      </div>
    </div>
  );
}
