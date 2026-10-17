import { useEffect, useState } from "react";
import Composer, { readiness } from "./Composer";
import Pet, { agentPet, useMood } from "./Pet";
import type { Opening } from "./Chat";
import type { PetState } from "./pet";
import type { Resource, Status } from "./types";

interface Props {
  agentKey: string;
  agentName: string;
  entrance: PetState;
  personas: Resource[];
  status: Status | null;
  disabled: boolean;
  onStart: (first: Opening) => void;
}

export default function Home({ agentKey, agentName, entrance, personas, status, disabled, onStart }: Props) {
  const draftKey = `kadan.draft.${agentKey}.home`;
  const [draft, setDraft] = useState(() => sessionStorage.getItem(draftKey) || "");
  const [persona, setPersona] = useState("");
  const ready = readiness(status);
  const [mood, react] = useMood(status && !ready.chat ? "waiting" : draft.trim() ? "review" : "idle");
  useEffect(() => react(entrance, 2), [entrance, react]);
  useEffect(() => {
    if (draft) sessionStorage.setItem(draftKey, draft);
    else sessionStorage.removeItem(draftKey);
  }, [draft, draftKey]);

  return (
    <section className="home" aria-label={`Start a thread with ${agentName}`}>
      <div className="home-stage">
        <Pet sheet={agentPet(agentKey)} state={mood} size={192} look grounded className="hero-pet" />
        <h2>{agentName}</h2>
      </div>
      <Composer
        agentName={agentName}
        value={draft}
        onChange={setDraft}
        onSend={(generate) => {
          if (!draft.trim() || disabled) return;
          onStart({ content: draft.trim(), persona, generate });
        }}
        sending={false}
        blocked={disabled}
        status={status}
        personas={personas}
        persona={persona}
        onPersona={setPersona}
        pet={null}
        autoFocus
      />
    </section>
  );
}
