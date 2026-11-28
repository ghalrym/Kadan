import type { CSSProperties } from "react";
import type { GroupChatParticipant } from "./groupChat";
import Pet from "./Pet";
import { agentHue, type PetSheet, type PetState } from "./pet";
import { useAgentPet, usePetRefSheet } from "./PetContext";
import { USER_SEED, type PetRef, type UserProfile } from "./profile";

interface Props {
  sheet: PetSheet;
  /** Stable colour seed so people with the default pet still look different. */
  seed: string;
  size?: number;
  state?: PetState;
  className?: string;
}

/** A pet framed in its owner's colour, used wherever people appear as avatars. */
export default function PetAvatar({ sheet, seed, size = 36, state = "idle", className = "" }: Props) {
  const style = { "--avatar-h": agentHue(seed), "--avatar-size": `${size}px` } as CSSProperties;
  return (
    <span className={`pet-avatar ${className}`} style={style} aria-hidden="true">
      <Pet sheet={sheet} state={state} size={Math.round(size * 0.94)} />
    </span>
  );
}

export function UserAvatar({ profile, ...rest }: { profile: UserProfile } & Omit<Props, "sheet" | "seed">) {
  return <PetAvatar sheet={usePetRefSheet(profile.pet)} seed={USER_SEED} {...rest} />;
}

function AgentAvatar({ agentKey, ...rest }: { agentKey: string } & Omit<Props, "sheet" | "seed">) {
  return <PetAvatar sheet={useAgentPet(agentKey)} seed={agentKey} {...rest} />;
}

function BorrowedAvatar({ pet, seed, ...rest }: { pet: PetRef | null; seed: string } & Omit<Props, "sheet" | "seed">) {
  return <PetAvatar sheet={usePetRefSheet(pet)} seed={seed} {...rest} />;
}

/** Kadan's agent shows its own pet; other agents show the pet assigned to them in Group Chat. */
export function ParticipantAvatar({ participant, ...rest }: {
  participant: Pick<GroupChatParticipant, "kind" | "agent_key" | "handle" | "pet">;
} & Omit<Props, "sheet" | "seed">) {
  return participant.kind === "kadan_agent" && participant.agent_key
    ? <AgentAvatar agentKey={participant.agent_key} {...rest} />
    : <BorrowedAvatar pet={participant.pet} seed={participant.handle} {...rest} />;
}
