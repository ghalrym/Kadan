import { useState } from "react";
import Pet from "./Pet";
import type { PetSheet, PetState } from "./pet";

const animations: { value: PetState; label: string }[] = [
  { value: "idle", label: "Idle" },
  { value: "waving", label: "Wave" },
  { value: "jumping", label: "Jump" },
  { value: "running-right", label: "Run right" },
  { value: "running-left", label: "Run left" },
  { value: "running", label: "Work" },
  { value: "waiting", label: "Wait" },
  { value: "review", label: "Review" },
  { value: "failed", label: "Failure" },
];
const compass: Record<number, string> = { 0: "up", 90: "right", 180: "down", 270: "left" };

export const animationLabel = (state: string) =>
  animations.find((animation) => animation.value === state)?.label ?? state;

interface Props {
  sheet: PetSheet;
  /** False while the atlas is still loading. */
  ready?: boolean;
  failed?: boolean;
}

/** Plays every animation of a sheet and, for v2 sheets, each of the 16 look poses. */
export default function PetStage({ sheet, ready = true, failed = false }: Props) {
  const [state, setState] = useState<PetState>("idle");
  const [look, setLook] = useState<number | undefined>(undefined);
  const directions = sheet.version === 2;
  return (
    <div className="pet-stage">
      <div className="pet-stage-dial">
        {ready ? <Pet sheet={sheet} size={192} grounded look={directions}
          state={look === undefined ? state : "idle"} lookDirection={look} />
          : <p role="status" className="quiet">{failed ? "The pet artwork could not be loaded." : "Loading artwork…"}</p>}
        {ready && directions && <div className="pet-look-ring" role="group" aria-label="Look direction">
          {Array.from({ length: 16 }, (_, index) => {
            const angle = index * 22.5;
            const radians = (angle * Math.PI) / 180;
            return <button key={angle} type="button" aria-pressed={look === angle}
              aria-label={`Look ${angle}°${compass[angle] ? ` (${compass[angle]})` : ""}`}
              title={`Look ${angle}°`}
              className={compass[angle] ? "cardinal" : ""}
              style={{ left: `${50 + 46 * Math.sin(radians)}%`, top: `${50 - 46 * Math.cos(radians)}%` }}
              onClick={() => setLook(look === angle ? undefined : angle)} />;
          })}
        </div>}
      </div>
      <div className="pet-stage-controls" role="group" aria-label="Animation">
        {animations.map((animation) => <button key={animation.value} type="button"
          className={`chip ${look === undefined && state === animation.value ? "active" : ""}`}
          aria-pressed={look === undefined && state === animation.value}
          onClick={() => { setState(animation.value); setLook(undefined); }}>
          {animation.label}
        </button>)}
      </div>
      <p className="field-hint pet-stage-hint">
        {directions
          ? look === undefined ? "Choose a dot to check a look direction, or move your cursor around the pet." : `Looking ${look}°. Choose the dot again to return to the animation.`
          : "This is a v1 sheet without look directions."}
      </p>
    </div>
  );
}
