import { useEffect, useRef, useState } from "react";
import { Palette } from "lucide-react";
import PetAvatar from "./PetAvatar";
import { usePetRefSheet } from "./PetContext";
import { saveProfile, USER_SEED, type UserProfile } from "./profile";

interface Props {
  profile: UserProfile;
  onSaved: (profile: UserProfile) => void;
  onEditPet: () => void;
  onReload: () => Promise<void>;
  onClose: () => void;
}

/** The person's own name and pet, shown on their Group Chat messages. */
export default function ProfileDialog({ profile, onSaved, onEditPet, onReload, onClose }: Props) {
  const dialog = useRef<HTMLDialogElement>(null);
  const active = useRef(true);
  const sheet = usePetRefSheet(profile.pet);
  const [name, setName] = useState(profile.display_name);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [stale, setStale] = useState(false);

  useEffect(() => {
    active.current = true;
    dialog.current?.showModal();
    return () => { active.current = false; };
  }, []);

  function close() {
    if (busy) return;
    onClose();
  }

  async function save(event: React.FormEvent) {
    event.preventDefault();
    const trimmed = name.trim();
    if (busy || stale || !trimmed) return;
    if (trimmed === profile.display_name) { onClose(); return; }
    setBusy(true);
    setError("");
    try {
      const saved = await saveProfile(profile, { display_name: trimmed });
      if (!active.current) return;
      onSaved(saved);
      onClose();
    } catch (e) {
      if (!active.current) return;
      setError(`${(e as Error).message} Reload your profile, then try again.`);
      setStale(true);
    } finally {
      if (active.current) setBusy(false);
    }
  }

  async function reload() {
    setBusy(true);
    try {
      await onReload();
      if (!active.current) return;
      setStale(false);
      setError("");
    } catch (e) {
      if (active.current) setError((e as Error).message);
    } finally {
      if (active.current) setBusy(false);
    }
  }

  return (
    <dialog ref={dialog} className="profile-dialog" aria-labelledby="profile-title"
      onCancel={(event) => { event.preventDefault(); close(); }}>
      <form onSubmit={save}>
        <h2 id="profile-title">Your profile</h2>
        <div className="profile-pet">
          <PetAvatar sheet={sheet} seed={USER_SEED} size={88} state="waving" />
          <button type="button" disabled={busy} onClick={onEditPet}>
            <Palette size={15} />
            Edit pet
          </button>
        </div>
        <label>
          Name
          <input autoFocus required maxLength={200} value={name} disabled={busy}
            onChange={(event) => setName(event.target.value)} />
        </label>
        <p className="field-hint">Shown on your Group Chat messages so agents know who is talking.</p>
        {error && <p className="error" role="alert">{error}</p>}
        {stale && <button type="button" disabled={busy} onClick={() => void reload()}>Reload profile</button>}
        <div className="form-actions">
          <button type="button" disabled={busy} onClick={close}>Cancel</button>
          <button className="primary" disabled={busy || stale || !name.trim()}>
            {busy ? "Waiting…" : "Save"}
          </button>
        </div>
      </form>
    </dialog>
  );
}
