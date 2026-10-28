import type RFB from "@novnc/novnc";

/** Keys belong to this viewer, not to the shared computer or other viewers. */
export class ComputerControl {
  private client: RFB;
  private connected = false;
  private held = new Map<string | number, { keysym: number; code: string | null }>();

  constructor(client: RFB) {
    this.client = client;
    client.viewOnly = true;
    client.addEventListener("connect", () => { this.connected = true; });
    client.addEventListener("disconnect", () => {
      this.connected = false;
      this.held.clear();
    });
    const sendKey = client.sendKey.bind(client);
    client.sendKey = (keysym, code, down) => {
      const accepted = this.connected && !client.viewOnly;
      sendKey(keysym, code, down);
      // Omitted down recursively calls the public sendKey for the press and release.
      if (!accepted || down === undefined) return;
      const key = code ?? keysym;
      if (down) this.held.set(key, { keysym, code });
      else this.held.delete(key);
    };
  }

  setControl(control: boolean): void {
    const client = this.client;
    if (!control) {
      // noVNC sets viewOnly before ungrab's keyups, which sendKey then drops.
      for (const { keysym, code } of this.held.values()) client.sendKey(keysym, code, false);
      this.held.clear();
    }
    client.viewOnly = !control;
    if (control) client.focus();
  }

  disconnect(): void {
    this.setControl(false);
    this.connected = false;
    this.client.disconnect();
  }
}
