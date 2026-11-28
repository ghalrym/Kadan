import { ExternalLink } from "lucide-react";
import { isDesktopApp } from "../kadan/desktop";
import { useServerAddress } from "./address";
import ThisComputer from "./ThisComputer";
import { CopyField, PageHeader } from "./common";

export default function SettingsPage() {
  const address = useServerAddress();
  return (
    <div className="srv-page">
      <PageHeader title="Settings" />
      <section className="srv-card">
        <header className="srv-card-head">
          <h2>Connect Kadan</h2>
          <p>
            To use this Kadan from another computer, open Kadan there, go to <strong>Settings → Kadan Server → Change</strong>,
            choose <strong>Another computer</strong> and paste this address.
          </p>
        </header>
        {address && <CopyField value={address} label="server address" />}
      </section>
      <section className="srv-card">
        <header className="srv-card-head">
          <h2>Connect other apps</h2>
          <p>
            Apps that work with OpenAI-style AI services can use the models on this computer. Give them this address; no key
            is needed.
          </p>
        </header>
        {address && <CopyField value={`${address}/hikari/v1`} label="AI address" />}
        <a className="button srv-link-button" href="/hikari/docs" target="_blank" rel="noreferrer">
          <ExternalLink size={14} />
          Technical reference for developers
        </a>
      </section>
      {isDesktopApp() ? (
        <ThisComputer />
      ) : (
        <section className="srv-card">
          <header className="srv-card-head">
            <h2>Starting and stopping</h2>
            <p>
              Start, stop and adjust Kadan’s services from the Kadan app on the computer that runs them, under Kadan
              Server → Settings.
            </p>
          </header>
        </section>
      )}
    </div>
  );
}
