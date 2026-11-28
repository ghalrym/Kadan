import { ExternalLink } from "lucide-react";
import { serverAddress } from "./api";
import { CopyField, PageHeader } from "./common";

export default function SettingsPage() {
  const address = serverAddress();
  return (
    <div className="srv-page">
      <PageHeader title="Settings" />
      <section className="srv-card">
        <header className="srv-card-head">
          <h2>Connect Kadan</h2>
          <p>In the Kadan app, open <strong>Settings</strong> and paste this address. Devices on the same network can use it too.</p>
        </header>
        <CopyField value={address} label="server address" />
      </section>
      <section className="srv-card">
        <header className="srv-card-head">
          <h2>Connect other apps</h2>
          <p>
            Apps that work with OpenAI-style AI services can use the models on this computer. Give them this address; no key
            is needed.
          </p>
        </header>
        <CopyField value={`${address}/hikari/v1`} label="AI address" />
        <a className="button srv-link-button" href="/hikari/docs" target="_blank" rel="noreferrer">
          <ExternalLink size={14} />
          Technical reference for developers
        </a>
      </section>
      <section className="srv-card">
        <header className="srv-card-head">
          <h2>Starting, stopping and updates</h2>
          <p>
            Use the Kadan Server window on this computer to start or stop the server, choose where it keeps its data, check
            for updates and read its logs.
          </p>
        </header>
      </section>
    </div>
  );
}
