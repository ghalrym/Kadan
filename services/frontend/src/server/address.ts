import { useEffect, useState } from "react";
import { getDesktopServerConfig, getHostStatus, isDesktopApp } from "../kadan/desktop";

/**
 * The address other devices use to reach this Kadan. In a browser it is the page's own
 * origin; inside the Kadan app the page comes from the app's private proxy, so the
 * address is this computer's network address (hosting) or the other computer's (connected).
 */
export function useServerAddress(): string {
  const [address, setAddress] = useState(() => (isDesktopApp() ? "" : window.location.origin));
  useEffect(() => {
    if (!isDesktopApp()) return;
    let alive = true;
    void (async () => {
      try {
        const config = await getDesktopServerConfig();
        if (config.mode === "connect") {
          if (alive) setAddress(config.active_host);
          return;
        }
        const host = await getHostStatus();
        const ip = host.settings.bind_all_interfaces && host.hardware.lan_ip ? host.hardware.lan_ip : "127.0.0.1";
        if (alive) setAddress(`http://${ip}:${host.settings.server_port}`);
      } catch {
        if (alive) setAddress(window.location.origin);
      }
    })();
    return () => {
      alive = false;
    };
  }, []);
  return address;
}
