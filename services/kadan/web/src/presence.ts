import type { Message } from "./types.ts";

const STEPS: [Intl.RelativeTimeFormatUnit, number][] = [
  ["minute", 60],
  ["hour", 60],
  ["day", 24],
  ["week", 7],
  ["month", 52 / 12],
  ["year", 12],
];

/** "Active 3 minutes ago" from the agent's latest delivered message. */
export function activeLabel(messages: Pick<Message, "role" | "created_at">[], now = Date.now()) {
  const last = [...messages].reverse().find((message) => message.role === "assistant");
  if (!last) return "Not active yet";
  const seconds = Math.max(0, (now - new Date(last.created_at).getTime()) / 1000);
  if (seconds < 60) return "Active just now";
  let value = seconds / 60;
  for (let index = 0; index < STEPS.length; index++) {
    const next = STEPS[index + 1];
    if (!next || value < next[1]) {
      const format = new Intl.RelativeTimeFormat("en", { numeric: "always" });
      return `Active ${format.format(-Math.floor(value), STEPS[index][0])}`;
    }
    value /= next[1];
  }
  return "Active";
}

export function vncUrl(location: Pick<Location, "protocol" | "host">) {
  return `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/api/computer/vnc`;
}
