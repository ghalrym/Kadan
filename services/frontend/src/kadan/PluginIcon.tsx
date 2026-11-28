import type { CSSProperties } from "react";
import { Code2, Plug } from "lucide-react";
import type { McpServer } from "./integrations";

export default function PluginIcon({ preset, name, size = 40 }: { preset: McpServer["preset"]; name: string; size?: number }) {
  const hue = [...name].reduce((sum, char) => sum + char.charCodeAt(0), 0) % 360;
  return (
    <span className={`plugin-icon ${preset}`} aria-hidden="true"
      style={{ width: size, height: size, "--plugin-h": hue } as CSSProperties}>
      {preset === "t3_code" ? <Code2 size={size * 0.5} /> : name.trim()[0]?.toUpperCase() || <Plug size={size * 0.5} />}
    </span>
  );
}
