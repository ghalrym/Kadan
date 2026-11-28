interface MdNode {
  type: string;
  value?: string;
  children?: MdNode[];
  data?: { hName: string; hProperties: Record<string, string> };
}

const MENTION = /(^|[\s(])@([a-zA-Z0-9][a-zA-Z0-9_-]{0,47})\b/g;

/** Turn `@handle` text for known participants into highlighted mentions, leaving code untouched. */
export function remarkMentions(handles: Set<string>) {
  const split = (value: string): MdNode[] => {
    const parts: MdNode[] = [];
    let last = 0;
    for (const match of Array.from(value.matchAll(MENTION))) {
      const handle = match[2];
      if (!handles.has(handle.toLowerCase())) continue;
      const start = (match.index ?? 0) + match[1].length;
      if (start > last) parts.push({ type: "text", value: value.slice(last, start) });
      parts.push({
        type: "mention",
        children: [{ type: "text", value: `@${handle}` }],
        data: { hName: "span", hProperties: { className: "mention" } },
      });
      last = start + handle.length + 1;
    }
    if (!parts.length) return [{ type: "text", value }];
    if (last < value.length) parts.push({ type: "text", value: value.slice(last) });
    return parts;
  };
  const walk = (node: MdNode) => {
    if (!node.children) return;
    node.children = node.children.flatMap((child) => {
      if (child.type === "text" && child.value) return split(child.value);
      if (child.type !== "link") walk(child);
      return [child];
    });
  };
  return () => (tree: MdNode) => walk(tree);
}
