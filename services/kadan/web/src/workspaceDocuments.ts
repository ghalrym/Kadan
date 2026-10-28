import type { Resource } from "./types.ts";

export function workspaceDocuments(resources: Resource[]): Resource[] {
  return resources.filter(resource => resource.kind === "space" || resource.kind === "deck");
}

export function documentTrail(document: Resource, documents: Resource[]): Resource[] {
  const byId = new Map(documents.map(row => [row.id, row]));
  const trail = [document];
  const seen = new Set([document.id]);
  let parent = document.parent_id ? byId.get(document.parent_id) : undefined;
  while (parent && !seen.has(parent.id)) {
    trail.unshift(parent);
    seen.add(parent.id);
    parent = parent.parent_id ? byId.get(parent.parent_id) : undefined;
  }
  return trail;
}

export interface DocumentRow {
  document: Resource;
  depth: number;
}

export function documentOutline(documents: Resource[]): DocumentRow[] {
  const byParent = new Map<string | null, Resource[]>();
  const ids = new Set(documents.map(document => document.id));
  for (const document of documents) {
    const parent = document.parent_id && ids.has(document.parent_id) ? document.parent_id : null;
    const siblings = byParent.get(parent) ?? [];
    siblings.push(document);
    byParent.set(parent, siblings);
  }
  const rows: DocumentRow[] = [];
  const seen = new Set<string>();
  function visit(document: Resource, depth: number) {
    if (seen.has(document.id)) return;
    seen.add(document.id);
    rows.push({ document, depth });
    for (const child of byParent.get(document.id) ?? []) visit(child, depth + 1);
  }
  for (const root of byParent.get(null) ?? []) visit(root, 0);
  // Keep documents reachable even if a stored parent link is incomplete or cyclic.
  for (const document of documents) visit(document, 0);
  return rows;
}
