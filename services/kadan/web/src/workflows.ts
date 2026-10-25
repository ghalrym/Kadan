export interface WorkflowBundle {
  schema_version: 1;
  title: string;
  instructions: string;
}

export interface WorkflowSelection {
  id: string;
  version: number;
}

export function parseWorkflow(raw: string): WorkflowBundle {
  // Bound before parsing so an import cannot smuggle a large unrelated payload.
  if (new TextEncoder().encode(raw).length > 80000) throw new Error("Workflow file is too large.");
  const value = JSON.parse(raw);
  if (!value || typeof value !== "object" || Array.isArray(value) ||
      Object.keys(value).sort().join(",") !== "instructions,schema_version,title" ||
      value.schema_version !== 1 || typeof value.title !== "string" ||
      !value.title.trim() || value.title.length > 500 ||
      typeof value.instructions !== "string" || !value.instructions.trim() ||
      value.instructions.length > 12000) {
    throw new Error("Use a version 1 workflow with only title and instructions (up to 12,000 characters).");
  }
  return value;
}

export function exportWorkflow(value: { title: string; markdown: string }): string {
  return JSON.stringify(parseWorkflow(JSON.stringify({
    schema_version: 1, title: value.title, instructions: value.markdown,
  })), null, 2);
}
