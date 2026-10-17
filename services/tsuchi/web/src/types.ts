export type Kind = "persona" | "memory" | "space" | "deck" | "conversation";
export type View =
  | "overview"
  | Kind
  | "images"
  | "attachments"
  | "activity"
  | "trash"
  | "search";
export interface Agent {
  key: string;
  name: string;
  created_at: string;
}
export interface SourceLink {
  label: string;
  url: string;
}
export interface Message {
  id: string;
  role: "system" | "user" | "assistant" | "tool";
  content: string;
  created_at: string;
  attachment_ids: string[];
  resource_ids: string[];
  sources: SourceLink[];
}
export interface Slide {
  id: string;
  layout: "title" | "text" | "split" | "image";
  title: string;
  body: string;
  notes: string;
  image_asset_id: string | null;
}
export interface Deck {
  schema_version: 1;
  width: 1920;
  height: 1080;
  slides: Slide[];
}
export interface ResourceFields {
  title: string;
  markdown: string;
  parent_id: string | null;
  tags: string[];
  sources: SourceLink[];
  attachment_ids: string[];
}
export interface MarkdownInput extends ResourceFields {
  kind: "persona" | "memory" | "space";
  deck: null;
  messages: [];
}
export interface DeckInput extends ResourceFields {
  kind: "deck";
  deck: Deck;
  messages: [];
}
export interface ConversationInput extends ResourceFields {
  kind: "conversation";
  deck: null;
  messages: Message[];
}
export type ResourceInput = MarkdownInput | DeckInput | ConversationInput;
export type Resource = ResourceInput & {
  id: string;
  agent_key: string;
  version: number;
  created_at: string;
  updated_at: string;
  deleted_at: string | null;
};
export interface Asset {
  id: string;
  agent_key: string;
  title: string;
  filename: string;
  media_type: string;
  size_bytes: number;
  sha256: string;
  is_image: boolean;
  tags: string[];
  sources: SourceLink[];
  version: number;
  created_at: string;
  updated_at: string;
  deleted_at: string | null;
}
export interface Stats {
  resources: Record<Kind, number>;
  assets: number;
  images: number;
  trashed_resources: number;
  trashed_assets: number;
  storage_bytes: number;
  database_bytes: number;
  revisions: number;
}
export interface Activity {
  id: string;
  action: string;
  entity_type: string;
  entity_id: string;
  title: string;
  created_at: string;
}
export interface Revision {
  version: number;
  created_at: string;
  snapshot: Resource;
}

export const kindLabels: Record<Kind, string> = {
  persona: "Personas",
  memory: "Memories",
  space: "Spaces",
  deck: "Slide decks",
  conversation: "Conversations",
};
export const singularLabels: Record<Kind, string> = {
  persona: "persona",
  memory: "memory",
  space: "Space",
  deck: "slide deck",
  conversation: "conversation",
};
export function newResource(kind: Kind): ResourceInput {
  const fields: ResourceFields = {
    title: "",
    markdown: "",
    parent_id: null,
    tags: [],
    sources: [],
    attachment_ids: [],
  };
  if (kind === "deck")
    return {
      ...fields,
      kind,
      deck: { schema_version: 1, width: 1920, height: 1080, slides: [] },
      messages: [],
    };
  return { ...fields, kind, deck: null, messages: [] };
}
