# File metadata queries

Tsuchi lists assets within one agent namespace, filtering active or trashed records and optionally images. Search matches every whitespace-separated, casefolded query token as a literal substring across titles, filenames, tags and source labels/URLs. Media types, extracted text and semantic similarity are not searched. See [file expectations](../../user-expectations/tsuchi/files.md).

The existing asset-list endpoint accepts an optional `limit` from 1 to 200 and a nonnegative `offset`, defaulting to zero. Omitting the limit preserves the existing unpaged result. Results keep `updated_at DESC, id ASC` ordering, with pagination applied after matching. Metadata rows are consumed lazily until the matching page is full; a late page or no matches can still scan the collection. Separate reads do not provide snapshot pagination.

Kadan requests one lookahead match for its bounded [file discovery](../kadan/file-delivery.md) tool. Existing library callers can continue to request all matching metadata.
