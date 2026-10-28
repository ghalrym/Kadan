# Images

Tsuchi's **Images** library stores and previews verified PNG, JPEG, WebP and GIF images for the selected agent. Users can upload images, edit titles, tags and source links, and download their original bytes. The same [file upload limits and save behavior](files.md) apply.

Tsuchi stores artwork supplied by the user or other services; opening the library does not generate images. Hikari owns image inference. Images delivered through Kadan are saved in Tsuchi, while unsaved results in Hikari's playground are not automatically added to this library.

Images can be attached to stored resources or selected for slides within the same agent. Changing an image's metadata does not alter its pixels. To replace artwork, upload a new image and update the relevant links. Search matches metadata, without recognizing objects or reading text in the picture.

Invalid image content must not be presented as a verified preview. A preview failure must remain distinguishable from an empty library. Supported image formats do not guarantee that the artwork meets the user's intent.

An image referenced by active content must be unlinked before it can be trashed. Trashing artwork used by the selected pet clears that selection. See [Trash](trash.md), [Restore](restore.md) and [Kadan pets](../kadan/pets.md).
