# Slide decks

Tsuchi lets users create and edit slide decks for the selected agent. Each deck uses a fixed widescreen canvas and supports up to 100 slides. Users can add, delete and reorder slides, edit titles, Markdown text and speaker notes, and choose title, text, split or image layouts.

Each slide can use one image from that agent's image library. Removing the image from a slide leaves the stored image available. Uploading and generating images are separate operations; the deck editor does not generate artwork.

The preview updates while editing, but changes persist only after saving. Closing with unsaved changes asks whether to discard them; reloading can lose them. Failed saves and conflicting edits must remain visible. Saved decks support revisions and trash/restore.

Namespace exports retain decks as editable JSON alongside their image files. The current editor has no freeform element positioning, animations, slideshow player or PowerPoint import/export. A preview does not establish compatibility with another presentation application.

See [Images](images.md), [Revisions](revisions.md) and [Export](export.md).
