# Downloading chat

Kadan's **Download saved chat as Markdown** button downloads a readable snapshot of the open conversation. It includes all saved messages loaded in Kadan, including older messages behind **Show earlier messages**, with their roles, saved timestamps and Markdown text. The file identifies the agent, conversation and saved snapshot.

Drafts, unsent or still-sending messages, live work indicators and retired tool logs are excluded. The button is disabled when there are no saved messages. Downloading requires no model or inference. If updates are unavailable, the snapshot can be older than the latest saved conversation; reload successfully before downloading when freshness matters. Later messages require another download.

Attachments appear as download links labeled with their stored IDs. Referenced resource IDs and saved source links are retained; unsupported source links are labeled unavailable. The download does not fetch or embed files, images or audio, and does not include a voice-call transcript. Links depend on the original Kadan installation or source remaining accessible and the referenced content remaining available. Moving the file elsewhere does not preserve those files or grant access to them.

The browser handles saving the file. A started download does not confirm that the file was saved; check the browser's download result. A reported failure can be retried with the same button. Markdown is for reading and sharing, with no conversation import or restore feature. For saved records and original files, see Tsuchi's [namespace export](../tsuchi/export.md).
