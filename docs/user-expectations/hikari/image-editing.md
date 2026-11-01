# Image editing

Hikari's **Edit an image** uses a source image and a description of the desired result. The browser accepts PNG, JPEG and WebP uploads up to 20 MiB each. It resizes images when needed, and the displayed output size can differ from the original.

Source editing keeps that normalized size. Edit strength starts at 65%: zero preserves the source pixels, while full strength regenerates the editable area. Intermediate strength does not guarantee a particular visual similarity.

An optional opaque mask must match the original source size. Black preserves pixels, white allows editing and gray blends. Replacing or removing the source clears its mask; transparent masks are rejected.

Reference editing is also available to applications using Hikari, including pet generation. It creates a new image guided by a reference's character and appearance, with an optional different output size. The current browser editor uses source editing. Reference editing does not use masks or strength and does not promise exact character identity or pixel preservation.

The same [progress, cancellation and result lifetime](image-generation.md) apply. Users need to inspect the artwork to decide whether an edit meets their intent.

See [Image worker architecture](../../architecture/hikari/worker-image.md) for input normalization and the source/reference contracts.

On the subscription-image branch, source and reference editing always remain local, including masks, strength and pet frame conditioning. The ChatGPT generation choice does not select an editing provider. Editing therefore still requires the local checkpoint and available memory. Subscription editing is not exposed. This branch behavior awaits owner validation and is not asserted as verified on master.
