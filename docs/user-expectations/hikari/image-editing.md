# Image editing

Edit an image requires a source and a nonblank description of the desired result. The browser accepts PNG, JPEG or WebP, at most 20 MiB per upload. It normalizes the source to PNG on a 32-pixel grid, within 3,072 pixels per side and 4,718,592 pixels total. The displayed output dimensions are the normalized dimensions, which can differ from the original.

The default is four images, a random seed and 65% edit strength. Source editing keeps the normalized source size. At 0%, it preserves source pixels; at 100%, the editable area starts from noise. Intermediate strength influences how much of the source is retained but does not guarantee a particular visual similarity.

An optional mask must be opaque and match the original source's dimensions. Source and mask receive the same resize. Black preserves pixels, white permits editing and gray blends. Replacing or removing the source clears its mask. Transparent masks are rejected rather than interpreted as preservation.

The current browser interface uses source editing. API clients can separately choose reference mode to generate a new image conditioned on a reference's character and appearance, as used by pet generation. Reference mode does not accept a mask or an explicitly supplied strength. It does not promise pixel preservation or exact character identity.

API uploads must already be normalized PNG data URLs within the same size limits. A reference needs at least 65,536 pixels. Reference output defaults to the reference size; an independent output width and height must be supplied together, as multiples of 32 within the side and total-pixel limits. Independent dimensions are rejected in source mode.

Editing requires the selected image checkpoint and normal memory admission. The same [generation progress, cancellation and gallery-retention limits](image-generation.md) apply. Validate the resulting artwork visually; supported inputs and structural checks do not establish editing quality on a particular prompt or checkpoint.
