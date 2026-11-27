# Pet generation references

Kadan coordinates pet creation, Hikari runs image inference (`flux.2-klein-4b`), and Tsuchi owns candidates, immutable artwork and provenance, leases and progress. See the user expectations for [pets](../../user-expectations/kadan/pets.md).

## Creation modes and compatibility

Creation accepts `reference_mode: template_strips | single_image | character_sheet`, defaulting to `template_strips` for new creations. Historical saved jobs in Tsuchi without a `reference_mode` field deserialize as `single_image` (`total_steps = 74`) so existing candidates, imports and active/failed legacy jobs remain intact without a database migration. The mode is part of the creation UUID's idempotency contract and cannot change during updates. Revisions inherit their parent's `reference_mode`.

## Template-strip workflow (`template_strips`, default — 12 steps)

Kadan checks in 12 pose template PNGs on flat `#ff00ff` under `services/kadan/kadan/api/pet_templates/`:

- `still.png` (`768×832`): front-facing neutral pose template.
- 11 horizontal animation/look strip templates (`1536×512`): `idle`, `running-right`, `running-left`, `waving`, `jumping`, `failed`, `waiting`, `running` (with the shared terracotta potted green sprout working motif), `review`, `look-0` (0°–157.5°), and `look-1` (180°–337.5°).

Generation runs in 12 checkpointed steps:

1. **`base` (serial, `768×832`):** Calls Hikari `POST /v1/images/edits` (`mode: "reference"`, `image: still.png`) to transfer the user's character prompt onto the neutral still pose. `prepare_still` chroma-keys `#ff00ff`, removes enclosed pure-magenta pockets, despills fringe pixels, and verifies clear margins.
2. **11 row strips (`idle` through `look-1`, `1536×512`, batched through Hikari's cohort loop):** Calls Hikari `POST /v1/images/edits` (`mode: "reference"`, `image: <row_template>.png`, `reference_images: [compose_on_magenta(base)]`). `prepare_strip` cuts out the strip and validates that `slice_strip` finds all `row.frame_count` non-blank silhouettes with distinct animation frames.
3. **Atlas assembly:** `assemble_atlas_from_strips` splits each prepared `1536×512` strip along minimum-ink column valleys into `row.frame_count` segments, computes one shared scale per row (preserving vertical lift for `jumping` and `running-*`, pinning grounded rows to a shared floor `baseline_y = 202`), packs each frame into a `192×208` cell on the `1536×2288` v2 RGBA canvas, zeroes transparent RGB channels, and encodes lossless WebP (`exact=True`).

For `template_strips` candidates, repair requests target whole animation/look rows (and also map legacy `row/frame` names to their parent row strip), retaining `base` and all other saved strips. Neither `base` nor legacy `character-sheet` can be replaced through repair.

## Legacy per-frame modes (`single_image` / `character_sheet`)

Existing `single_image` (74 steps: `base` + 73 frames) and `character_sheet` (75 steps: `character-sheet`, `base`, + 73 frames) jobs remain supported in Tsuchi and Kadan so saved candidates can still be resumed or repaired.

## Shipped default pet

The frontend serves `/kadan-default-spritesheet.png` (`services/frontend/public/kadan-default-spritesheet.png`, `1536×2288` v2 RGBA atlas) as the default pet for every agent without a selected custom pet.

## Verification

Unit tests mock inference and exercise template loading, strip slicing, mode persistence, multi-reference routing, retries, recovery and atlas assembly. Image checks establish structure only: they cannot prove one character, correct pose, identity consistency or better quality. Those claims require owner review of real generated artwork; no model inference is part of these checks.
