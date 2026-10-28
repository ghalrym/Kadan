---
name: kadan-pet
description: Create, repair, or install a Codex-compatible animated pet (8-column sprite atlas plus pet.json) for a Kadan agent using a local image generator instead of Codex's hatch-pet. Use when making a new agent avatar, replacing the procedural hatchling default, fixing a pet sheet, or wiring real per-agent pets into the Kadan web UI.
---

# Kadan Pet

Kadan agents use the same animated pet format as Codex. Any image generator can produce one: generation is only the art step, and slicing, cleanup, and assembly are deterministic image processing.

Codex's own workflow lives in `~/.codex/skills/hatch-pet/` when installed. Its `references/` hold the authoritative contract and QA rubric, and its `scripts/` (Pillow) extract frames, remove chroma spill, compose and validate atlases. Read its `LICENSE.txt` before reusing scripts; do not copy them into this repository.

## Contract

- Cells are `192x208`, 8 columns per row. Unused cells are fully transparent.
- v1 atlas: 9 rows, `1536x1872`. v2 atlas: 11 rows, `1536x2288`.
- Rows and frame durations (ms):

| Row | State | Frames | Durations | Shows |
|---|---|---|---|---|
| 0 | idle | 6 | 280 110 110 140 140 320 | Breath, blink, tiny bob only |
| 1 | running-right | 8 | 120 ×7, 220 | Travelling right |
| 2 | running-left | 8 | 120 ×7, 220 | Travelling left; may mirror row 1 once approved |
| 3 | waving | 4 | 140 ×3, 280 | Limb pose only |
| 4 | jumping | 5 | 140 ×4, 280 | Vertical body motion only |
| 5 | failed | 8 | 140 ×7, 240 | Deflated; tears or smoke must touch the body |
| 6 | waiting | 6 | 150 ×5, 260 | Expectant, needs user input |
| 7 | running | 6 | 120 ×5, 220 | Working or thinking, not literal running |
| 8 | review | 6 | 150 ×5, 280 | Lean, squint, head tilt; no new props |
| 9 | look 000–157.5 | 8 | static | v2 only, clockwise in 22.5° steps |
| 10 | look 180–337.5 | 8 | static | v2 only |

- Look angle 0° means looking **up**, not neutral or front-facing.
- Package as a folder containing `pet.json` and `spritesheet.webp`:

```json
{
  "id": "my-pet",
  "displayName": "My Pet",
  "description": "One short sentence",
  "spriteVersionNumber": 2,
  "spritesheetPath": "spritesheet.webp"
}
```

Omit `spriteVersionNumber` (or use `1`) for a 9-row sheet. Codex loads pets from `~/.codex/pets/<id>/`.

## Workflow

1. **Base character.** Generate one front-facing full-body image on a flat chroma-key background (pure magenta or green). Any style works if it stays a compact silhouette readable at 192x208. No text, logos, shadows, or scenery. The chosen image is the identity reference for every later job.
   **Optional character sheet.** Choose **Character sheet** under **Generation reference** to generate a multi-view design reference from the description first. Derive a separate neutral base for preview and idle reuse; condition articulated poses directly on the saved sheet. Never fit the full sheet into one frame or copy its layout or repeated views into output poses. This experimental option adds one generation step and has no demonstrated quality advantage. It does not accept reference-sheet uploads.
2. **Individual poses.** Generate each articulated pose on a fixed 384x416 canvas, conditioned on the canonical base or optional saved character sheet. Keep the same camera, body scale and floor. Reuse the separate neutral base only for genuinely neutral opening/closing idle poses, recording that reuse in provenance.
3. **Look rows (v2).** Generate the sixteen required gaze directions individually using the selected reference; inspect their directional behavior in the player.
4. **Cut out.** Remove the chroma background and validate source and final-size margins. Preserve the whole canvas so jump height and body proportions are not normalized away.
5. **Assemble.** Pack 192x208 cells into the eight-column atlas, leaving unused cells transparent. Validate dimensions, row usage and pose variation.
6. **Review in motion.** Play every row at its intended display size. Report small cosmetic differences. Repair clipping, broken anatomy, missing behavior or unusable jitter by rejecting the specific frame and creating a revision that retains every other frame and all provenance. Do not disguise anatomy errors by fitting bounding boxes.

## Kadan integration

- Player: `services/kadan/web/src/Pet.tsx` renders any v1 or v2 sheet with the official timings. `services/kadan/web/src/pet.ts` holds the row contract and look-direction math.
- Default: agents without a real pet get a procedural hatchling from `services/kadan/web/src/hatchling.ts`, tinted by `agentHue(key)`. Its review pose holds a magnifying glass, which the contract above discourages.
- Selection: `PetContext.tsx` resolves the selected Tsuchi candidate for roster, home and chat. `agentPet(key)` supplies the procedural fallback. Generated artwork is never selected automatically; only the user's **Use pet** or **Use hatchling** action changes selection.
- Storage: Tsuchi owns immutable artwork files, the candidate's v2 manifest, generation provenance, durable job progress and the agent's `selected_pet_id`. Jobs use version checks and worker leases. Trashing referenced artwork clears selection. Exports include candidates and their assets.
- Import: an existing Codex/ChatGPT pet folder, zip or spritesheet can be imported through **Edit pet → New pet → Import** (`POST /api/agents/{key}/pets/imports`). Kadan validates the atlas size and stores a ready, unselected candidate with `origin: "imported"`.
- Creation: use Kadan's **Edit pet** dialog or `POST /api/agents/{key}/pets` with a caller UUID. Retain the job UUID and use `GET /pets/{id}`, `/pets/{id}/cancel` or `/pets/{id}/resume`. Creation returns before generation finishes. After interruption, explicit resume reconciles saved output before continuing; do not start a replacement job to recover an existing one.
- Hikari owns all inference. Kadan calls `/v1/images/generations` for the initial reference and `/v1/images/edits` for each articulated pose, using native Qwen reference-image conditioning. Single-image mode uses the canonical base; optional `reference_mode: "character_sheet"` uses the sheet and also derives the neutral base through reference conditioning. The durable mode survives resume and frame repair. Repeated seeds do not replace source conditioning. Structural validation does not establish identity, articulated movement or correct gaze; review every animation and look pose before choosing **Use pet**. See [pet generation architecture](../../../docs/architecture/kadan/pet-generation.md).
- Repair: the editor creates a revision through `/api/agents/{key}/pets/{id}/revisions` with a new id, expected version, rejected frame names and correction reason. Keep the same request id when retrying uncertain creation. The prior candidate stays intact; this is targeted repair, not a restart.
- Kadan prepares 384x416 source cells and downsamples generated cells to 192x208. It preserves each full cell's geometry so jump height and pose alignment are not normalized away. No still-image or transformed-copy fallback substitutes for failed generation.
- Tests stay unit-only, with mocked Hikari/Tsuchi boundaries and drawn image fixtures. Real-model quality, resource use and latency require an explicitly authorized owner validation run; never start inference as part of ordinary unit/build checks.
