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
2. **Row strips.** For each state, generate one horizontal strip with that row's frame count, grounded on the base image (img2img, IP-Adapter, or a character LoRA). Generate a whole row per image so poses stay coherent. Reject strips with detached effects (sparkles, speed lines, floor shadows, floating symbols) or chroma-adjacent colors on the pet.
3. **Look rows (v2).** Generate a four-pose cardinal strip (up, right, down, left) and approve it. Then generate row 9 as one coherent eight-pose strip from those cardinals, check it, and only then generate row 10 using the cardinals and row 9 as references.
4. **Cut out.** Key out the background, suppress colored edge spill, split each strip into frames, and register every frame into a 192x208 cell at the same scale with the feet on the same baseline. Check suspect cells on a high-contrast background for accidental holes inside the body.
5. **Assemble.** Compose the 8-column atlas, leave unused cells transparent, and save as `spritesheet.webp`. Validate dimensions, row usage, and transparency.
6. **Review in motion.** Play every row at the contract durations. Fix the whole containing row when a frame pops, rescales, crops, or drifts in identity; never patch a single cell with output from a different generation.

## Kadan integration

- Player: `services/kadan/web/src/Pet.tsx` renders any v1 or v2 sheet with the official timings. `services/kadan/web/src/pet.ts` holds the row contract and look-direction math.
- Default: agents without a real pet get a procedural hatchling from `services/kadan/web/src/hatchling.ts`, tinted by `agentHue(key)`. Its review pose holds a magnifying glass, which the contract above discourages.
- Hook: `agentPet(key)` in `Pet.tsx` is the single place that chooses a sheet. Return `{ url, version }` (add `pixelated: true` only for pixel-art sheets) for a real pet.
- Storage: the Kadan API has no avatar field yet. Persisting a pet per agent needs a backend design decision (where sheets are stored, how they are uploaded, and how the agent model references them); confirm the approach with the project owner before building it.
- Generation is an offline asset-making step run with the user's own tools. Do not add image-generation calls or model runtimes to Kadan application code, and do not seed pets into Postgres.
- Tests stay unit-only: `npm test` in `services/kadan/web` covers the atlas contract and look math.
