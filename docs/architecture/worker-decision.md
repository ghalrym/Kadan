# Decisions worker (`decision`)

Answers structured decisions (choices, scores and Noul answers) with the pinned
Laya model. It is small and CPU-only so agents get fast answers without taking
GPU memory from generation.

## Serves

- `/v1/decisions`.

## Engine

- Laya on the CPU, inside a fixed 3 GiB host envelope.

## Memory

- **Minimum and preferred:** the 3 GiB envelope in RAM. It uses no VRAM.

## Scheduling

- Priority 1, shared with the language model. Lower-priority workers leave its
  RAM minimum free while it is not loaded.
- Its budget never includes a GPU.

## Constraints

- Keep it host-only. Giving it VRAM would compete with the language model for
  no benefit.
