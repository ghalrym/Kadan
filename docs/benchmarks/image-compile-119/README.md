# Isolated regional compilation experiment

These are the exact harness, supervisor, manifest and launch arguments for the
GPU1 experiment against source `6e9529d0902ccffbc89c0d3ee546880c0b44d57f`.
They are evidence, not a production compile setting or portable deployment script.
The manifest pins settings and worker hashes. Paths in launch-argv.json are local
to the recorded host. Read the GPU UUID, volumes, memory bounds and source mounts
before reproducing; never replace the live GPU0 service.

FIFO order: text A → eager image B → text C → first compiled image D → text E →
warm compiled image F → text G. All images use BF16, 2048×2048, 40 steps, seed42,
component offload, identical prompt and VAE tiling. Only the 32 repeated transformer
blocks are compiled, in default mode with fullgraph=True and the existing attention
processor. Compilation remains harness-only. Call10 is traced with operator shapes.
Warm F must not recompile after park/unpark; parked Torch allocation must fit the
existing512MiB framework allowance. Native cache occupancy/source-byte checks remain.

## Compiler memory accounting

The original96GiB run was stopped after eager B, before compilation: its91.080GiB
logical envelope left4.920GiB without an explicit compiler reservation. That run
is not compilation evidence. Its completed eager output and cached-step trace are
preserved at `image-compile-119/evidence/` beside the workspace checkout.

The corrected experiment uses a separate128GiB logical and no-swap cgroup limit.
A non-evictable32GiB host reservation covers compiler processes, retained compiler
state, and disk-cache page residency; it stays accounted until process exit.
Combined declared host envelope is123.080GiB, leaving4.920GiB margin. This does not
change production admission or the earlier96GiB acceptance result.

The compiler cache is a writable ext4 disk bind on /dev/sda1, not tmpfs. Its5GiB
size cap bounds disk storage; it does not bound compiler RAM. Up to that full5GiB
of resident file pages is included in the32GiB allowance, leaving27GiB for compiler
process/state estimates. These are accounting allowances, not separately enforced
subprocess limits. The128GiB cgroup hard limit covers all workers and charged file
pages; the supervisor stops at126GiB, recording total current/peak, anonymous,
file and shared-memory usage every5seconds. A transient allocation can exceed the
sampling threshold, so the hard limit remains the last boundary. Worker concurrency
is2. Host preflight requires144GiB available, with16GiB ongoing minimum.

Evidence directory for the corrected run:
`/home/andrew/Documents/Codex/2026-10-08/task-4/image-compile-119-budgeted/evidence/`.
The candidate failed its GPU headroom guard; see RESULT.md and result-summary.json. No compilation speedup is claimed.
