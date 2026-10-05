# FLUX.2 klein base 4B

Adds one undistilled checkpoint, `black-forest-labs/FLUX.2-klein-base-4B`, pinned to `a3b4f4849157f664bdbc776fd7453c2783562f4d` under Apache 2.0. The approximately 16 GB native bundle includes the text encoder, tokenizer, transformer, VAE and scheduler, excluding the duplicate root transformer export.

The native recipe uses 50 steps and guidance 4.0 for generation and editing, as the official model card specifies. It reuses the saved Image Settings choice, explicit API model override, Kadan resource leases, offload, cancellation and PNG delivery from the klein 4B infrastructure PR. Model download does not start inference.

Fixture tests verify native pipeline arguments for both text and image inputs. No real weights or GPU inference were run; performance and memory peaks are unmeasured.

Source: https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B
