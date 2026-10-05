# FLUX.2 klein 9B KV

Adds the single `black-forest-labs/FLUX.2-klein-9b-kv` checkpoint pinned to `a6dfb36eca3a3906eb2fd460795adfb844e5fcce`, identified by its official public README commit link. The native bundle is approximately 34.8 GB and includes the text encoder, tokenizer, transformer, VAE and scheduler, excluding the duplicate root transformer export.

Generation and editing use `Flux2KleinKVPipeline` from the pinned Diffusers integration with 4 steps. The shared Kadan image lifecycle owns memory admission, offload, cancellation and PNG publication. The existing Image Settings selection persists independently of language-model selection.

The non-commercial license notice links the official terms and requires explicit Continue. An existing authorized HF_TOKEN and prior approved Hub access are required for actual downloads; acknowledgement grants neither commercial rights nor Hub access. All applicable deployment obligations still apply. No credentials were created or changed.

Fixture tests verify the actual native call for both text and image inputs. No model weights, live gated download, GPU inference, quality or throughput tests were run. Public file previews establish artifact names; gated contents were not fetched in development.

Source: https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-kv
