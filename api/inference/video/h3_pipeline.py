"""Direct H3 tensor execution in the API process; no scheduler or worker service."""
import gc
import logging
import ctypes
import os
from pathlib import Path
import sys
import tempfile
import traceback
from types import SimpleNamespace

from api.inference.resources import ResourceCancelled

log = logging.getLogger(__name__)


def check_cancel(cancellation):
    if cancellation.is_set():
        raise ResourceCancelled('H3 generation cancelled')


def require_turbo(pipeline):
    active = pipeline.get_lora_status().get('active', {}).get('transformer', [])
    if (len(active) != 1 or active[0].get('merged') is not False
            or active[0].get('strengths') != [1.0]):
        raise RuntimeError('H3 Turbo must be active in dynamic mode at strength 1')


def trim_cpu_heap():
    """Return free glibc arena pages after model disposal on Linux."""
    if sys.platform != 'linux':
        return
    trim = getattr(ctypes.CDLL(None), 'malloc_trim', None)
    if trim is not None:
        trim.argtypes = [ctypes.c_size_t]
        trim.restype = ctypes.c_int
        trim(0)


def render(checkpoint, sampling, device, cancellation, session=None):
    """Release native frames before collection, including cancelled forwards."""
    # CUDA is optional until this provider is selected.
    import torch

    failure = None
    try:
        if session is None:
            _render(checkpoint, sampling, device, cancellation)
        else:
            _render(checkpoint, sampling, device, cancellation, session)
    except BaseException as error:
        # Native forward frames own model tensors. Keeping their traceback while
        # collecting would retain those tensors after Kadan releases its lease.
        error.add_note('Native H3 traceback:\n' + ''.join(traceback.format_exception(error)))
        seen = set()
        current = error
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            traceback.clear_frames(current.__traceback__)
            current.__traceback__ = None
            current = current.__cause__ or current.__context__
        failure = error
    finally:
        # _render's successful locals and failed native frames are now gone.
        gc.collect()
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
            torch.accelerator.memory.empty_host_cache()
        trim_cpu_heap()
    if failure is not None:
        raise failure


class H3Session:
    """Retain native host stores and their single-rank distributed context.

    The provider accounts for host weights, a bounded idle CUDA context, and an
    independently releasable execution budget. Never materialize all streamed
    transformer or encoder layers on the GPU.
    """
    def __init__(self, checkpoint, device):
        self.checkpoint, self.device = checkpoint, device
        self.pipeline = self.args = self.rendezvous = None
        self.cancellation = None
        self.owns_group = False
        self.parked = False
        self.loads = 0
        self.baseline_cuda = 0
        self.full_resident = False

    def load(self, cancellation):
        render(self.checkpoint, None, self.device, cancellation, self)

    def render(self, sampling, cancellation):
        render(self.checkpoint, sampling, self.device, cancellation, self)

    def park(self, context_budget):
        # Optional runtime imports follow the first successful native load.
        import torch

        if self.pipeline is None or self.parked:
            return
        with torch.cuda.device(self.device):
            for component in self.pipeline.modules.values():
                if not isinstance(component, torch.nn.Module):
                    continue
                for offload in getattr(component, 'layerwise_offload_managers', ()) or ():
                    offload.release_all()
                # Native per-stage parking can leave non-layer placeholders.
                # Restore them from the existing CPU copies, never from disk.
                parked = getattr(component, '_parked_non_layer_weights', {})
                parameters = dict(component.named_parameters())
                for name, tensor in parked.items():
                    parameters[name].data = tensor
                parked.clear()
                component.to('cpu')
                # LoRA keeps an unregistered base snapshot/view alongside its
                # registered weights. Keep that backing store on the host too.
                for module in component.modules():
                    backing = getattr(module, 'cpu_weight', None)
                    if isinstance(backing, torch.Tensor) and backing.device.type != 'cpu':
                        module.cpu_weight = backing.to('cpu')
                backing = None
            # The native pipeline separately caches the original Turbo adapter
            # tensors in a plain dict. Module.to() cannot move these ~1.9 GB of
            # aliases; active layer weights already have their own host stores.
            for adapter in getattr(self.pipeline, 'lora_adapters', {}).values():
                for name, tensor in adapter.items():
                    if isinstance(tensor, torch.Tensor) and tensor.device.type != 'cpu':
                        adapter[name] = tensor.to('cpu')
                tensor = None
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
            remaining = max(0, torch.cuda.memory_allocated(self.device) - self.baseline_cuda)
            if remaining > context_budget:
                raise RuntimeError(f'H3 idle CUDA allocations ({remaining} bytes) exceed their context reservation ({context_budget} bytes)')
            self.parked = True
            log.info('H3 session %s parked in RAM; idle CUDA bytes=%s', id(self.pipeline), remaining)

    def resume(self):
        # The native helpers preserve explicitly host-resident vocabulary tables.
        import torch
        from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
            detach_host_resident_tables, restore_host_resident_tables)

        if not self.parked:
            return
        with torch.cuda.device(self.device):
            for component in self.pipeline.modules.values():
                if not isinstance(component, torch.nn.Module):
                    continue
                if self.full_resident or getattr(component, 'layerwise_offload_managers', None):
                    # Only placeholders, non-layer parameters and buffers move;
                    # the real INT8 layers remain in the native CPU host stores.
                    tables = detach_host_resident_tables(component)
                    component.to(torch.device('cuda', self.device))
                    restore_host_resident_tables(tables, torch.device('cuda', self.device))
            self.parked = False
            log.info('H3 session %s restored from retained RAM stores', id(self.pipeline))

    def close(self):
        # Cleanup is called under the provider's shared resource ownership.
        import torch
        from sglang.multimodal_gen.runtime.distributed.parallel_state import cleanup_dist_env_and_memory
        from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import peek_global_component_residency_manager

        manager = peek_global_component_residency_manager()
        retained = self.pipeline or (manager.pipeline if manager is not None and self.owns_group else None)
        if retained is not None:
            for component in retained.modules.values():
                for offload in getattr(component, 'layerwise_offload_managers', ()) or ():
                    offload.release_all()
                    offload.enabled = False
                    offload.release_host_stores()
            retained.modules.clear()
            retained._stages.clear()
            retained._stage_name_mapping.clear()
        if manager is not None and self.owns_group:
            manager.refresh_pipeline(SimpleNamespace(component_residency_strategies={}, _stage_name_mapping={}))
        self.pipeline = retained = None
        try:
            if self.owns_group:
                with torch.cuda.device(self.device):
                    torch.cuda.synchronize()
                cleanup_dist_env_and_memory()
        finally:
            self.owns_group = False
            if self.rendezvous is not None:
                self.rendezvous.cleanup()
                self.rendezvous = None
            gc.collect()
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()
                torch.accelerator.memory.empty_host_cache()
            trim_cpu_heap()


def _render(checkpoint, sampling, device, cancellation, session=None):
    """Run the pinned native pipeline synchronously under Kadan's resource lease.

    Imports follow platform activation and remain lazy so a broken optional
    dependency cannot prevent API startup. The single-rank group is local to
    this process; no launcher, scheduler client, or process pool is constructed.
    """
    from sglang.multimodal_gen.runtime.platforms import initialize_current_platform
    initialize_current_platform()
    from sglang.multimodal_gen.runtime.platforms.plugins import load_plugins, apply_plugin_hooks
    load_plugins()
    apply_plugin_hooks()
    import torch
    from sglang.multimodal_gen.configs.pipeline_configs.minimax_h3 import MiniMaxH3PipelineConfig
    from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
    from sglang.multimodal_gen.runtime.server_args import ServerArgs, set_global_server_args
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        init_distributed_environment, initialize_model_parallel, cleanup_dist_env_and_memory)
    from sglang.multimodal_gen.runtime.pipelines.minimax_h3_pipeline import MiniMaxH3Pipeline
    from sglang.multimodal_gen.runtime.pipelines_core.executors.sync_executor import SyncExecutor
    from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
        get_global_component_residency_manager, peek_global_component_residency_manager)
    from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import configure_layerwise_offload_modules
    from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request, save_outputs

    ephemeral = session is None
    session = session or H3Session(checkpoint, device)
    session.cancellation = cancellation

    class CancellableExecutor(SyncExecutor):
        def before_stage(self, *args, **kwargs):
            check_cancel(session.cancellation)
            return super().before_stage(*args, **kwargs)

    check_cancel(cancellation)
    if session.pipeline is None and torch.distributed.is_initialized():
        raise RuntimeError('H3 requires ownership of the process-local model-parallel context')
    toolkit = Path(sys.prefix) / 'lib' / f'python{sys.version_info.major}.{sys.version_info.minor}' / 'site-packages/nvidia/cu13'
    if (toolkit / 'bin/nvcc').is_file():
        os.environ.setdefault('CUDA_HOME', str(toolkit))
    root = Path(checkpoint)
    config = MiniMaxH3PipelineConfig()
    config.dit_config.arch_config.qkv_checkpoint_grouped = False
    args = session.args or ServerArgs.from_kwargs(
        model_path=str(root / 'FL2VA'), backend='sglang', pipeline_config=config,
        num_gpus=1, tp_size=1, ulysses_degree=1, base_gpu_id=device,
        component_weights_paths={name: str(root / 'FL2VA' / name / 'model.safetensors')
                                 for name in ('transformer', 'text_encoder', 'video_vae', 'audio_vae')},
        component_precisions={'text_encoder': 'fp16', 'video_vae': 'fp16', 'audio_vae': 'fp32'},
        lora_path=str(root / 'loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors'),
        lora_merge_mode='dynamic', lora_scale=1.0,
        cpu_offload_components=[] if session.full_resident else ['video_vae', 'audio_vae'],
        layerwise_resident_layers={'text_encoder': 0}, layerwise_prefetch_size={'text_encoder': 1},
        attention_backend='torch_sdpa', performance_mode='manual' if session.full_resident else 'memory',
        layerwise_offload_components=[] if session.full_resident else ['dit', 'text_encoder'],
        dit_offload_prefetch_size=1, dit_layerwise_resident_layers=0,
        enable_torch_compile=False, disable_conditioning_cache=True,
    )
    pipeline = session.pipeline
    req = result = None
    succeeded = False
    hooks = []
    original_device = torch.cuda.current_device()
    rendezvous = session.rendezvous or tempfile.TemporaryDirectory(prefix='kadan-h3-rendezvous-')
    session.rendezvous, session.args = rendezvous, args
    try:
        torch.cuda.set_device(device)
        set_global_server_args(args)
        if pipeline is None:
            session.baseline_cuda = torch.cuda.memory_allocated(device)
            session.owns_group = True
            init_distributed_environment(world_size=1, rank=0, local_rank=device,
                distributed_init_method=Path(rendezvous.name, 'group').as_uri(),
                device_id=torch.device('cuda', device), timeout=60)
            initialize_model_parallel(tensor_parallel_degree=1, sequence_parallel_degree=1)
            pipeline = MiniMaxH3Pipeline(args.model_path, args, executor=CancellableExecutor(args))
            session.pipeline = pipeline
            session.loads += 1
            log.info('H3 session %s constructed; checkpoint loads=%s', id(pipeline), session.loads)
            check_cancel(cancellation)
            require_turbo(pipeline)
            manager = get_global_component_residency_manager(pipeline, args)
            configure_layerwise_offload_modules(pipeline.modules, args,
                component_names=args.layerwise_offload_components, pin_budget=manager.host_pin_budget)
        else:
            session.resume()
            require_turbo(pipeline)
        if sampling is None:
            check_cancel(cancellation)
            succeeded = True
            return
        def guard(module, inputs):
            check_cancel(cancellation)
        for component in pipeline.modules.values():
            if isinstance(component, torch.nn.Module):
                for module in component.modules():
                    hooks.append(module.register_forward_pre_hook(guard))
        params = SamplingParams.from_user_sampling_params_args(args.model_path, server_args=args, **sampling)
        params._set_output_file_name()
        req = prepare_request(args, params)
        params.prepare_video_request_for_queue(req)
        with torch.inference_mode():
            result = pipeline.forward(req, args)
        check_cancel(cancellation)
        if result.error or result.output is None or len(result.output) != 1:
            raise RuntimeError(result.error or 'H3 produced no single audiovisual output')
        output = Path(sampling['output_path']) / sampling['output_file_name']
        paths = save_outputs(result.output, req.data_type, req.fps, True, lambda _: str(output),
                             audio=result.audio, audio_sample_rate=result.audio_sample_rate)
        params.validate_video_final_outputs(paths, req)
        check_cancel(cancellation)
        succeeded = True
    finally:
        if req is not None:
            req.sampling_params.cleanup_video_request(req)
        for hook in hooks:
            hook.remove()
        req = result = None
        try:
            if ephemeral or not succeeded:
                session.close()
        finally:
            torch.cuda.set_device(original_device)
