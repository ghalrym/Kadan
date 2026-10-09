"""Owned rank entrypoint. Bootstrap parent-death fencing before GPU imports."""
import ctypes
from datetime import timedelta
import gc
import importlib
import json
import logging
import os
from pathlib import Path
import signal
import socket
import sys
import time

from api.inference.image.rank_processes import encode_rank_message, receive_rank_message
from api.inference.image.cuda_wait import configure_blocking_sync


def configure_progress_logging():
    # Rank workers do not inherit uvicorn's logging setup. Scope output to this
    # bounded progress stream instead of enabling verbose dependency logging.
    logger = logging.getLogger('api.inference.image.two_rank_pipeline')
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.addHandler(logging.StreamHandler(sys.stderr))


def configure_execution(torch, config):
    """Apply validated rank policy after process ownership is established."""
    if config['threads'] is not None:
        torch.set_num_threads(config['threads'])
        torch.set_num_interop_threads(config['threads'])
    device = config['device']
    torch.cuda.set_device(device)
    wait_policy = configure_blocking_sync(device) if config['blocking_sync'] else {'policy': 'default'}
    torch.cuda.set_per_process_memory_fraction(config['execution_bytes']/torch.cuda.get_device_properties(device).total_memory)
    torch.backends.cuda.matmul.allow_tf32 = False
    return wait_policy


def main():
    connection = socket.socket(fileno=int(sys.argv[1]))
    config = json.loads(sys.argv[2])
    # Linux is required by this opt-in runtime. Killing the API must not leave a
    # detached GPU rank. Check the parent again after installing the death signal.
    if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), 'Cannot install rank parent-death fence')
    if os.getppid() != config['parent_pid']:
        raise RuntimeError('Rank controller exited during bootstrap')
    configure_progress_logging()
    os.sched_setaffinity(0, config['cpus'])
    # Optional imports happen only after process ownership is established.
    torch = importlib.import_module('torch')
    dist = importlib.import_module('torch.distributed')
    diffusers = importlib.import_module('diffusers')
    engine_type = importlib.import_module('api.inference.image.two_rank_pipeline').TwoRankQwenPipeline
    device, rank = config['device'], config['rank']
    wait_policy = configure_execution(torch, config)
    def groups(sequence):
        dist.init_process_group('nccl', init_method=(Path(config['directory'])/f'rendezvous-{sequence}').as_uri(),
            rank=rank, world_size=2, timeout=timedelta(seconds=config['collective_seconds']), device_id=torch.device('cuda',device))
        return dist.new_group(backend='gloo', timeout=timedelta(seconds=config['collective_seconds']))

    pipeline = engine = control = None
    sequence = 0
    state = 'new'
    while True:
        command = receive_rank_message(connection)
        started = time.monotonic()
        if command.get('version') != 1 or command.get('session') != config['session'] or command.get('sequence') != sequence+1:
            raise ValueError('Stale or mismatched rank command')
        sequence += 1
        operation = command['operation']
        receipt = dict(command, rank=rank, device=device, status='ok', cuda_wait=wait_policy)
        try:
            if operation == 'ready' and state == 'new':
                control = groups(sequence)
                pipeline = diffusers.QwenImage21Pipeline.from_pretrained(config['checkpoint'],
                    local_files_only=True, torch_dtype=torch.bfloat16, use_safetensors=True)
                if not pipeline.transformer.config.causal_condition or len(pipeline.transformer.transformer_blocks) != 32:
                    raise ValueError('Unsupported Qwen transformer configuration')
                pipeline.vae.enable_tiling()
                pipeline.enable_model_cpu_offload(gpu_id=device)
                engine = engine_type(pipeline, rank, control)
                state = 'ready'
            elif operation == 'park' and state == 'ready':
                pipeline.remove_all_hooks()
                pipeline.to('cpu')
                # Idle text must not pay for retained NCCL buffers. Preserve CPU
                # weights, destroy both communicators, retain only CUDA context.
                dist.destroy_process_group(control)
                dist.destroy_process_group()
                control = engine.control = None
                gc.collect()
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
                if torch.cuda.memory_reserved(device) != 0:
                    raise RuntimeError('Rank still owns Torch device allocations after parking')
                state = 'parked'
            elif operation == 'restore' and state == 'parked':
                control = groups(sequence)
                engine.control = control
                pipeline.enable_model_cpu_offload(gpu_id=device)
                state = 'ready'
            elif operation == 'execute' and state == 'ready':
                payload = command['payload']
                if set(payload) != {'prompt', 'seed'} or not isinstance(payload['prompt'],str) or not 0 < len(payload['prompt']) <= 2000:
                    raise ValueError('Invalid image request payload')
                if type(payload['seed']) is not int or not 0 <= payload['seed'] < 2**53:
                    raise ValueError('Invalid image seed')
                # CPU control consensus catches mismatched request/session data
                # before either rank starts entering CUDA collectives.
                peers = [None, None]
                dist.all_gather_object(peers, command, group=control)
                if peers[0] != peers[1]:
                    raise ValueError('Rank request identity differs')
                image, elapsed = engine.generate(command['job'], payload['prompt'], payload['seed'])
                if rank == 0:
                    output = Path(config['directory'])/'output.png'
                    image.save(output, format='PNG')
                    if output.stat().st_size > 24*1024**2:
                        raise RuntimeError('Image output exceeds its admitted disk staging cap')
                receipt['request_seconds'] = elapsed
                image = None
            else:
                raise ValueError('Invalid rank lifecycle transition')
            torch.cuda.synchronize(device)
            receipt.update(resident_bytes=torch.cuda.memory_reserved(device), operation_seconds=time.monotonic()-started)
            connection.sendall(encode_rank_message(receipt))
        except BaseException as exc:
            # Never reuse a possibly broken communicator after OOM/peer failure.
            receipt.update(status='error', error=(type(exc).__name__+': '+str(exc))[:512], resident_bytes=0)
            # Preserve the bounded cause even if the controller observes exit
            # before reading the socket acknowledgement. Do not echo payloads.
            print(json.dumps(dict(rank=rank, job=command.get("job"), error=receipt["error"])),
                file=sys.stderr, flush=True)
            try:
                connection.sendall(encode_rank_message(receipt))
            finally:
                os._exit(1)


if __name__ == '__main__':
    main()
