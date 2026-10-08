"""Smoke-check the launcher's network-none rendezvous without GPU exposure."""
from datetime import timedelta
import json

import torch
import torch.distributed as dist


def main():
    dist.init_process_group('gloo', timeout=timedelta(seconds=10))
    try:
        assert dist.get_world_size() == 2
        value = torch.tensor([dist.get_rank()+1])
        dist.all_reduce(value)
        assert value.item() == 3
        print(json.dumps(dict(rank=dist.get_rank(), sum=value.item(), rendezvous='127.0.0.1')))
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
