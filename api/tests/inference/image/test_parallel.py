"""Two actual Gloo ranks compare original sequence exchange with pinned eager math."""
from datetime import timedelta
import json
import os
from pathlib import Path
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as multiprocessing
from diffusers.models.transformers.transformer_qwenimage21 import (
    QwenImage21KVLayerCache, QwenImage21TransformerBlock,
)

from api.inference.image.parallel import (
    TokenShard, cached_block, compact_prefix, heads_to_sequence, sequence_to_heads,
)


def numerical_rank(rank, rendezvous, output):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + rendezvous,
        rank=rank, world_size=2, timeout=timedelta(seconds=60))
    reports = []
    try:
        # Make every row/head identifiable and check both redistribution axes.
        full = torch.arange(1 * 10 * 4 * 2).reshape(1, 10, 4, 2).float()
        local = full[:, rank * 5:(rank + 1) * 5]
        exchanged = sequence_to_heads((local, local + 100, local + 200))
        for index, actual in enumerate(exchanged):
            torch.testing.assert_close(actual, full[:, :, rank * 2:(rank + 1) * 2] + index * 100, rtol=0, atol=0)
        torch.testing.assert_close(heads_to_sequence(exchanged[0]), local, rtol=0, atol=0)
        for dtype in (torch.float32, torch.bfloat16):
            # Joint prefill27+16 is odd. Cached target17 is also padded unevenly.
            # The final case includes an eager condition-image prefix block.
            for prefix_rows, target_rows, segments in ((27, 16, [(0, 27, True)]),
                    (27, 17, [(0, 27, True)]), (7, 15, [(0, 3, True), (3, 7, False)])):
                torch.manual_seed(2026 + prefix_rows + target_rows)
                block = QwenImage21TransformerBlock(32, 4, 8).to(dtype).eval()
                total = prefix_rows + target_rows
                joint = torch.randn(1, total, 32, dtype=dtype)
                modulation = torch.randn(2, 128, dtype=dtype)
                angles = torch.arange(total).float()[:, None] * torch.tensor([.17, .31, .47, .71])[None]
                rotary = torch.polar(torch.ones_like(angles), angles)
                valid = torch.ones(1, total, dtype=torch.bool)
                valid[:, 1] = False
                valid[:, -2] = False
                if target_rows == 16:
                    valid = None
                target_mask = torch.arange(total) >= prefix_rows
                cache = QwenImage21KVLayerCache()
                with torch.no_grad():
                    block(joint, modulation, rotary_emb=rotary, target_token_mask=target_mask,
                        layer_cache=cache, kv_cache_mode='extract', cache_write_slice=slice(0, prefix_rows),
                        segments=segments, key_valid=valid)
                    hidden = torch.randn(1, target_rows, 32, dtype=dtype)
                    expected = block(hidden, modulation, rotary_emb=rotary[prefix_rows:],
                        target_token_mask=torch.ones(target_rows, dtype=torch.bool),
                        layer_cache=cache, kv_cache_mode='cached', attention_mask=None if valid is None else valid[:, None, None])
                shard = TokenShard(target_rows, rank, 2)
                for mode in ('all_gather', 'ulysses'):
                    prefix = compact_prefix(*cache.get(), mode, rank, 2)
                    before = tuple(value.clone() for value in prefix)
                    actual = cached_block(block, shard.take(hidden), modulation,
                        shard.take(rotary[prefix_rows:], dim=0), prefix, shard, mode=mode, key_valid=valid)
                    gathered = [torch.empty_like(actual) for _ in range(2)]
                    dist.all_gather(gathered, actual)
                    result = torch.cat(gathered, dim=1)[:, :target_rows]
                    tolerance = 2e-6 if dtype == torch.float32 else .02
                    torch.testing.assert_close(result, expected, rtol=tolerance, atol=tolerance)
                    for original, retained in zip(before, prefix):
                        torch.testing.assert_close(original, retained, rtol=0, atol=0)
                        assert retained.untyped_storage().nbytes() == retained.numel() * retained.element_size()
                    reports.append(dict(mode=mode, dtype=str(dtype), prefix_rows=prefix_rows,
                        target_rows=target_rows, max_abs_error=float((result.float()-expected.float()).abs().max()),
                        prefix_bytes=sum(value.untyped_storage().nbytes() for value in prefix)))
        Path(output, f'rank-{rank}.json').write_text(json.dumps(reports, indent=2))
    finally:
        dist.destroy_process_group()


class SequenceParallelTests(unittest.TestCase):
    def test_two_rank_math_order_padding_prefix_and_cached_modulation(self):
        with tempfile.TemporaryDirectory() as root:
            multiprocessing.spawn(numerical_rank, args=(str(Path(root, 'rendezvous')), root), nprocs=2, join=True)
            combined = []
            for rank in range(2):
                rows = json.loads(Path(root, f'rank-{rank}.json').read_text())
                self.assertEqual(len(rows), 12)
                combined.append({"rank": rank, "cases": rows})
            if os.environ.get("KADAN_PARALLEL_TEST_REPORT"):
                Path(os.environ["KADAN_PARALLEL_TEST_REPORT"]).write_text(json.dumps(combined, indent=2))

    def test_partition_preserves_global_order_and_pads_only_tail(self):
        original = torch.arange(7).reshape(1, 7, 1)
        first, second = TokenShard(7, 0, 2), TokenShard(7, 1, 2)
        self.assertEqual(first.take(original).flatten().tolist(), [0, 1, 2, 3])
        self.assertEqual(second.take(original).flatten().tolist(), [4, 5, 6, 0])
        with self.assertRaises(ValueError):
            TokenShard(0, 0, 2)

    def test_real_prefix_ownership_is_compact_and_head_sharded(self):
        key = torch.randn(1, 16411, 32, 128, dtype=torch.bfloat16)[:, :27]
        value = torch.randn_like(key)
        for rank in range(2):
            owned = compact_prefix(key, value, 'ulysses', rank, 2)
            self.assertEqual([v.untyped_storage().nbytes() for v in owned], [110592, 110592])
            self.assertTrue(all(v.storage_offset() == 0 for v in owned))
        owned = compact_prefix(key, value, 'all_gather', 0, 2)
        self.assertEqual([v.untyped_storage().nbytes() for v in owned], [221184, 221184])


if __name__ == '__main__':
    unittest.main()
