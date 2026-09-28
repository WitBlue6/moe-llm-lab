from pathlib import Path
import copy
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from moe_llm.siglip import SiglipConfig, ContrastiveSiglip, sigmoid_pair_loss
from moe_llm.siglip_training import distributed_loss


def worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2)
    try:
        torch.manual_seed(17)
        c = SiglipConfig(image_size=8, patch_size=4, hidden_size=8, intermediate_size=16,
                        num_hidden_layers=1, num_attention_heads=2)
        local = ContrastiveSiglip(c, 32, text_length=8, projection_size=4, text_layers=1)
        reference = copy.deepcopy(local)
        pixels = torch.randn(4, 3, 8, 8)
        tokens = torch.tensor([[1, 7, 2], [1, 8, 2], [1, 9, 2], [1, 10, 2]])
        mask = torch.ones_like(tokens, dtype=torch.bool)
        expected = sigmoid_pair_loss(*reference(pixels, tokens, mask))
        expected.backward()
        model = DDP(local)
        sl = slice(rank * 2, (rank + 1) * 2)
        actual = distributed_loss(model(pixels[sl], tokens[sl], mask[sl]), rank, 2)
        actual.backward()
        mean = actual.detach().clone()
        dist.all_reduce(mean); mean /= 2
        torch.testing.assert_close(mean, expected.detach(), atol=2e-6, rtol=2e-5)
        for (name, a), (_, b) in zip(local.named_parameters(), reference.named_parameters()):
            assert a.grad is not None and b.grad is not None, name
            torch.testing.assert_close(a.grad, b.grad, atol=3e-6, rtol=5e-4, msg=name)
    finally:
        dist.destroy_process_group()


def test_distributed_negatives_match_global_batch_gradients(tmp_path):
    torch.multiprocessing.spawn(worker, args=((tmp_path / 'rendezvous').as_uri(),), nprocs=2, join=True)
