import math
from typing import List

import torch
import torch.nn as nn


class Slicer(nn.Module):
    def __init__(self, max_blocks: int, block_mask: torch.Tensor) -> None:
        super().__init__()
        self.block_size = block_mask.size(0)
        #print("Block Size", self.block_size)
        self.num_kept_tokens = block_mask.sum().long().item()
        #print("number_kept_tokens", self.num_kept_tokens)
        kept_indices = torch.where(block_mask)[0].repeat(max_blocks)
        #print("Kept Indices", kept_indices.size())
        offsets = torch.arange(max_blocks).repeat_interleave(self.num_kept_tokens)
        self.register_buffer('indices', kept_indices + block_mask.size(0) * offsets)
        #print("BUFFER", self.register_buffer('indices', kept_indices + block_mask.size(0) * offsets))

    def compute_slice(self, num_steps: int, prev_steps: int = 0) -> torch.Tensor:
        total_steps = num_steps + prev_steps
        num_blocks = math.ceil(total_steps / self.block_size)
        indices = self.indices[:num_blocks * self.num_kept_tokens]
        #print("indices", indices.size())
        return indices[torch.logical_and(prev_steps <= indices, indices < total_steps)] - prev_steps


    def forward(self, *args, **kwargs):
        raise NotImplementedError


class Head(Slicer):
    def __init__(self, max_blocks: int, block_mask: torch.Tensor, head_module: nn.Module) -> None:
        super().__init__(max_blocks, block_mask)
        assert isinstance(head_module, nn.Module)
        self.head_module = head_module

    def forward(self, x: torch.Tensor, num_steps: int, prev_steps: int) -> torch.Tensor:
        x_sliced = x[:, self.compute_slice(num_steps, prev_steps)]  # x is (B, T, E)
        return self.head_module(x_sliced)

## I just commwnt the assert as I dont have more than 1 embeeding table I have only observations no rewards or actions 
class Embedder(nn.Module):
    def __init__(self, max_blocks: int, block_masks: List[torch.Tensor], embedding_tables: List[nn.Embedding]) -> None:
        super().__init__()
        #print("Block Mask, embedd",len(block_masks), embedding_tables[0].weight.size())
        #print("Length block mask", len(block_masks))
        #print("Length of embedding tables",len(embedding_tables))
        assert len(block_masks) == len(embedding_tables)
        #assert (sum(block_masks) == 1).all()  # block mask are a partition of a block
        self.embedding_dim = embedding_tables[0].embedding_dim
        assert all([e.embedding_dim == self.embedding_dim for e in embedding_tables])
        self.embedding_tables = embedding_tables
        self.slicers = [Slicer(max_blocks, block_mask) for block_mask in block_masks]

    def forward(self, tokens: torch.Tensor, num_steps: int, prev_steps: int) -> torch.Tensor:
        assert tokens.ndim == 2  # x is (B, T)
        output = torch.zeros(*tokens.size(), self.embedding_dim, device=tokens.device) 
        #print("OUTPUT in EMBEDDER", output.size())
        #for slicer in self.slicers:
            #print("Slicer block_size:", slicer.block_size)
            #print("Slicer num_kept_tokens:", slicer.num_kept_tokens)
        for slicer, emb in zip(self.slicers, self.embedding_tables):
           
            s = slicer.compute_slice(num_steps, prev_steps)
            output[:, s] = emb(tokens[:, s])
            #print("slicer output", output[:,s].shape)
        return output
