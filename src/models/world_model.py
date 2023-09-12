from dataclasses import dataclass
from typing import Any, Optional, Tuple

from einops import rearrange
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import Batch
from .kv_caching import KeysValues
from .slicer import Embedder, Head
from .tokenizer import Tokenizer
from .transformer import Transformer, TransformerConfig
from utils import init_weights, LossWithIntermediateLosses


@dataclass
class WorldModelOutput:
    output_sequence: torch.FloatTensor
    logits_observations: torch.FloatTensor
    logits_ends: torch.FloatTensor
    
class WorldModel(nn.Module):
    def __init__(self, obs_vocab_size: int, config: TransformerConfig) -> None:
        super().__init__()
        self.obs_vocab_size = obs_vocab_size
        self.config = config
        self.transformer = Transformer(config)
        self.sequence_length = 16 
        self.num_steps =  9 
        self.prev_steps = 7 
        obs_tokens_pattern = torch.ones(config.tokens_per_block)
        ends_tokens_pattern = torch.zeros(self.config.tokens_per_block)
        ends_tokens_pattern[-1] = 1
                
        self.pos_emb = nn.Embedding(config.max_tokens, config.embed_dim)
        
        self.embedder = Embedder(
            max_blocks=config.max_blocks,
            block_masks=[obs_tokens_pattern],
            embedding_tables=nn.ModuleList([nn.Embedding(obs_vocab_size, config.embed_dim)])
        )

        self.head_observations = Head(
            max_blocks=config.max_blocks,
            block_mask=obs_tokens_pattern,
            head_module=nn.Sequential(
                nn.Linear(config.embed_dim, config.embed_dim),
                nn.ReLU(),
                nn.Linear(config.embed_dim, obs_vocab_size)
            )
        )
        
        self.head_ends = Head(
            max_blocks=config.max_blocks,
            block_mask=ends_tokens_pattern,
            head_module=nn.Sequential(
                nn.Linear(config.embed_dim, config.embed_dim),
                nn.ReLU(),
                nn.Linear(config.embed_dim, 2)
            )
        )
        self.apply(init_weights)
           

    def __repr__(self) -> str:
        return "world_model"
            
    def forward(self, obs_tokens: torch.LongTensor, past_keys_values: Optional[KeysValues] = None) -> WorldModelOutput:

        #obs_tokens = obs_tokens.to('cuda:0')
        num_steps = obs_tokens.size(1)  # (B, T)
        print("number of steps",num_steps)
        prev_steps = 0 if past_keys_values is None else past_keys_values.size
        print("prev_steps",prev_steps)
        sequences = self.embedder(obs_tokens, num_steps, prev_steps) + self.pos_emb(prev_steps + torch.arange(num_steps, device=obs_tokens.device))
        print("sequences ", sequences.size())

        x = self.transformer(sequences, past_keys_values)
        logits_observations = self.head_observations(x, num_steps=num_steps, prev_steps=prev_steps)
        logits_ends = self.head_ends(x, num_steps=num_steps, prev_steps=prev_steps)
        print("x world", x.size())
        print("logit_obs", logits_observations.size())
        print("logit_end", logits_ends.size())
        return WorldModelOutput(x, logits_observations, logits_ends)
    
    def compute_loss(self, batch: Batch, tokenizer: Tokenizer, **kwargs: Any) -> LossWithIntermediateLosses:
        
        with torch.no_grad():
            observations= rearrange(batch['observations'], 'b c t h w  -> (b t) c h w')
            obs_tokens = tokenizer.encode(observations).tokens  # (BL, K)
        obs_tokens=obs_tokens.unsqueeze(0)
        print("tokens_obs",obs_tokens.size())
        tokens = rearrange(obs_tokens, 'b l k1 -> b (l k1)')  # (B, L(K+1))
        print(tokens.dtype)
        print("reshape.tokens",tokens.size())
        outputs = self.forward(tokens)
        
        print("output_sequence size:", outputs.output_sequence.size())
        print("logits_observations size:", outputs.logits_observations.size())
        print("logits_ends size:", outputs.logits_ends.size())
        #labels_observations, labels_ends = self.compute_labels_world_model(obs_tokens, batch['ends'], batch['mask_padding'])

        #logits_observations = rearrange(outputs.logits_observations[:, :-1], 'b t o -> (b t) o')
        #loss_obs = F.cross_entropy(logits_observations,labels_observations)
        #loss_ends = F.cross_entropy(rearrange(outputs.logits_ends, 'b t e -> (b t) e'), labels_ends)
        #print("Cross entropy Losses", #loss_obs)
        return outputs.output_sequence, outputs.logits_observations, outputs.logits_ends
        #return LossWithIntermediateLosses(loss_obs=loss_obs,loss_ends=loss_ends)
    
    def compute_labels_world_model(self, obs_tokens: torch.Tensor, ends: torch.Tensor, mask_padding: torch.BoolTensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert torch.all(ends.sum(dim=1) <= 1)  # at most 1 done
        mask_fill = torch.logical_not(mask_padding)
        labels_observations = rearrange(obs_tokens.masked_fill(mask_fill.unsqueeze(-1).expand_as(obs_tokens), -100), 'b t k -> b (t k)')[:, 1:]
        labels_ends = ends.masked_fill(mask_fill, -100)
        return labels_observations.reshape(-1), labels_ends.reshape(-1)
