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
class WorldModel(nn.Module):
    def __init__(self, obs_vocab_size: int, config: TransformerConfig) -> None:
        super().__init__()
        self.max_blocks = 9
        self.obs_vocab_size = obs_vocab_size
        self.sequence_length = 9 * 256
        self.num_steps =  6 * 256
        self.prev_steps =3 * 256
        self.embed_dim = 256
        self.config = config
        self.transformer = Transformer(config)


        input_block_mask = torch.cat([torch.ones(self.prev_steps), torch.zeros(self.num_steps - self.prev_steps)])

        block_masks = [input_block_mask]
        embedding_table = [nn.Embedding(self.obs_vocab_size, self.embed_dim).to('cuda:1')]
        self.image_embedder = Embedder(self.max_blocks, block_masks, embedding_table)
        self.pos_emb =  nn.Embedding(self.sequence_length, self.embed_dim)
        # Generate dummy values for num_steps and prev_steps

        self.head_observations = Head(
            max_blocks=config.max_blocks,
            block_mask=input_block_mask,
            head_module=nn.Sequential(
                nn.Linear(config.embed_dim, config.embed_dim),
                nn.ReLU(),
                nn.Linear(config.embed_dim, obs_vocab_size)
            )
        )



        self.apply(init_weights)
           

    def __repr__(self) -> str:
        return "world_model"
            
   

    def forward(self, obs_tokens: torch.LongTensor, past_keys_values: Optional[KeysValues] = None) -> torch.FloatTensor:

        obs_tokens = obs_tokens.to('cuda:1')
        num_steps = obs_tokens.size(1)  # (B, T)
        prev_steps = 0 if past_keys_values is None else past_keys_values.size
       

        embedded_output = self.image_embedder(obs_tokens, num_steps, prev_steps).to('cuda:1')
        self.embed = embedded_output

        
        position= self.pos_emb(prev_steps +  torch.arange(num_steps).to('cuda:1'))
        self.context_image = embedded_output + position.unsqueeze(0)
        #print("Context Image", self.context_image.size())
        

        x = self.transformer(self.context_image)
        logits_observations = self.head_observations(x, num_steps=num_steps, prev_steps=prev_steps)
        #print("x world", x.size())
        #print("logit", logits_observations.size())

        return x, logits_observations
    


    def compute_loss(self, batch: Batch,**kwargs: Any) -> LossWithIntermediateLosses:
        

        obs_tokens=batch
        obs_tokens=obs_tokens
        shape=obs_tokens.shape
        #print("Observation token", shape[1])
       
        x, logits_observations = self.forward(obs_tokens)
       # print("logit output", (logits_observations.view(-1,self.obs_vocab_size)).size())
        #logits_observations=x
        #print(logits_observations)

        target_obs = obs_tokens[:, self.prev_steps:shape[1]]
        
        #print("Target observation", target_obs.size())
        
        # Compute the loss between the predicted observations and the real target observations
        #predicted_obs = torch.cat([target_obs[:,:self.prev_steps, :],logits_observations], dim=1)
        predicted_obs=logits_observations.view(-1,self.obs_vocab_size)
        #print("Predicted Obs", predicted_obs.size())
        loss_obs = F.cross_entropy(predicted_obs, (target_obs.view(-1)))
        #print("Cross entropy Losses", loss_obs)

        return LossWithIntermediateLosses(loss_obs=loss_obs)
