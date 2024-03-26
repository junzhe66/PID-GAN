import random
from typing import List, Optional, Union

import gym
from einops import rearrange
import numpy as np
from PIL import Image
import torch
from torch.distributions.categorical import Categorical
import torchvision
from transformers import top_k_top_p_filtering

class WorldModelEnv:

    def __init__(self, tokenizer: torch.nn.Module, world_model: torch.nn.Module, device: Union[str, torch.device], env: Optional[gym.Env] = None) -> None:

        self.device = torch.device(device)
        self.world_model = world_model.to(self.device).eval()
        self.tokenizer = tokenizer.to(self.device).eval()

        self.keys_values_wm, self.obs_tokens, self._num_observations_tokens = None, None, None

        self.env = env

    @property
    def num_observations_tokens(self) -> int:
        return self._num_observations_tokens

    @torch.no_grad()
    def reset_from_initial_observations(self, observations: torch.FloatTensor) -> torch.FloatTensor:
        obs_tokens = self.tokenizer.encode(observations, should_preprocess=True).tokens    # (B, C, H, W) -> (B, K)
        obs_tokens=rearrange(obs_tokens,'B T H -> B (T H)')
        self.obs_tokens = obs_tokens

        return obs_tokens

    @torch.no_grad()
    def step(self , observations, num_steps) -> None:
        sample=observations
        cond_len = observations.shape[1]
       # print('Initial condition length:', cond_len)
        past = None
        x=sample
        
        

        for k in range(num_steps):
          #  print(f"Step {k}!!!!!!!!!!")
            outputs_wm = self.world_model.forward_with_past(x, past, past_length = (k+cond_len-1))
          #  print(f"Step {k}: logits_observations shape:", outputs_wm.logits_observations.shape)


            if past is None:
                past = [outputs_wm.past_keys_values]
              #  print('Initial past keys values set.', past[0].shape)
              #  print(len(past))
            else:
                past.append(outputs_wm.past_keys_values)
             #   print(f"Step {k}: Updated past keys values length:", past[k].shape)
             #   print(len(past))

            logits = outputs_wm.logits_observations
           # print(f"Step {k}: Raw Logits shape:", logits.shape)
            logits=logits[:, -1, :]
            #print(f"Step {k}: Logits shape:", logits.shape)
            
            #logits = top_k_top_p_filtering(logits, top_k=100, top_p=0.9)
            token = Categorical(logits=logits).sample()
            #print(f"Step {k}: Sampled token:", token)

            x = token.unsqueeze(1) 
            #print(f"Step {k}: x shape after unsqueeze:", x.shape)
            sample = torch.cat((sample, x), dim=1)
           # print(f"Step {k}: Sample shape after concatenation:", sample.shape)

        

        sample = sample[:, :] 
       # print("Final sample shape:", sample.shape)


        return sample 


    @torch.no_grad()
    def decode_obs_tokens(self, obs_tokens) -> List[Image.Image]:
        generated_sequence=obs_tokens
        generated_sequence=generated_sequence.squeeze(0)
        embedded_tokens = self.tokenizer.embedding(generated_sequence)     # (B, K, E)
        z = rearrange(embedded_tokens, '(b h w) e -> b e h w', e=2048, h=8, w=8).contiguous()
        rec = self.tokenizer.decode(z, should_postprocess=True)         # (B, C, H, W)
        rec= rec.unsqueeze(0)
        return rec
