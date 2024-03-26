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
from torch.distributions.categorical import Categorical
from .Discriminator import DiscriminatorAENN

@dataclass
class WorldModelOutput:
    output_sequence: torch.FloatTensor
    logits_observations: torch.FloatTensor
    past_keys_values: torch.tensor

class WorldModel(nn.Module):
    def __init__(self, obs_vocab_size: int, config: TransformerConfig) -> None:
        super().__init__()
        self.obs_vocab_size = obs_vocab_size
        self.config = config
        self.transformer = Transformer(config)
        obs_tokens_pattern = torch.ones(config.tokens_per_block)
  
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

        self.head_discriminator = DiscriminatorAENN(relu_alpha=0.2, batch_norm=True, drop_out=True)
        self.apply(init_weights)
           

    def __repr__(self) -> str:
        return "world_model"
            
    def forward(self, obs_tokens: torch.LongTensor, past_keys_values: Optional[KeysValues] = None) -> WorldModelOutput:

        num_steps = obs_tokens.shape[1]  # (B, T)
        assert num_steps <= self.config.max_tokens
        prev_steps = 0 if past_keys_values is None else past_keys_values.size

        sequences = self.embedder(obs_tokens, num_steps, prev_steps) + self.pos_emb(prev_steps + torch.arange(num_steps, device=obs_tokens.device))

        x = self.transformer.forward(sequences, past_keys_values)
        logits_observations = self.head_observations(x, num_steps=num_steps, prev_steps=prev_steps)
        return WorldModelOutput(x, logits_observations, past_keys_values)
    
    
    def forward_with_past(self, obs_tokens: torch.LongTensor, past_keys_values=None, past_length = None) -> WorldModelOutput:
        # inference only
        assert not self.training
        num_steps = obs_tokens.shape[1]  # (B, T)
        #print("Number of steps:", num_steps)
        
        if past_keys_values is not None:
            assert past_length is not None
            past_keys_values= torch.cat(past_keys_values, dim=-2) 
            past_shape = list(past_keys_values.shape)
            expected_shape = [self.config.num_layers, 2, obs_tokens.shape[0], self.config.num_heads, past_length, self.config.embed_dim//self.config.num_heads]
            assert past_shape == expected_shape, f"{past_shape} =/= {expected_shape}"
            #print("size of last past key", past_keys_values.shape)
        else:
            past_length=0
        #print("Number of past steps:", past_length)
        a = self.embedder(obs_tokens, num_steps, past_length)
        #print("embedder shape",a.shape)
        b =  self.pos_emb(past_length + torch.arange(num_steps, device=obs_tokens.device))
       # print("Poisition embedder shape",b.shape)
        sequences = a + b 
        #print("Sequences shape:", sequences.size())

        x, past_keys_values = self.transformer.forward_with_past(sequences, past_keys_values)
       # print("Output after transformer shape:", x.size())
        #print("Past keys values after transformer:", past_keys_values.shape)
        logits_observations = self.head_observations(x, num_steps=num_steps, prev_steps=past_length)
      #  print("Logits observations shape:", logits_observations.size())

        return WorldModelOutput(x, logits_observations, past_keys_values)
    




    def compute_loss(self, batch: Batch, phy_data: Batch, tokenizer: Tokenizer, train_world_model: bool, **kwargs: Any) -> LossWithIntermediateLosses:
        
        with torch.no_grad():
            observations= rearrange(batch, 'b t c h w  -> (b t) c h w')
            obs_tokens = tokenizer.encode(observations, should_preprocess=True).tokens  # (BL, K)
            shape_obs = batch.size()
            shape_token= obs_tokens.size()

        b = shape_obs[0]
        l = shape_obs[1]
        k = shape_token[1]
        
        tokens = obs_tokens.view(b, l*k) # (B, L(K))
        #print("obs_token",tokens.size())
        
        labels_observations = self.compute_labels_world_model(tokens)
        #print("labels obs", labels_observations.size())
        outputs = self.forward(tokens)
        #print("output_sequence size:", outputs.output_sequence.size())
        #print("output_logits size:", outputs.logits_observations.size())
        
        logits_observations_image = rearrange(outputs.logits_observations[:, 63:-1], 'b t o -> (b t) o')
        #print("Logits", logits_observations_image.size())
        token = Categorical(logits=logits_observations_image).sample()
        #print(token.size())
        
        with torch.no_grad():
            z_q=tokenizer.embedding(token.long())
            z_q = rearrange(z_q, '(b t h w) e -> b t e h w',t=8, e=2048, h=8, w=8).contiguous()
            reco = tokenizer.decode(z_q,should_postprocess=True)
            #print(reco.size())

            reco = F.interpolate(reco, size=(1, 256, 256), mode='trilinear', align_corners=False)
            #print("reco",reco.size())
            
            image = F.interpolate(batch, size=(1, 256, 256), mode='trilinear', align_corners=False)
            #print("image",image.size())
            
        radar_data_gen=reco*40
        lambda_val=0.1
        #print("phy",phy_data.shape)
        final=phy_data-radar_data_gen
        #print(final.shape)
    
        prob_f=self.expo_transformation(lambda_val, final)
        #print("probF",prob_f.size())
        fake_logits = self.head_discriminator(torch.cat([radar_data_gen, prob_f], dim=1))
        #print("fake_logits",fake_logits.shape)

        image1=image[:,1:,:,:,:]*40
        final2=phy_data-image1
        #print(final2.shape)
    
        prob_R=self.expo_transformation(lambda_val, final2)
       # print("probR",prob_f.size())
        
        real_logits = self.head_discriminator(torch.cat([image1, prob_R], dim=1))
        #print("real_logits",real_logits.shape)
    
        if train_world_model == True:

          
            logits_observations = rearrange(outputs.logits_observations[:, :-1], 'b t o -> (b t) o')
            loss_obs = F.cross_entropy(logits_observations, labels_observations)
            #print(loss_obs)
            adv_loss = self.generator_loss(fake_logits) 
            #print(adv_loss)
            return LossWithIntermediateLosses(loss_obs=loss_obs,adv_loss=adv_loss)
        else:  

            d_loss = self.discriminator_loss(real_logits, fake_logits)
                
            return LossWithIntermediateLosses(d_loss=d_loss)
        
        
    def discriminator_loss(self, logits_real_u, logits_fake_u):
    #   def discriminator_loss(self, logits_real_u, logits_fake_u, logits_fake_f):
        loss =  - torch.mean(torch.log(1.0 - torch.sigmoid(logits_real_u) + 1e-8) + torch.log(torch.sigmoid(logits_fake_u) + 1e-8)) 
           #- torch.mean(torch.log(torch.sigmoid(logits_fake_f) + 1e-8))
        return loss
    
    
    def generator_loss(self, logits_fake_u):
        gen_loss = torch.mean(logits_fake_u) 
        return gen_loss
    
    
    def compute_labels_world_model(self, obs_tokens: torch.Tensor):
        labels_observations = obs_tokens[:, 1:] # obs tokens from t to t_end, remove only the first one 
        return labels_observations.reshape(-1)
    
    def expo_transformation(self, lambda_phy, phyloss):
        probs = torch.exp(-lambda_phy * torch.abs(phyloss))
        return probs
    

