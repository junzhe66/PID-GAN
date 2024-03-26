from pathlib import Path

import torch
from torch.distributions.categorical import Categorical
import torch.nn as nn

from models.tokenizer import Tokenizer
from models.world_model import WorldModel
from utils import extract_state_dict
from models.tokenizer import NLayerDiscriminator


class Agent(nn.Module):
    def __init__(self, tokenizer: Tokenizer, world_model: WorldModel,discriminator: NLayerDiscriminator,discriminator_AENN):
        super().__init__()
        self.tokenizer = tokenizer
        self.world_model = world_model
        self.discriminator = discriminator
        self.discriminator_AENN = world_model.head_discriminator

    def load(self, path_to_checkpoint: Path, path_to_checkpoint_trans: Path, device: torch.device, load_tokenizer: bool = True, load_world_model: bool = True,load_discriminator: bool = True, load_discriminator_AENN: bool = True,) -> None: 
        agent_state_dict = torch.load(path_to_checkpoint, device)
        if path_to_checkpoint_trans is not None:
            agent_state_dict_trans = torch.load(str(path_to_checkpoint_trans), map_location=device)
            #print("Transformer checkpoint keys:", agent_state_dict_trans.keys())
        if load_tokenizer:
            self.tokenizer.load_state_dict(extract_state_dict(agent_state_dict, 'tokenizer'))
            print("Loading the tokenizer successfully")
        if load_world_model:
            self.world_model.load_state_dict(extract_state_dict(agent_state_dict_trans, 'world_model'))
            print("Loading the world_model successfully")
        if load_discriminator:
           self.discriminator.load_state_dict(extract_state_dict(agent_state_dict, 'discriminator'))
           print("Loading the discriminator successfully")
        if load_discriminator_AENN:
           self.discriminator_AENN.load_state_dict(extract_state_dict(agent_state_dict_trans, 'discriminator_AENN'))
           print("Loading the AENN_discriminator successfully")

