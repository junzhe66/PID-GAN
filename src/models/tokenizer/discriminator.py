from typing import Any, Tuple
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import torch.nn as nn
from .util import ActNorm
from utils import LossWithIntermediateLosses


def generate_noise(input, noise_std=0.02):
    if noise_std > 0.0:
        return torch.randn_like(input) * noise_std
    else:
        return torch.zeros_like(input)


def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """
    def __init__(self, input_nc=1, ndf=256, n_layers=3, use_actnorm=False, use_dropout=False, dropout_prob=0.5, noise_std=0.02, disc_loss="vanilla"):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        self.noise_std = noise_std
        if not use_actnorm:
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = ActNorm
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d
        assert disc_loss in ["hinge", "vanilla"]

        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")
        print(f"VQLPIPSWithDiscriminator running with {disc_loss} loss.")
        

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
            if use_dropout:
               sequence += [nn.Dropout(dropout_prob)]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)
        self.apply(weights_init)

    def forward(self, input):
        """Standard forward with noise injection."""
        if self.training and self.noise_std > 0.0:
            noise = generate_noise(input, self.noise_std)
            input = input + noise
        return self.main(input)
    
    def compute_loss(self, batch , tokenizer: 'Tokenizer', **kwargs: Any) -> LossWithIntermediateLosses:
        from .tokenizer import Tokenizer  # Late import inside the method

        observations= rearrange(batch, 'b t c h w  -> (b t) c h w')
        reconstructions = tokenizer.encode_decode(observations, should_preprocess=True, should_postprocess=True)
        
        logits_real = self.forward(observations.contiguous().detach())
        logits_fake = self.forward(reconstructions.contiguous().detach())
        
        d1_loss = self.disc_loss(logits_real, logits_fake)
        
        return LossWithIntermediateLosses(d1_loss=d1_loss)
    

