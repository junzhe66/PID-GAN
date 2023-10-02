from einops import rearrange
import numpy as np
from PIL import Image
import torch
import os
import matplotlib.pyplot as plt
from pysteps.verification.detcatscores import det_cat_fct
from pysteps.verification.detcontscores import det_cont_fct
from pysteps.verification.spatialscores import intensity_scale
from pysteps.visualization import plot_precip_field


@torch.no_grad()
def make_reconstructions_from_batch(batch, save_dir, epoch, tokenizer):
    #check_batch(batch)

    original_frames = rearrange(batch['observations'], 'c t b h w  -> (b t) c h w')
    batch_tokenizer = batch['observations']

    rec_frames = generate_reconstructions_with_tokenizer(batch_tokenizer, tokenizer)
    
    os.makedirs(save_dir, exist_ok=True)

    for i in range(5):
        original_frame = original_frames[i,0,:,:]
        a_display = tensor_to_np_frames(original_frame)
        rec_frame = rec_frames[i,0,:,:]
        ar_display = tensor_to_np_frames(rec_frame)

        # Plot the precipitation fields using your plot_precip_field function
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plot_precip_field(a_display, title="Input")
        
        plt.subplot(1, 2, 2)
        plot_precip_field(ar_display, title="Reconstruction")

        plt.savefig(os.path.join(save_dir, f'epoch_{epoch:03d}_t_{i:03d}.png'))

        # Optionally, display the figure if needed
        plt.show()

        # Close the figure to free up resources
        plt.close()


    return


# def check_batch(batch):
    assert sorted(batch.keys()) == ['actions', 'ends', 'mask_padding', 'observations', 'rewards']
    b, t, _, _, _ = batch['observations'].shape  # (B, T, C, H, W)
    assert batch['actions'].shape == batch['rewards'].shape == batch['ends'].shape == batch['mask_padding'].shape == (b, t)


def tensor_to_np_frames(inputs):
    #check_float_btw_0_1(inputs)
    return inputs.cpu().numpy()*40

# 
# def check_float_btw_0_1(inputs):
    # assert inputs.is_floating_point() and (inputs >= 0).all() and (inputs <= 1).all()


@torch.no_grad()
def generate_reconstructions_with_tokenizer(batch, tokenizer):
    #check_batch(batch)
    inputs = rearrange(batch, 'c t b h w  -> (b t) c h w')
    outputs = reconstruct_through_tokenizer(inputs, tokenizer)
    b, t, _, _, _ = batch.size()
    # outputs = rearrange(outputs, '(b t) c h w -> b t h w c', b=b, t=t)
    rec_frames = outputs
    return rec_frames


@torch.no_grad()
def reconstruct_through_tokenizer(inputs, tokenizer):
    #check_float_btw_0_1(inputs)
    reconstructions = tokenizer.encode_decode(inputs, should_preprocess=True, should_postprocess=True)
    return torch.clamp(reconstructions, 0, 1)
