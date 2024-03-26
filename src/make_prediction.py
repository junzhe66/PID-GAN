from einops import rearrange, repeat
import numpy as np
from PIL import Image
import torch
import os
import matplotlib.pyplot as plt
from pysteps.verification.detcatscores import det_cat_fct
from pysteps.verification.detcontscores import det_cont_fct
from pysteps.verification.spatialscores import intensity_scale
from pysteps.visualization import plot_precip_field
from envs.world_model_env import WorldModelEnv
import torch.nn.functional as F
import time
import hydra
import math



@torch.no_grad()
def generate(batch, tokenizer, world_model, latent_dim, horizon, obs_time):
        
    initial_observations = batch
    device = initial_observations.device
    wm_env = WorldModelEnv(tokenizer, world_model, device)
    input_image = initial_observations[:,:obs_time,:,:,:].to(device=device)
    obs_tokens = wm_env.reset_from_initial_observations(input_image)
    generated_sequence  = wm_env.step(obs_tokens, num_steps=horizon*latent_dim)#, top_k=False, sample=False)
    reconstructed_predicted_sequence= wm_env.decode_obs_tokens(generated_sequence)

    return reconstructed_predicted_sequence



def compute_metrics_pre (batch, tokenizer, world_model):
    input_images = batch
    predicted_observations= generate(input_images, tokenizer= tokenizer, world_model= world_model, latent_dim=64, horizon=6, obs_time=3)
    #print("input_images",input_images.size())
    #print("predicted_observations",predicted_observations.size())
    lead_times=6
    # input_images = F.interpolate(input_images, size=(1, 256, 256), mode='trilinear', align_corners=False)
    # predicted_observations= F.interpolate(predicted_observations, size=(1, 256, 256), mode='trilinear', align_corners=False)
    print("input_images",input_images.size())
    print("predicted_observations",predicted_observations.size())                                 
    avg_metrics = {
        'MSE:': 0, 'MAE:': 0, 'PCC:': 0, 'CSI(1mm):': 0, 'CSI(2mm):': 0, 
        'CSI(8mm):': 0, 'ACC(1mm):': 0, 'ACC(2mm):': 0, 'ACC(8mm):': 0, 
        'FSS(1km):': 0, 'FSS(10km):': 0, 'FSS(20km):': 0, 'FSS(30km):': 0
    }

    
    
    for i in range(lead_times):
        input_images_npy = input_images[0,i+3,0,:,:].cpu().numpy()*40
        reconstruction_npy = predicted_observations[0,i+3,0,:,:].cpu().numpy()*40
        scores_cat1 = det_cat_fct(reconstruction_npy, input_images_npy, 1)
        scores_cat2 = det_cat_fct(reconstruction_npy, input_images_npy, 2)
        scores_cat8 = det_cat_fct(reconstruction_npy, input_images_npy, 8)
        scores_cont = det_cont_fct(reconstruction_npy, input_images_npy, scores = ["MSE", "MAE", "corr_p"], thr=0.1)
        scores_spatial = intensity_scale(reconstruction_npy, input_images_npy, 'FSS', 0.1, [1,10,20,30])
        
        metrics = {'MSE:': scores_cont['MSE'],
                   'MAE:': scores_cont['MAE'], 
                   'PCC:': scores_cont['corr_p'], 
                   'CSI(1mm):': scores_cat1['CSI'],
                   'CSI(2mm):': scores_cat2['CSI'],
                   'CSI(8mm):': scores_cat8['CSI'],
                   'ACC(1mm):': scores_cat1['ACC'],
                   'ACC(2mm):': scores_cat2['ACC'],
                   'ACC(8mm):': scores_cat8['ACC'],
                   'FSS(1km):': scores_spatial[0][0],
                   'FSS(10km):': scores_spatial[1][0],
                   'FSS(20km):': scores_spatial[2][0],
                   'FSS(30km):': scores_spatial[3][0]
        }
        
        # Update avg_metrics dictionary
        for key in avg_metrics:
            avg_metrics[key] += metrics[key]
        
    # Compute average for each metric
    for key in avg_metrics:
        avg_metrics[key] = np.around(avg_metrics[key] / lead_times, 3)
    
    return avg_metrics





