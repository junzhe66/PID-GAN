import random
import sys
from typing import List, Optional, Union

from einops import rearrange
import numpy as np
import torch
from tqdm import tqdm
import wandb

from agent import Agent
from dataset import EpisodesDataset
from envs import SingleProcessEnv, MultiProcessEnv
from episode import Episode
from utils import EpisodeDirManager
from PIL import Image
import torch.nn.functional as F

import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from torch.cuda.amp import autocast
from torch.autograd import Variable
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import h5py
import torch
import numpy as np
from torchvision.transforms import ToTensor, Compose, CenterCrop







class Collector: 

    def __init__(self, env: Union[SingleProcessEnv, MultiProcessEnv], dataset: EpisodesDataset, episode_dir_manager: EpisodeDirManager, obs_time, pred_time, time_interval):
        self.training_data = []
        self.testing_data = []
        self.eval_data = []
        self.env = env
        self.dataset = dataset
        self.episode_dir_manager = episode_dir_manager
        self.obs = self.env.reset()
        self.episode_ids = [None] * self.env.num_envs
        self.i=0
        self.batch_counter = 0
        self.batch_counter_10 = 0
        self.obs_time = obs_time
        self.pred_time = pred_time
        self.time_interval = time_interval 
        self.length = self.pred_time + self.obs_time 




    def collect_data(self): 
        root_dir = '/home/hbi/RAD_NL25_RAP_5min/' 
    

        df_train = pd.read_csv('/space/zboucher/World_Model/catchment/training_Delfland08-14_20.csv', header=None)
        event_times = df_train[0].to_list()
        dataset_train = radarDataset(root_dir, event_times, self.obs_time, self.pred_time, self.time_interval, transform=Compose([ToTensor()]))

        df_train_s = pd.read_csv('/space/zboucher/World_Model/catchment/training_Delfland08-14.csv', header=None)
        event_times = df_train_s[0].to_list()
        dataset_train_del = radarDataset(root_dir, event_times, self.obs_time, self.pred_time, self.time_interval, transform=Compose([ToTensor()]))

        df_test = pd.read_csv('/space/zboucher/World_Model/catchment/testing_Delfland18-20.csv', header=None)
        event_times = df_test[0].to_list()
        dataset_test = radarDataset(root_dir, event_times,self.obs_time, self.pred_time, self.time_interval, transform=Compose([ToTensor()]))

        df_vali = pd.read_csv('/space/zboucher/World_Model/catchment/validation_Delfland15-17.csv', header=None)
        event_times = df_vali[0].to_list()
        dataset_vali = radarDataset(root_dir, event_times, self.obs_time, self.pred_time, self.time_interval, transform=Compose([ToTensor()]))

        df_train_aa = pd.read_csv('/space/zboucher/World_Model/catchment/training_Aa08-14.csv', header=None)
        event_times = df_train_aa[0].to_list()
        dataset_train_aa = radarDataset(root_dir, event_times,self.obs_time, self.pred_time, self.time_interval, transform=Compose([ToTensor()]))

        df_train_dw = pd.read_csv('/space/zboucher/World_Model/catchment/training_Dwar08-14.csv', header=None)
        event_times = df_train_dw[0].to_list()
        dataset_train_dw = radarDataset(root_dir, event_times, self.obs_time, self.pred_time, self.time_interval, transform=Compose([ToTensor()]))

        df_train_re = pd.read_csv('/space/zboucher/World_Model/catchment/training_Regge08-14.csv', header=None)
        event_times = df_train_re[0].to_list()
        dataset_train_re = radarDataset(root_dir, event_times, self.obs_time, self.pred_time, self.time_interval, transform=Compose([ToTensor()]))

        data_list = [dataset_train_aa, dataset_train_dw, dataset_train_del, dataset_train_re]
        train_aadedwre = torch.utils.data.ConcatDataset(data_list)

        print(len(dataset_train), len(dataset_test), len(dataset_vali))
        loaders = {'train': DataLoader(train_aadedwre, batch_size=1, shuffle=True, num_workers=8),
                    'test': DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=8),
                    'valid': DataLoader(dataset_vali, batch_size=1, shuffle=False, num_workers=8),

                    'train_aa5': DataLoader(dataset_train_aa, batch_size=1, shuffle=False, num_workers=8),
                    'train_dw5': DataLoader(dataset_train_dw, batch_size=1, shuffle=False, num_workers=8),
                    'train_del5': DataLoader(dataset_train_del, batch_size=1, shuffle=True, num_workers=8),
                    'train_re5': DataLoader(dataset_train_re, batch_size=1, shuffle=False, num_workers=8),}
        return loaders, len(dataset_train), len(dataset_test), len(dataset_vali)
    
    def collect_training_data(self):
        loaders, length_train, _ , _ = self.collect_data()
        loaders_train = loaders['train']
        length = length_train 

        return loaders_train, length
    
    def collect_testing_data(self):
        loaders, _ , length_test, _ = self.collect_data()
        loaders_test = loaders['test']
        length = length_test 
        return loaders_test, length
    
    def collect_validation_data(self):
        loaders, _ , _ , length_validation = self.collect_data()
        loaders_validation = loaders['valid']
        length = length_validation
        return loaders_validation, length
    
    
            
    def get_next_batch(self,epoch, batch_size, start_index, training_data):
        self.training_data= training_data
        self.batch_size= batch_size 
        self.start_index= start_index
        end_index= start_index + batch_size
        self.epoch= epoch
        training_batch=[]
        
        for i, images in enumerate(training_data, start= start_index):
            if i >= end_index:
                break
            image = images.view(self.length,1,128,128)
            # image= image.unsqueeze(-1)
            training_batch.append(image)

        batch = training_batch
        
        current_index = end_index
        self.episode_ids = [None] * self.env.num_envs
        self.process_and_add_episodes(batch, batch_size, epoch, len(training_data))
            
        return batch, current_index 

    def process_and_add_episodes(self, batch, batch_size, epoch, length):
        # Assuming you have the logic to process episodes here
        # Extract observations, actions, rewards, dones, etc.
        batch_size = batch_size
        length_data = length 
        if self.batch_counter_10 < 10:
            episode_tensor = torch.cat(batch, dim=1)
            # for o in observations:
            #o = torch.tensor(batch)
            episode = Episode(
                observations=episode_tensor
                )
            if self.episode_ids[self.batch_counter_10] is None:
                self.episode_ids[self.batch_counter_10] = self.dataset.add_episode(episode)
                self.episode_dir_manager.save(episode, self.batch_counter, epoch)
                
            # else:
            #     self.dataset.update_episode(self.episode_ids[index], episode)
            #            self.first_10_batch_counter += 1
        
        # Reset the first_10_batch_counter to 0 if it exceeds 10
            if self.batch_counter_10>= 10:
                self.batch_counter_10= 0
                
                # Increment the batch_counter for the whole batches
            self.batch_counter += 1
            if self.batch_counter == (length_data/batch_size): 
                self.batch_counter=0
            


# Correct

class radarDataset(Dataset):
    def __init__(self, root_dir, event_times, obs_number, pred_number, time_interval, transform=None):
        # event_times is an array of starting time t(string)
        # transform is the preprocessing functions
        self.root_dir = root_dir
        self.transform = transform
        self.event_times = event_times
        self.obs_number = obs_number
        self.pred_number = pred_number
        self.time_interval = time_interval
    def __len__(self):
        return len(self.event_times)
    def __getitem__(self, idx):
        start_time = str(self.event_times[idx])
        time_list_pre, time_list_obs = eventGeneration(start_time, self.obs_number, self.pred_number, self.time_interval)
        output = []
        time_list = time_list_obs + time_list_pre
        #print(time_list)
        for time in time_list:
            year = time[0:4]
            month = time[4:6]
            #path = self.root_dir + year + '/' + month + '/' + 'RAD_NL25_RAC_MFBS_EM_5min_' + time + '_NL.h5'
            path = self.root_dir + year + '/' + month + '/' + 'RAD_NL25_RAP_5min_' + time + '.h5'
            image = np.array(h5py.File(path)['image1']['image_data'])
            #image = np.ma.masked_where(image == 65535, image)
            image = image[264:520,242:498]
            image[image == 65535] = 0
            image = image.astype('float32')
            image = image/100*12
            image = np.clip(image, 0, 128)
            image = image/40
            #image = 2*image-1 #normalize to [-1,1]
            output.append(image)
        output = torch.tensor((np.array(output)), dtype = torch.float32)
 # Reshape tensor to include a batch dimension (B, C, H, W)
        output = output.unsqueeze(0)
        # Resize tensor using F.interpolate
        output = F.interpolate(output, size=(128, 128), mode='bilinear', align_corners=False)
        # Remove batch dimension
        output = torch.squeeze(output, 0)
        # Permute and transform
        output = torch.permute(output, (1, 2, 0))
        output = self.transform(np.array(output))
        return output
       
       
        

def eventGeneration(start_time, obs_time ,lead_time, time_interval):
# Generate event based on starting time point, return a list: [[t-4,...,t-1,t], [t+1,...,t+72]]
# Get the start year, month, day, hour, minute
    year = int(start_time[0:4])
    month = int(start_time[4:6])
    day = int(start_time[6:8])
    hour = int(start_time[8:10])
    minute = int(start_time[10:12])
    #print(datetime(year=year, month=month, day=day, hour=hour, minute=minute))
    times = [(datetime(year, month, day, hour, minute) + timedelta(minutes=time_interval * (x+1))) for x in range(lead_time)]
    lead = [dt.strftime('%Y%m%d%H%M') for dt in times]
    times = [(datetime(year, month, day, hour, minute) - timedelta(minutes=time_interval * x)) for x in range(obs_time)]
    obs = [dt.strftime('%Y%m%d%H%M') for dt in times]
    obs.reverse()
    return lead, obs

    



# import random
# import sys
# from typing import List, Optional, Union

# from einops import rearrange
# import numpy as np
# import torch
# from tqdm import tqdm
# import wandb

# from agent import Agent
# from dataset import EpisodesDataset
# from envs import SingleProcessEnv, MultiProcessEnv
# from episode import Episode
# from utils import EpisodeDirManager, RandomHeuristic


# class Collector:
#     def __init__(self, env: Union[SingleProcessEnv, MultiProcessEnv], dataset: EpisodesDataset, episode_dir_manager: EpisodeDirManager) -> None:
#         self.env = env
#         self.dataset = dataset
#         self.episode_dir_manager = episode_dir_manager
#         self.obs = self.env.reset()
#         self.episode_ids = [None] * self.env.num_envs
#         self.heuristic = RandomHeuristic(self.env.num_actions)

#     @torch.no_grad()
#     def collect(self, agent: Agent, epoch: int, epsilon: float, should_sample: bool, temperature: float, burn_in: int, *, num_steps: Optional[int] = None, num_episodes: Optional[int] = None):
#         assert self.env.num_actions == agent.world_model.act_vocab_size
#         assert 0 <= epsilon <= 1

#         assert (num_steps is None) != (num_episodes is None)
#         should_stop = lambda steps, episodes: steps >= num_steps if num_steps is not None else episodes >= num_episodes

#         to_log = []
#         steps, episodes = 0, 0
#         returns = []
#         observations, actions, rewards, dones = [], [], [], []

#         burnin_obs_rec, mask_padding = None, None
#         if set(self.episode_ids) != {None} and burn_in > 0:
#             current_episodes = [self.dataset.get_episode(episode_id) for episode_id in self.episode_ids]
#             segmented_episodes = [episode.segment(start=len(episode) - burn_in, stop=len(episode), should_pad=True) for episode in current_episodes]
#             mask_padding = torch.stack([episode.mask_padding for episode in segmented_episodes], dim=0).to(agent.device)
#             burnin_obs = torch.stack([episode.observations for episode in segmented_episodes], dim=0).float().div(255).to(agent.device)
#             burnin_obs_rec = torch.clamp(agent.tokenizer.encode_decode(burnin_obs, should_preprocess=True, should_postprocess=True), 0, 1)

#         agent.actor_critic.reset(n=self.env.num_envs, burnin_observations=burnin_obs_rec, mask_padding=mask_padding)
#         pbar = tqdm(total=num_steps if num_steps is not None else num_episodes, desc=f'Experience collection ({self.dataset.name})', file=sys.stdout)

#         while not should_stop(steps, episodes):

#             observations.append(self.obs)
#             obs = rearrange(torch.FloatTensor(self.obs).div(255), 'n h w c -> n c h w').to(agent.device)
#             act = agent.act(obs, should_sample=should_sample, temperature=temperature).cpu().numpy()

#             if random.random() < epsilon:
#                 act = self.heuristic.act(obs).cpu().numpy()

#             self.obs, reward, done, _ = self.env.step(act)

#             actions.append(act)
#             rewards.append(reward)
#             dones.append(done)

#             new_steps = len(self.env.mask_new_dones)
#             steps += new_steps
#             pbar.update(new_steps if num_steps is not None else 0)

#             # Warning: with EpisodicLifeEnv + MultiProcessEnv, reset is ignored if not a real done.
#             # Thus, segments of experience following a life loss and preceding a general done are discarded.
#             # Not a problem with a SingleProcessEnv.

#             if self.env.should_reset():
#                 self.add_experience_to_dataset(observations, actions, rewards, dones)

#                 new_episodes = self.env.num_envs
#                 episodes += new_episodes
#                 pbar.update(new_episodes if num_episodes is not None else 0)

#                 for episode_id in self.episode_ids:
#                     episode = self.dataset.get_episode(episode_id)
#                     self.episode_dir_manager.save(episode, episode_id, epoch)
#                     metrics_episode = {k: v for k, v in episode.compute_metrics().__dict__.items()}
#                     metrics_episode['episode_num'] = episode_id
#                     metrics_episode['action_histogram'] = wandb.Histogram(np_histogram=np.histogram(episode.actions.numpy(), bins=np.arange(0, self.env.num_actions + 1) - 0.5, density=True))
#                     to_log.append({f'{self.dataset.name}/{k}': v for k, v in metrics_episode.items()})
#                     returns.append(metrics_episode['episode_return'])

#                 self.obs = self.env.reset()
#                 self.episode_ids = [None] * self.env.num_envs
#                 agent.actor_critic.reset(n=self.env.num_envs)
#                 observations, actions, rewards, dones = [], [], [], []

#         # Add incomplete episodes to dataset, and complete them later.
#         if len(observations) > 0:
#             self.add_experience_to_dataset(observations, actions, rewards, dones)

#         agent.actor_critic.clear()

#         metrics_collect = {
#             '#episodes': len(self.dataset),
#             '#steps': sum(map(len, self.dataset.episodes)),
#         }
#         if len(returns) > 0:
#             metrics_collect['return'] = np.mean(returns)
#         metrics_collect = {f'{self.dataset.name}/{k}': v for k, v in metrics_collect.items()}
#         to_log.append(metrics_collect)

#         return to_log

#     def add_experience_to_dataset(self, observations: List[np.ndarray], actions: List[np.ndarray], rewards: List[np.ndarray], dones: List[np.ndarray]) -> None:
#         assert len(observations) == len(actions) == len(rewards) == len(dones)
#         for i, (o, a, r, d) in enumerate(zip(*map(lambda arr: np.swapaxes(arr, 0, 1), [observations, actions, rewards, dones]))):  # Make everything (N, T, ...) instead of (T, N, ...)
#             episode = Episode(
#                 observations=torch.ByteTensor(o).permute(0, 3, 1, 2).contiguous(),  # channel-first
#                 actions=torch.LongTensor(a),
#                 rewards=torch.FloatTensor(r),
#                 ends=torch.LongTensor(d),
#                 mask_padding=torch.ones(d.shape[0], dtype=torch.bool),
#             )
#             if self.episode_ids[i] is None:
#                 self.episode_ids[i] = self.dataset.add_episode(episode)
#             else:
#                 self.dataset.update_episode(self.episode_ids[i], episode)
