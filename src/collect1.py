# Define dataset
import random
import sys
from typing import List, Optional, Union

from einops import rearrange
import numpy as np
import torch
from tqdm import tqdm


from agent import Agent
from dataset import EpisodesDataset
from envs import SingleProcessEnv, MultiProcessEnv
from episode import Episode
from utils import EpisodeDirManager

from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from torch.cuda.amp import autocast
from torch.autograd import Variable
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import h5py
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
        # self.root_dir = root_dir
        # self.transform = transform
        self.time_interval = time_interval
        self.obs_time = obs_time
        self.pred_time = pred_time




    def collect_data(self): 
        root_dir = '/home/hbi/RAD_NL25_RAP_5min/' 
    

        df_train = pd.read_csv('/space/zboucher/World_Model/catchment/training_Delfland08-14_20.csv', header=None)
        event_times = df_train[0].to_list()
        dataset_train = self.radarDataset(root_dir, event_times, transform=Compose([ToTensor()]))

        df_train_s = pd.read_csv('/space/zboucher/World_Model/catchment/training_Delfland08-14.csv', header=None)
        event_times = df_train_s[0].to_list()
        dataset_train_del = self.radarDataset(root_dir, event_times, transform=Compose([ToTensor()]))

        df_test = pd.read_csv('/space/zboucher/World_Model/catchment/testing_Delfland18-20.csv', header=None)
        event_times = df_test[0].to_list()
        dataset_test = self.radarDataset(root_dir, event_times, transform=Compose([ToTensor()]))

        df_vali = pd.read_csv('/space/zboucher/World_Model/catchment/validation_Delfland15-17.csv', header=None)
        event_times = df_vali[0].to_list()
        dataset_vali = self.radarDataset(root_dir, event_times, transform=Compose([ToTensor()]))

        df_train_aa = pd.read_csv('/space/zboucher/World_Model/catchment/training_Aa08-14.csv', header=None)
        event_times = df_train_aa[0].to_list()
        dataset_train_aa = self.radarDataset(root_dir, event_times, transform=Compose([ToTensor()]))

        df_train_dw = pd.read_csv('/space/zboucher/World_Model/catchment/training_Dwar08-14.csv', header=None)
        event_times = df_train_dw[0].to_list()
        dataset_train_dw = self.radarDataset(root_dir, event_times, transform=Compose([ToTensor()]))

        df_train_re = pd.read_csv('/space/zboucher/World_Model/catchment/training_Regge08-14.csv', header=None)
        event_times = df_train_re[0].to_list()
        dataset_train_re = self.radarDataset(root_dir, event_times, transform=Compose([ToTensor()]))

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
    

    def __len__(self):
        return len(self.event_times)

    def radarDataset(self, idx):
        start_time = str(self.event_times[idx])
        time_list_pre, time_list_obs = self.eventGeneration(start_time, self.obs_time, self.pred_time)
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
        output = torch.permute(torch.tensor(np.array(output)), (1, 2, 0))
        output = self.transform(np.array(output))
        return output
    
    def eventGeneration(self, start_time):
# Generate event based on starting time point, return a list: [[t-4,...,t-1,t], [t+1,...,t+72]]
# Get the start year, month, day, hour, minute
        year = int(start_time[0:4])
        month = int(start_time[4:6])
        day = int(start_time[6:8])
        hour = int(start_time[8:10])
        minute = int(start_time[10:12])
        #print(datetime(year=year, month=month, day=day, hour=hour, minute=minute))
        times = [(datetime(year, month, day, hour, minute) + timedelta(minutes=self.time_interval * (x+1))) for x in range(self.pred_rime)]
        lead = [dt.strftime('%Y%m%d%H%M') for dt in times]
        times = [(datetime(year, month, day, hour, minute) - timedelta(minutes=self.time_interval * x)) for x in range(self.obs_time)]
        obs = [dt.strftime('%Y%m%d%H%M') for dt in times]
        obs.reverse()
        return lead, obs

    def collect_training_data(self):
        loaders, length_train, length_test, length_val = self.collect_data()
        loaders_train = loaders['train']
        length = length_train
        # training_samples=[]
        # loaders=self.collect_data()
        # num_samples=1
        # for i, images in enumerate(loaders['train']):
        #     if num_samples >= 50: 
        #          break 
            
        #     image = images.view(1,16,256,256)
        #     #image= image.squeeze(-1)
        #     self.training_data.append(image)
        #     num_samples=num_samples+1
        #     length= len(self.training_data)

        return loaders_train, length
            
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
            image = images.view(1,16,256,256)
            #image= image.unsqueeze(-1)
            training_batch.append(image)

        batch = training_batch
        
        current_index = end_index
        self.episode_ids = [None] * self.env.num_envs
        self.process_and_add_episodes(batch, current_index, epoch)
            
        return batch, current_index 

    def process_and_add_episodes(self, batch, index, epoch):
        # Assuming you have the logic to process episodes here
        # Extract observations, actions, rewards, dones, etc.
        if self.batch_counter_10 < 10:
            episode_tensor = torch.cat(batch, dim=0)
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


# # Correct

# class radarDataset(Dataset):
#     def __init__(self, root_dir, event_times, obs_number = 7, pred_number = 9, transform=None):
#         # event_times is an array of starting time t(string)
#         # transform is the preprocessing functions
#         self.root_dir = root_dir
#         self.transform = transform
#         self.event_times = event_times
#         self.obs_number = obs_number
#         self.pred_number = pred_number
#     def __len__(self):
#         return len(self.event_times)
#     def __getitem__(self, idx):
#         start_time = str(self.event_times[idx])
#         time_list_pre, time_list_obs = eventGeneration(start_time, self.obs_number, self.pred_number)
#         output = []
#         time_list = time_list_obs + time_list_pre
#         #print(time_list)
#         for time in time_list:
#             year = time[0:4]
#             month = time[4:6]
#             #path = self.root_dir + year + '/' + month + '/' + 'RAD_NL25_RAC_MFBS_EM_5min_' + time + '_NL.h5'
#             path = self.root_dir + year + '/' + month + '/' + 'RAD_NL25_RAP_5min_' + time + '.h5'
#             image = np.array(h5py.File(path)['image1']['image_data'])
#             #image = np.ma.masked_where(image == 65535, image)
#             image = image[264:520,242:498]
#             image[image == 65535] = 0
#             image = image.astype('float32')
#             image = image/100*12
#             image = np.clip(image, 0, 128)
#             image = image/40
#             #image = 2*image-1 #normalize to [-1,1]
#             output.append(image)
#         output = torch.permute(torch.tensor(np.array(output)), (1, 2, 0))
#         output = self.transform(np.array(output))
#         return output

# def eventGeneration(start_time, obs_time = 7 ,lead_time = 9, time_interval = 15):
# # Generate event based on starting time point, return a list: [[t-4,...,t-1,t], [t+1,...,t+72]]
# # Get the start year, month, day, hour, minute
#     year = int(start_time[0:4])
#     month = int(start_time[4:6])
#     day = int(start_time[6:8])
#     hour = int(start_time[8:10])
#     minute = int(start_time[10:12])
#     #print(datetime(year=year, month=month, day=day, hour=hour, minute=minute))
#     times = [(datetime(year, month, day, hour, minute) + timedelta(minutes=time_interval * (x+1))) for x in range(lead_time)]
#     lead = [dt.strftime('%Y%m%d%H%M') for dt in times]
#     times = [(datetime(year, month, day, hour, minute) - timedelta(minutes=time_interval * x)) for x in range(obs_time)]
#     obs = [dt.strftime('%Y%m%d%H%M') for dt in times]
#     obs.reverse()
#     return lead, obs
