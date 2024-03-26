import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset

import random
import sys
from typing import List, Optional, Union
from einops import rearrange
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from torch.cuda.amp import autocast
from torch.autograd import Variable
import pandas as pd
import h5py
from torchvision.transforms import ToTensor, Compose, CenterCrop


class Collector: 
    def __init__(self):
        self.training_data = []
        self.testing_data = []
        self.eval_data = []

    class CustomDataset(Dataset): # Radar dataset 
        def __init__(self, file_path):
            # Load the combined data of images and start times
            self.data = np.load(file_path, allow_pickle=True)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index):
            # Retrieve image and start time for the given index
            image, start_time = self.data[index]
            # Return as NumPy arrays (or in their original format)
            return image
#, start_time

    class CombinedDataset(Dataset): # combine radar and phy dataset
        def __init__(self, phy_dataset, radar_dataset):
            assert len(phy_dataset) == len(radar_dataset)
            self.phy_dataset = phy_dataset
            self.radar_dataset = radar_dataset

        def __len__(self):
            return len(self.phy_dataset)

        def __getitem__(self, idx):
            phy_data = self.phy_dataset[idx]
            radar_data , start_time = self.radar_dataset[idx]
            
            # if start_time_phy != start_time_radar:
            #     raise ValueError("start_time_phy and start_time_radar do not match")
            return (phy_data, radar_data) , start_time
#start_time_phy, start_time_phy1, time_list
# start_time_phy, start_time_phy1, start_time_radar, time_list


    def collect_data(self, batch_size): 

        root_dirs = ["/space/junzheyin/phy_data/evapotranspiration maps", "/space/junzheyin/phy_data/Specific Humidity maps","/space/junzheyin/phy_data/U100","/space/junzheyin/phy_data/V100","/space/junzheyin/phy_data/Wind_u","/space/junzheyin/phy_data/Wind_v","/space/junzheyin/phy_data/Dew Temperature maps"]  # List of root directories
        file_names = ["evapot_kriging_", "humidity_kriging_","Wind_U100_kriging_","Wind_V100_kriging_","Wind_U_kriging_","Wind_V_kriging_","temp_kriging_"]  # List of base file names
        dataset_names = ["eva", "humidity","u100","v100","Wind_U","Wind_V","DW_temp"]  # List of dataset names within each HDF5 file

        df_train_s = pd.read_csv('/space/junzheyin/haorandata1/training_Delfland08-14.csv', header=None)
        event_times = df_train_s[0].to_list()
        dataset_train_del = phyDataset(event_times, root_dirs, file_names, dataset_names, transform=Compose([ToTensor()]))
       # dataset_train_del1 = radarDataset(event_times, transform = None)

        df_train_aa = pd.read_csv('/space/junzheyin/haorandata1/training_Aa08-14.csv', header=None)
        event_times = df_train_aa[0].to_list()
        dataset_train_aa = phyDataset(event_times, root_dirs, file_names, dataset_names, transform=Compose([ToTensor()]))
       # dataset_train_aa1 = radarDataset(event_times, transform = None)

        df_train_dw = pd.read_csv('/space/junzheyin/haorandata1/training_Dwar08-14.csv', header=None)
        event_times = df_train_dw[0].to_list()
        dataset_train_dw = phyDataset(event_times, root_dirs, file_names, dataset_names, transform=Compose([ToTensor()]))
        #dataset_train_dw1 = radarDataset(event_times, transform = None)

        df_train_re = pd.read_csv('/space/junzheyin/haorandata1/training_Regge08-14.csv', header=None)
        event_times = df_train_re[0].to_list()
        dataset_train_re = phyDataset(event_times, root_dirs, file_names, dataset_names, transform=Compose([ToTensor()]))
      #  dataset_train_re1 = radarDataset(event_times, transform = None)

        df_test = pd.read_csv('/space/junzheyin/haorandata1/testing_Delfland18-20.csv', header=None)
        event_times = df_test[0].to_list()
        dataset_test = phyDataset(event_times, root_dirs, file_names, dataset_names, transform=Compose([ToTensor()]))
       # dataset_test1 = radarDataset(event_times, transform = None)

        df_vali = pd.read_csv('/space/junzheyin/haorandata1/validation_Delfland15-17.csv', header=None)
        event_times = df_vali[0].to_list()
        dataset_vali = phyDataset(event_times, root_dirs, file_names, dataset_names, transform=Compose([ToTensor()]))
      #  dataset_vali1 = radarDataset(event_times, transform = None)

        # extreme, top 1%
        df_train_ext = pd.read_csv('/space/junzheyin/haorandata1/training_Delfland08-14_ext.csv', header = None)
        event_times = df_train_ext[0].to_list()
        mfbs = df_train_ext[1].to_list()
        dic_mfbs1 = dict(zip(event_times, mfbs))
        dataset_train_ext = phyDataset(event_times, root_dirs, file_names, dataset_names, transform=Compose([ToTensor()]))
       # dataset_train_ext1 = radarDataset(event_times, transform = None)   

        df_test_ext = pd.read_csv('/space/junzheyin/haorandata1/testing_Delfland18-20_ext.csv', header = None)
        event_times = df_test_ext[0].to_list()
        mfbs = df_test_ext[1].to_list()
        dic_mfbs2 = dict(zip(event_times, mfbs))
        dataset_test_ext = phyDataset(event_times, root_dirs, file_names, dataset_names, transform=Compose([ToTensor()]))
        dataset_test_ext1= radarDataset(event_times, transform=Compose([ToTensor()]))

        df_vali_ext = pd.read_csv('/space/junzheyin/haorandata1/validation_Delfland15-17_ext.csv', header = None)
        event_times = df_vali_ext[0].to_list()
        mfbs = df_vali_ext[1].to_list()
        dic_mfbs3 = dict(zip(event_times, mfbs))
        dataset_vali_ext = phyDataset(event_times, root_dirs, file_names, dataset_names, transform=Compose([ToTensor()]))
       # dataset_vali_ext1 = radarDataset(event_times, transform = None)

        dic_mfbs = {}
        dic_mfbs.update(dic_mfbs1)
        dic_mfbs.update(dic_mfbs2)
        dic_mfbs.update(dic_mfbs3)

        #
        #201907261340
        #201907111245
        new_valid = ['201903070300', '201903070315', '201904241715', '201904241730', '201904241745', '201904241800', '201904241815', '201904241830', '201904241845', '201904241900', '201904241915', '201905280330', '201905280345', '201905280400', '201905280415', '201905280430', '201905280445', '201905280500', '201905280515', '201906052035', '201906052050', '201906052100', '201906120545', '201906120600', '201906120615', '201906120630', '201906120645', '201906120700', '201906120715', '201906120730', '201906120745', '201906120800', '201906120815', '201906120830', '201906120845', '201906120900', '201906120915', '201906120930', '201906120945', '201906121000', '201906121015', '201906121030', '201906121045', '201906121100', '201906150035', '201906150050', '201906150100', '201906150115', '201906150130', '201906150145', '201906150200', '201906150215', '201906150230', '201906150245', '201906150300', '201906150315', '201906150330', '201906190400', '201906190415', '201906190430', '201906190445', '201906190500', '201906190515', '201906190530', '201906190545', '201906190600', '201908121840', '201910210250', '201910210300', '201910210315', '201910210330', '201910210345', '201910210400', '201910210415', '201910210430', '201910210445', '201911280500', '201911280515', '201911280530', '201911280545', '201911280600', '201911280615', '202002091800', '202002091815', '202002091830', '202002091845', '202002091900', '202002161520', '202002161535', '202002161550', '202003051615', '202003051630', '202003051645', '202003051700', '202006050400', '202006050415', '202006050430', '202006050445', '202006050500', '202006050515', '202006050530', '202006050545', '202006050600', '202006050615', '202006050630', '202006121915', '202006121930', '202006121945', '202006122000', '202006122015', '202006122030', '202006122045', '202006161035', '202006161050', '202006161100', '202006161115', '202006161130', '202006161145', '202006161200', '202006161215', '202006161230', '202006161245', '202006171730', '202006171745', '202006171800', '202006171815', '202006171830', '202006171845', '202006171900', '202006171915', '202006171930', '202006171945', '202006172000', '202006172015', '202007252020', '202007252035', '202007252050', '202007252100', '202007252115', '202008161345', '202008161400', '202008161415', '202008161430', '202008161445', '202008161500', '202008161515', '202008161530', '202008161545', '202008161600', '202008161615', '202009232000', '202009232015', '202102031100', '202102031115', '202102031130', '202105131415', '202105131430', '202105131445', '202105131500', '202105131515', '202105131530', '202105131545', '202105131600', '202105131615', '202105131630', '202105131645', '202105131700', '202105131715', '202106180100', '202106180115', '202106180130', '202106180145', '202106180200', '202106192015', '202106192030', '202106192045', '202106192100', '202106192115', '202106192130', '202106192145', '202106192200', '202106192215', '202106192230', '202106192245', '201904241800', '201904241815', '201904241830', '201907111120', '201907111135', '201907111150', '201907111200', '201907111215', '201907111230', '201907111300', '201907111315', '201907111330', '201908091035', '201908091050', '201908091100', '201908091115', '201908091130', '201908091145', '201908091200', '201908121410', '201908121425', '201908121440', '201908121500', '201908121515', '201908121530', '201908121545', '201908121600', '201908121615', '201908121630', '201908121645', '201908282300', '201908282315', '201908282330', '201908282345', '201908290000', '201908290015', '201908290030', '201908290045', '201908290100', '201908290115', '201908290130', '201908290145', '201909261805', '201909261820', '201909261835', '201909261850', '201909261900', '201909261915', '201909261930', '201909261945', '201909262000', '201910011820', '201910011835', '201910011850', '201910011900', '201910011915', '201910011930', '201910011945', '201910012000', '201910012015', '201910012030', '202006141135', '202006141150', '202006141200', '202006141215', '202006141230', '202006141245', '202006141300', '202006141315', '202006141330', '202006141345', '202006141400', '202006141415', '202006141430', '202006141445', '202006141500', '202006141515', '202006141530', '202006141545', '202006141600', '202006141615', '202006141630', '202006141645', '202006141700', '202006141715', '202006141730', '201904241750', '201904241800', '201905191600', '201905191615', '201905191630', '201905191645', '201905191700', '201905191715', '201905191730', '201905191745', '201906052135', '201906052150', '201906052200', '201906052215', '201906150100', '201906150115', '201906150130', '201906150145',  '201907261415', '201907261430', '201907261445', '201907261500', '201907261515', '201908282240', '201908282300', '201908282315', '201908282330', '201908282345', '201908290000', '201908290015', '201908290030', '201908290045', '201908290100', '201908290115', '201910010435', '201910010450', '201910010500', '201910010515', '201910010530', '201910010545', '201910010600', '201910200930', '201910200945', '202006122010', '202006122025', '202006122040', '202006122100', '202006122115', '202006122130', '202006122145', '202006122200', '202006122215', '202006122230', '202006122245', '202006171525', '202006171540', '202006171600', '202006171615', '202006171630', '202006171645', '202006171700', '202006171715', '202006171730', '202006171745', '202006171800', '202006171815', '202006171830', '202006171845', '202006261700', '202006261715', '202006261730', '202006261745', '202006261800', '202006261815', '202006261830', '202006261845', '202006261900', '202006261915', '202006261930', '202006261945', '202006262000', '202009261920', '202009261935', '202009261950', '202009262000']

        new_valid.sort()      
        dataset_ext = phyDataset(new_valid, root_dirs, file_names, dataset_names, transform=Compose([ToTensor()]))
        dataset_ext1 = radarDataset(new_valid, transform=Compose([ToTensor()]))


        data_list = [dataset_train_aa, dataset_train_dw, dataset_train_del, dataset_train_re]
        train_aadedwre = torch.utils.data.ConcatDataset(data_list)


        train_file_path = '/space/junzheyin/all_data_train.npy'
        train_dataset = self.CustomDataset(train_file_path)

        test_file_path = '/space/junzheyin/all_data_test.npy'
        test_dataset = self.CustomDataset(test_file_path)


        # Combine datasets for training and testing
        combined_train_dataset = self.CombinedDataset(train_aadedwre, train_dataset)
        combined_test_dataset = self.CombinedDataset(dataset_test, test_dataset)
        combined_test_ext_dataset = self.CombinedDataset(dataset_test_ext, dataset_test_ext1)
        combined_ext_dataset = self.CombinedDataset(dataset_ext, dataset_ext1)


        loaders = { 'train' :DataLoader(combined_train_dataset, batch_size=batch_size, shuffle=True, num_workers=1),
                    'test' :DataLoader(combined_test_dataset, batch_size=batch_size, shuffle=False, num_workers=1), 
                    'test_ext' :DataLoader( combined_test_ext_dataset, batch_size=batch_size, shuffle=False, num_workers=1),
                    'ext' :DataLoader(combined_ext_dataset, batch_size=batch_size, shuffle=False, num_workers=1), 
                }
        return loaders
    
    def collect_training_data(self, batch_size):
        loaders = self.collect_data(batch_size)
        return loaders['train']
    
    
    def collect_testing_data(self, batch_size):
        loaders = self.collect_data(batch_size)
        return loaders['test']
    

    def collect_ext_testing_data(self, batch_size):
        loaders = self.collect_data(batch_size)
        return loaders['test_ext']
    

    def collect_ext_data(self, batch_size):
        loaders = self.collect_data(batch_size)
        return loaders['ext'], len(loaders['ext'])
    



class radarDataset(Dataset):
    def __init__(self, event_times, obs_number = 3, pred_number = 6, transform=None):
        # event_times is an array of starting time t(string)
        # transform is the preprocessing functions
        self.root_dir =  '/space/junzheyin/RAD_NL25_RAP_5min/' 
        self.transform = transform
        self.event_times = event_times
        self.obs_number = obs_number
        self.pred_number = pred_number
    def __len__(self):
        return len(self.event_times)
    def __getitem__(self, idx):
        start_time = str(self.event_times[idx])
        time_list_pre, time_list_obs = self.eventGeneration(start_time, self.obs_number, self.pred_number)
        output = []
        time_list = time_list_obs + time_list_pre
        #print(time_list)
        for time in time_list:
            year = time[0:4]
            month = time[4:6]
            #path = self.root_dir + year + '/' + month + '/' + 'RAD_NL25_RAC_MFBS_EM_5min_' + time + '_NL.h5'
            path = self.root_dir + year + '/' + month + '/' + 'RAD_NL25_RAP_5min_' + time + '.h5'
            image = np.array(h5py.File(path)['image1']['image_data'])
            image = image[264:520,242:498]
            image[image == 65535] = 0
            image = image.astype('float32')
            image = image/100*12
            image = np.clip(image, 0, 128)
            image = image/40
            output.append(image)
        output = torch.tensor((np.array(output)), dtype = torch.float32)
        # Reshape tensor to include a batch dimension (B, C, H, W)
        output = output.unsqueeze(0)
        # Resize tensor using F.interpolate
        output = F.interpolate(output, size=(128, 128), mode='bilinear', align_corners=False)
        # Remove batch dimension
        output = torch.squeeze(output, 0) # output : [C, H, W]
        return output, start_time
    #, start_time
    

    def eventGeneration(self, start_time, obs_time = 3 ,lead_time = 6, time_interval = 30):
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




class phyDataset(Dataset):
    def __init__(self, event_times, root_dirs, file_names, dataset_names, obs_number=3, pred_number=6, transform=None):
        self.root_dirs = root_dirs  # List of root directories for each dataset
        self.file_names = file_names  # List of base file names for each dataset
        self.dataset_names = dataset_names  # List of dataset names within each HDF5 file
        self.transform = transform
        self.event_times = event_times
        self.obs_number = obs_number
        self.pred_number = pred_number
    
    def round_to_nearest_half_hour(self, time_str):
        # Convert string to datetime object
        dt = datetime.strptime(time_str, '%Y%m%d%H%M')

        # Round minutes to the nearest 0 or 30
        if dt.hour == 0 and dt.minute <= 15:
            # Round up to 00:30
            dt = dt.replace(minute=30)
        elif dt.hour == 0 and dt.minute == 0:
            # If time is exactly 00:00, round to previous day 23:30
            dt = dt - timedelta(hours=1)
            dt = dt.replace(minute=30)
        elif dt.hour == 23 and dt.minute >= 45:
            # If time is 23:45 or later, round to 23:30
            dt = dt.replace(minute=30)
        else:
            # Regular rounding for other times
            minute = dt.minute
            if minute <= 15:
                dt = dt.replace(minute=0)
            elif 15 < minute <= 45:
                dt = dt.replace(minute=30)
            else:
                dt = dt.replace(minute=0) + timedelta(hours=1)
            

        # Convert back to string
        return dt.strftime('%Y%m%d%H%M')

    def __len__(self):
        return len(self.event_times)
    
    def __getitem__(self, idx):
        start = str(self.event_times[idx])
        start_time = self.round_to_nearest_half_hour(start)
        time_list_pre, time_list_obs = self.eventGeneration(start_time, self.obs_number, self.pred_number)
          


        channel_data = []
        datasets_to_multiply = ["u100", "v100", "Wind_U", "Wind_V"]
        #datasets_to_multiply = []
        for root_dir, file_name, dataset_name in zip(self.root_dirs, self.file_names, self.dataset_names):
            output = []
            time_list = time_list_obs + time_list_pre
            for time in time_list:
                year = time[0:4]
                month = time[4:6]
                path = f"{root_dir}/{year}/{month}/{file_name}{time}.h5"
                with h5py.File(path, 'r') as hdf:
                    image = np.array(hdf[dataset_name][:])
                    if dataset_name in datasets_to_multiply:
                       image = image * 3.6 # m/s to km/h 
                output.append(image)

            output_tensor = torch.tensor(np.array(output)) if not isinstance(output, torch.Tensor) else output
            if self.transform:
                # Apply transform if the data is not already a tensor
                output_tensor = self.transform(output_tensor) if not isinstance(output_tensor, torch.Tensor) else output_tensor
            channel_data.append(output_tensor)

        final_output = torch.stack(channel_data, dim=1)  # Adding the channel dimension
        return final_output
    # start, start_time, time_list
    def adjust_midnight_time(self, dt):
        # Adjusts a datetime object if it is exactly 00:00 to 23:30 of the previous day
        if dt.hour == 0 and dt.minute == 0:
            return dt - timedelta(hours=0, minutes=30)
        return dt
    

    def eventGeneration(self, start_time, obs_time = 3 ,lead_time = 6, time_interval = 30):
        # Generate event based on starting time point, return a list: [[t-4,...,t-1,t], [t+1,...,t+72]]
        # Get the start year, month, day, hour, minute
        year = int(start_time[0:4])
        month = int(start_time[4:6])
        day = int(start_time[6:8])
        hour = int(start_time[8:10])
        minute = int(start_time[10:12])
        #print(datetime(year=year, month=month, day=day, hour=hour, minute=minute))
        lead_times = [(datetime(year, month, day, hour, minute) + timedelta(minutes=time_interval * (x+1))) for x in range(lead_time)]
        #lead = [dt.strftime('%Y%m%d%H%M') for dt in times]
        obs_times = [(datetime(year, month, day, hour, minute) - timedelta(minutes=time_interval * x)) for x in range(obs_time)]
        
        # Adjust any times that are exactly 00:00 to 23:30 of the previous day
        adjusted_lead = [self.adjust_midnight_time(dt) for dt in lead_times]
        adjusted_obs = [self.adjust_midnight_time(dt) for dt in obs_times]

        lead = [dt.strftime('%Y%m%d%H%M') for dt in adjusted_lead]
        obs = [dt.strftime('%Y%m%d%H%M') for dt in adjusted_obs]
        
        
        #obs = [dt.strftime('%Y%m%d%H%M') for dt in times]
        obs.reverse()
        return lead, obs
    


