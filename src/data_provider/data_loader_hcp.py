import os
import datetime
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import warnings

from .timefeatures import time_features
import h5py
import json
warnings.filterwarnings('ignore')


# class Dataset_ETT_hour(Dataset):
#     def __init__(self, root_path, flag='train', size=None, data_path='ETTh1.csv',
#                  scale=False, features='M', target='OT', drop_short=True, freq = 's'):
#         self.seq_len = size[0]
#         self.label_len = size[1]
#         self.pred_len = size[2]
#         self.flag = flag
#         assert flag in ['train', 'test', 'val']
#         type_map = {'train': 0, 'val': 1, 'test': 2}
#         self.set_type = type_map[flag]
#         self.features = features
#         self.target = target
#         self.scale = scale
#         self.root_path = root_path
#         self.drop_short = drop_short
#         # self.data_path = self.root_path + self.flag + '/ts/' + data_path
#         if self.flag=='val':
#             self.flag='test'
#         # self.data_path = self.root_path + '/'+ self.flag +'/'+ data_path #formatted_data_corr_HCP_{self.flag}.json"
#         self.data_path=self.root_path + '/'+ self.flag+'/'+ data_path
#         self.freq = freq
#         self.__read_data__()

#     def __read_data__(self):
#         # temp = dd.io.load(self.data_path)
#         with open(self.data_path, "r") as f:
#             json_file = json.load(f)
#         # df_stamp = temp['text']
#         self.json_file=json_file

#         self.data_0=torch.tensor(json_file[0]['timeseires'])

#         # df_stamp=temp['label']
#         # # print('label',label)
#         # # print('df_stamp',df_stamp)
#         # # with h5py.File(os.path.join(path, f'{data_name}.h5'), "r") as f:
#         # #     data_temp=np.array(f['timeseires']) #(82, 1200)
#         # data_stamp = time_features(pd.to_datetime(df_stamp), freq=self.freq)
#         # data_stamp = data_stamp.transpose(1, 0)
#         # # temp = temp['fMRI']
#         # temp = torch.tensor(temp['timeseires']).transpose(1, 0)
#         # # temp = torch.tensor(temp['timeseires'])
#         # self.max = torch.unsqueeze(torch.mean(temp, dim=1), 1)
#         # self.min = torch.unsqueeze(torch.std(temp, dim=1), 1)
#         # temp = torch.div((temp - self.max), (self.min))
#         # self.data_x = temp.T
#         # # self.data_x = temp
#         # # print('data_x',self.data_x.shape)
#         # self.data_y = temp.T
#         # # self.data_y = temp
#         # self.data_stamp = data_stamp


#     def __getitem__(self, index):
#         print(index)
#         # print()
#         # print(len(self.json_file)) #696
#         temp=self.json_file[index]

#         df_stamp=temp['label']
#         data_stamp = time_features(pd.to_datetime(df_stamp), freq=self.freq)
#         self.data_stamp = data_stamp

#         data = torch.tensor(temp['timeseires'])

#         self.data_x=data.T
#         self.data_y=data.T

#         s_begin = index
#         s_end = s_begin + self.seq_len
#         r_begin = s_end - self.label_len
#         r_end = r_begin + self.label_len + self.pred_len

#         seq_x = self.data_x[s_begin:s_end]
#         seq_y = self.data_y[r_begin:r_end]
#         seq_x_mark = self.data_stamp[s_begin:s_end]
#         seq_y_mark = self.data_stamp[r_begin:r_end]

#         return seq_x, seq_y, seq_x_mark, seq_y_mark

#     def __len__(self):
#         return len(self.data_0.T) - self.seq_len - self.pred_len + 1
#         return len(self.data_0.T) - self.seq_len - self.pred_len + 1

#     def inverse_transform(self, data):
#         return self.scaler.inverse_transform(data)
    



class Dataset_fmri(Dataset):
    def __init__(self, flag='train', size=None, data= None,
                 scale=False, features='M', target='OT', drop_short=True, freq = 's'):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.flag = flag
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.features = features
        self.target = target
        self.scale = scale
        self.data=data

        self.drop_short = drop_short
        if self.flag=='val':
            self.flag='test'
        # self.data_path = self.root_path + '/'+ self.flag +'/'+ data_path #formatted_data_corr_HCP_{self.flag}.json"

        self.freq = freq
        self.__read_data__()

    def __read_data__(self):
        # temp = dd.io.load(self.data_path)
        # df_stamp = np.array(temp['timeseires'])

        df_stamp = self.data[0]
        # df_stamp = list(temp['timeseires'])
        # print(type(df_stamp))

        # print(df_stamp)
        # data_stamp = [str(x) for x in data_stamp] # self.data_stamp是list的形式
        data_stamp = time_features(pd.to_datetime(df_stamp), freq=self.freq)
        # print(data_stamp)
        data_stamp = data_stamp.transpose(1, 0)
        # print(data_stamp.shape) #(1,6)
        temp = torch.tensor(self.data)
        # self.max = torch.unsqueeze(torch.mean(temp, dim=1), 1)
        # self.min = torch.unsqueeze(torch.std(temp, dim=1), 1)
        # temp = torch.div((temp - self.max), (self.min))
        self.data_x = temp.T
        self.data_y = temp.T
        self.data_stamp = data_stamp


    def __getitem__(self, index):
        # print(len(self.data_x) - self.seq_len - self.pred_len + 1) #1121
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        # print(seq_x.shape)
        # print(seq_y.shape)
        # print(seq_x_mark.shape)
        # print(seq_y_mark.shape)

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


