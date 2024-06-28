import pandas as pd
import torch
import os
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

def day_padding(x, padding_size):
    if len(x) < padding_size:
        pad_len = padding_size - len(x)
        x = np.pad(x, (0, pad_len), 'constant', constant_values=(0, 0))
    else:
        x = x[:padding_size]

    return x

# min max scaling 함수
def min_max_scaling(row, columns):
    for col in columns:
        x_list = row[col]
        min_x = min(x_list)
        max_x = max(x_list)
        range_x = max_x-min_x

        row[col] = [(x - min_x)/range_x for x in x_list]
    return row

def replace_zero(hr_list):
    idx=0
    x_list = hr_list.copy()
    while idx < len(x_list):
        if x_list[idx] == 0:
            zero_start = idx
            zero_end = idx
            while zero_end+1 < len(x_list) and x_list[zero_end+1] == 0 :
                zero_end += 1

            replace_length = zero_end-zero_start+1
            # 앞에서부터 0인 경우
            if zero_start == 0:
                rep_list = [x_list[zero_end+1]] * replace_length
            # 맨끝이 0인 경우
            elif zero_end == len(x_list)-1:
                rep_list = [x_list[zero_start-1]] * replace_length
            # 중간에 0이 있는 경우
            else:
                avg_hr = (x_list[zero_start-1] + x_list[zero_end+1])/2
                rep_list = [avg_hr] * replace_length

            x_list[zero_start:zero_end+1] = rep_list
            idx = zero_end+1
        else:
            idx = idx+1
    return x_list

class LifelogDataset(Dataset):
    def __init__(self,
                 acc,
                 gps: pd.DataFrame,
                 hr: pd.DataFrame,
                 act: pd.DataFrame,
                 label: pd.DataFrame = None,
                 mode: str = 'train'
                 ):

        # user, date 추출
        self.user = gps['user'].to_list()
        self.date = gps['date'].to_list()

        padding_size = 54240

        acc['x_gravity'] = acc['x_gravity'].apply(lambda x: day_padding(x, padding_size=padding_size))
        acc['y_gravity'] = acc['y_gravity'].apply(lambda x: day_padding(x, padding_size=padding_size))
        acc['z_gravity'] = acc['z_gravity'].apply(lambda x: day_padding(x, padding_size=padding_size))
        acc['x_body'] = acc['x_body'].apply(lambda x: day_padding(x, padding_size=padding_size))
        acc['y_body'] = acc['y_body'].apply(lambda x: day_padding(x, padding_size=padding_size))
        acc['z_body'] = acc['z_body'].apply(lambda x: day_padding(x, padding_size=padding_size))


        g_x = acc['x_gravity'].values
        g_y = acc['y_gravity'].values
        g_z = acc['z_gravity'].values
        b_x = acc['x_body'].values
        b_y = acc['y_body'].values
        b_z = acc['z_body'].values


        gravity = []
        for i in range(len(acc['x_gravity'].values)):
            lst = []
            lst.append(g_x[i])
            lst.append(g_y[i])
            lst.append(g_z[i])
            arr = np.array(lst)
            arr = arr.reshape((-1, 3))
            gravity.append(arr)

        body = []
        for i in range(len(acc['x_body'].values)):
            lst = []
            lst.append(b_x[i])
            lst.append(b_y[i])
            lst.append(b_z[i])
            arr = np.array(lst)
            arr = arr.reshape((-1, 3))
            body.append(arr)

        body_gravity = []
        for i in range(len(body)):
            # m = med[i]
            b = body[i]
            g = gravity[i]
            body_gravity.append(np.concatenate((b, g), axis=1))

        train_bg = np.array(body_gravity)
        self.acc = torch.FloatTensor(train_bg)

        # hear rate
        # 0인 값은 앞뒤 평균으로 대체
        hr['hr'] = hr['hr'].apply(replace_zero)
        hr_padding = 1000
        hr['hr'] = hr['hr'].apply(lambda x: day_padding(x, padding_size=hr_padding))
        hr = np.array(hr['hr'].to_list())
        self.hr = torch.FloatTensor(hr)

        # gps
        gps = gps.drop(columns={'user', 'date', 'lat', 'lon'})
        gps = gps.to_numpy()
        self.gps = torch.FloatTensor(gps)

        # activity
        act = act.drop(columns={'user', 'date'})
        act = act.to_numpy()
        self.act = torch.FloatTensor(act)


        # train이나 valid인 경우, label 값 가져오기
        self.mode = mode
        if self.mode in ['train', 'valid']:
            label.rename(columns={'subject_id': 'user'}, inplace=True)
            self.label_df = label

            label = label.drop(columns={'user', 'date'}).to_numpy()
            self.label = torch.FloatTensor(label)

    def __len__(self):
        return len(self.user)

    def __getitem__(self, index):
        if self.mode in ['train', 'valid']:
            return self.user[index], self.date[index], self.acc[index], self.hr[index], self.gps[index], self.act[index], self.label[
                index]
        elif self.mode == 'test':
            return self.user[index], self.date[index], self.acc[index], self.hr[index], self.gps[index], self.act[index]


