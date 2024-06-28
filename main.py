from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F
from dataset import *
import time
from tqdm import tqdm
from model import *
from train import train
from test import test

# jupyter kernel 안잡힐 때
##### $ python -m ipykernel install --user --name conv-codes


import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "0, 1"

# 데이터 불러오기
data_dir = './ETRI_data'

train_acc = pd.read_parquet(os.path.join(data_dir,'train/acc_final.parquet'))
train_hr = pd.read_parquet(os.path.join(data_dir, 'train/hr_final.parquet'))
train_act = pd.read_parquet(os.path.join(data_dir, 'train/act_final.parquet'))
train_gps = pd.read_parquet(os.path.join(data_dir, 'train/gps_final.parquet'))

valid_acc = pd.read_parquet(os.path.join(data_dir,'valid/acc_final.parquet'))
valid_hr = pd.read_parquet(os.path.join(data_dir, 'valid/hr_final.parquet'))
valid_act = pd.read_parquet(os.path.join(data_dir, 'valid/act_final.parquet'))
valid_gps = pd.read_parquet(os.path.join(data_dir, 'valid/gps_final.parquet'))

valid_acc['user'] = valid_acc['user'].astype('str')
valid_hr['user'] = valid_hr['user'].astype('str')
valid_act['user'] = valid_act['user'].astype('str')
valid_gps['user'] = valid_gps['user'].astype('str')

test_acc = pd.read_parquet(os.path.join(data_dir,'test/acc_final.parquet'))
test_hr = pd.read_parquet(os.path.join(data_dir, 'test/hr_final.parquet'))
test_act = pd.read_parquet(os.path.join(data_dir, 'test/act_final.parquet'))
test_gps = pd.read_parquet(os.path.join(data_dir, 'test/gps_final.parquet'))

# final2는 패딩 길이 통일
train_label = pd.read_csv("./ETRI_data/train/new_train/train_label_new.csv")
valid_label = pd.read_csv("./ETRI_data/val_label.csv")

concat_acc = pd.concat([train_acc, valid_acc], axis=0)
concat_hr = pd.concat([train_hr, valid_hr], axis=0)
concat_act = pd.concat([train_act, valid_act], axis=0)
concat_gps = pd.concat([train_gps, valid_gps], axis=0)
#
concat_label = pd.concat([train_label, valid_label],axis=0)

# 터 더
# train_data = LifelogDataset(train_acc, train_gps, train_hr, train_act, train_label ,mode='train')
# train_data = DataLoader(train_data, batch_size=8, shuffle=True)
#
# val_data = LifelogDataset(valid_acc, valid_gps, valid_hr, valid_act, valid_label ,mode='valid')
# val_data = DataLoader(val_data, batch_size=8, shuffle=False, drop_last=True)

train_data = LifelogDataset(concat_acc, concat_gps, concat_hr, concat_act, concat_label ,mode='train')
train_data = DataLoader(train_data, batch_size=8, shuffle=True)

test_data = LifelogDataset(test_acc, test_gps, test_hr, test_act, mode='test')
test_data = DataLoader(test_data, batch_size=8)




if __name__ == '__main__':
    seed = 1234
    np.random.seed(seed)
    torch.manual_seed(seed)

    loss_fn = nn.BCELoss()
    epochs = 100
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = BasicModel(acc_in=6, hr_in=1, act_in=7, num_units=200).to(device)
    model.load_state_dict(torch.load('./log/final_model_cbam_residual.pt', map_location=device))
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    train_loss_lst = []
    train_acc_lst = []

    ## Training & Validation
    start_time = time.time()
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train(device, train_data, model, optimizer, loss_fn)
        train_loss_lst.append(train_loss)
        train_acc_lst.append(train_acc)
    torch.save(model.state_dict(), './log/final_model_cbam_residual2.pt')
    end_time = time.time()

    ## TEST
    final_prediction = test(device, test_data, model)

    final_test_df = pd.read_csv("./dataset/answer_sample.csv")
    columns = list(final_test_df.columns)[2:]
    final_prediction = final_prediction.cpu().detach()
    for idx, col in enumerate(columns):
        final_test_df[col] = final_prediction[:, idx]

    data_dir = "./result"
    final_test_df.to_csv(os.path.join(data_dir, 'final_result_cbam_residual2.csv'), index=False)


import matplotlib.pyplot as plt
Epoch = list(range(epochs))
plt.figure(figsize=(6, 4))
plt.plot(Epoch, train_loss_lst, '-', color='b', label='Train')

plt.xlim(1, epochs)
plt.xlabel('epoch')
plt.ylabel('BCE')
plt.title('Loss Plot')
plt.grid(True, which="both", ls="--")
plt.legend()
plt.show()


Epoch = list(range(epochs))
plt.figure(figsize=(6, 4))
plt.plot(Epoch, train_acc_lst, '--', label='Train', color='b')

plt.xlim(1, epochs)
plt.xlabel('epoch')
plt.ylabel('ACC')
plt.title('Accuracy Plot')
plt.grid(True, which="both", ls="--")
plt.legend()
plt.show()

