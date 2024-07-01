from cnn_module import *
import torch

# 개별 모델 선언
# accelerator model 선언
class Accelerator(nn.Module):
    def __init__(self, acc_in, num_units):
        super().__init__()
        dr = 0.5

        self.cnn_only = nn.Sequential(nn.Conv1d(acc_in, num_units, 4, 2, groups=2),
                                    nn.BatchNorm1d(num_units),
                                    nn.ReLU(),
                                    nn.Dropout(dr),

                                    nn.Conv1d(num_units, num_units//2, 4, 2, groups=2),
                                    nn.BatchNorm1d(num_units//2),
                                    nn.ReLU(),
                                    nn.Dropout(dr),

                                    nn.Conv1d(num_units // 2, num_units // 10, 4, 2, groups=2),
                                    nn.BatchNorm1d(num_units//10),
                                    nn.ReLU(),
                                    nn.Dropout(dr),

                                    nn.Conv1d(num_units // 10, num_units // 2, 4, 2, groups=2),
                                    nn.BatchNorm1d(num_units//2),
                                    nn.ReLU(),
                                    nn.Dropout(dr),

                                    nn.Conv1d(num_units // 2, num_units, 4, 2, groups=2))

        self.rnn = nn.GRU(num_units, num_units, num_layers=2, batch_first=True, dropout=0.2,
                          bidirectional=False)

        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.2)


    def forward(self, x):

        out = x.permute(0, 2, 1)
        out = self.cnn_only(out)

        out = out.permute(0, 2, 1)
        out, _ = self.rnn(out)
        out = out.permute(0, 2, 1)
        return out


# heart rate model 선언
class HeartRate(nn.Module):
    def __init__(self, hr_in, num_units):
        super().__init__()
        dr = 0.5

        self.cnn_only = nn.Sequential(nn.Conv1d(hr_in, num_units, 2, 1),
                                    nn.BatchNorm1d(num_units),
                                    nn.ReLU(),
                                    nn.Dropout(dr),

                                    nn.Conv1d(num_units, num_units//2, 2, 1),
                                    nn.BatchNorm1d(num_units//2),
                                    nn.ReLU(),
                                    nn.Dropout(dr),

                                    nn.Conv1d(num_units // 2, num_units // 10, 2, 1),
                                    nn.BatchNorm1d(num_units//10),
                                    nn.ReLU(),
                                    nn.Dropout(dr),

                                    nn.Conv1d(num_units // 10, num_units // 2, 2, 1),
                                    nn.BatchNorm1d(num_units//2),
                                    nn.ReLU(),
                                    nn.Dropout(dr),

                                    nn.Conv1d(num_units // 2, num_units, 2, 1))
        self.rnn = nn.GRU(num_units, num_units, num_layers=2, batch_first=True, dropout=0.2,
                          bidirectional=False)

        self.drop = nn.Dropout(0.2)

    def forward(self, x):
        out = self.cnn_only(x)

        out = out.permute(0, 2, 1)
        out, _ = self.rnn(out)
        out = out.permute(0, 2, 1)
        return out


# activity model 선언
class Act(nn.Module):
    def __init__(self, act_in, num_units):
        super().__init__()
        self.mlp1 = nn.Linear(act_in, num_units)
        self.mlp2 = nn.Linear(num_units, num_units)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.2)

    def forward(self, x):
        out = self.mlp1(x)
        out = self.relu(out)
        out = self.mlp2(out)
        out = self.relu(out)
        return out



# prediction model 선언
class PredModel(nn.Module):
    def __init__(self, num_units):
        super().__init__()
        self.mlp1 = nn.Linear(2690, num_units) #55237 #55227 #2685 #2690 #1416 #1015 #495
        self.mlp2 = nn.Linear(num_units, 7)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.drop = nn.Dropout(0.2)

    def forward(self, x):
        out = self.mlp1(x)
        out = self.relu(out)
        out = self.mlp2(out)
        out = self.sigmoid(out)
        return out



# model 합치기
class BasicModel(nn.Module):
    def __init__(self, acc_in, hr_in, act_in, num_units):
        super().__init__()

        self.acc_model = Accelerator(acc_in, num_units)
        self.hr_model = HeartRate(hr_in, num_units)
        self.act_model = Act(act_in, num_units)
        # gps model 선언
        self.gps_model = nn.Sequential(nn.BatchNorm1d(1),
                                       nn.Conv1d(1, num_units, 1, 1))
        self.se = SELayer(num_units)
        self.ca = ChannelAttention(num_units)
        self.sa = SpatialAttention()

        self.fusion =  nn.Sequential(nn.Conv1d(num_units, num_units // 2, 1, 1),
                                    nn.BatchNorm1d(num_units // 2),
                                    nn.ReLU(),
                                    nn.Conv1d(num_units // 2, num_units // 10, 1, 1),
                                    nn.BatchNorm1d(num_units // 10),
                                    nn.ReLU(),
                                    nn.Conv1d(num_units // 10, num_units, 1, 1))
        self.cnn = nn.Conv1d(200, 1, 1)

        self.pred_model = PredModel(num_units)

    def forward(self, acc, hr, act, gps):
        acc_out = self.acc_model(acc)
        hr_out = self.hr_model(hr)
        act_out = self.act_model(act)
        act_out = act_out.unsqueeze(dim=2)
        gps_out = self.gps_model(gps)

        x_concat = torch.concat([acc_out, hr_out, act_out, gps_out], dim=2)
        residual = x_concat

        x = self.ca(x_concat) * x_concat
        x = self.sa(x) * x
        x += residual

        x = self.fusion(x)
        x += x_concat
        x = self.cnn(x)

        x = x.squeeze()
        out = self.pred_model(x)
        return out
