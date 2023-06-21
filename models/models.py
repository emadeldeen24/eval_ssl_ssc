import torch
from torch import nn
from copy import deepcopy
from .helpers import SEBasicBlock
from .helpers import MultiHeadedAttention, TCE, PositionwiseFeedForward, EncoderLayer


def get_network_class(network_name):
    """Return the algorithm class with the given name."""
    if network_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(network_name))
    return globals()[network_name]


class classifier(nn.Module):
    def __init__(self, configs, hparams):
        super(classifier, self).__init__()
        print(hparams)
        self.logits = nn.Linear(hparams["clf"], configs.num_classes)

    def forward(self, x):
        # print(x.shape)
        x_flat = x.reshape(x.shape[0], -1)
        predictions = self.logits(x_flat)
        return predictions


##########################################################################################

class cnn1d_fe(nn.Module):
    def __init__(self, configs):
        super(cnn1d_fe, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, configs.mid_channels, kernel_size=configs.kernel_size,
                      stride=configs.stride, bias=False, padding=(configs.kernel_size // 2)),
            nn.BatchNorm1d(configs.mid_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(configs.mid_channels, configs.mid_channels * 2, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.mid_channels * 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(configs.mid_channels * 2, configs.final_out_channels, kernel_size=8, stride=1, bias=False,
                      padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )


    def forward(self, x_in):
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        # print(x.shape)
        return x


class cnn1d_temporal(nn.Module):
    def __init__(self, hparams):
        super(cnn1d_temporal, self).__init__()

    def forward(self, x):
        return x


##########################################################################################


class attnsleep_fe(nn.Module):
    def __init__(self, configs):
        super(attnsleep_fe, self).__init__()

        self.features1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, 64, kernel_size=50, stride=6, bias=False, padding=24),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=8, stride=2, padding=4),
            nn.Dropout(0.2),

            nn.Conv1d(64, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            nn.GELU(),

            nn.Conv1d(128, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            nn.GELU(),

            nn.MaxPool1d(kernel_size=4, stride=4, padding=2)
        )

        self.features2 = nn.Sequential(
            nn.Conv1d(configs.input_channels, 64, kernel_size=400, stride=50, bias=False, padding=200),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=4, stride=2, padding=2),
            nn.Dropout(0.2),

            nn.Conv1d(64, 128, kernel_size=7, stride=1, bias=False, padding=3),
            nn.BatchNorm1d(128),
            nn.GELU(),

            nn.Conv1d(128, 128, kernel_size=7, stride=1, bias=False, padding=3),
            nn.BatchNorm1d(128),
            nn.GELU(),

            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )
        self.dropout = nn.Dropout(0.2)
        self.inplanes = 128
        self.AFR = self._make_layer(SEBasicBlock, 30, 1)

    def _make_layer(self, block, planes, blocks, stride=1):  # makes residual SE block
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.features1(x)
        x2 = self.features2(x)
        x_concat = torch.cat((x1, x2), dim=2)
        x_concat = self.dropout(x_concat)
        x_concat = self.AFR(x_concat)
        return x_concat


class attnsleep_temporal(nn.Module):
    def __init__(self, hparams):
        super(attnsleep_temporal, self).__init__()

        if hparams["features_len"] % 99 == 0:
            d_model = 99
            h = 3
        elif hparams["features_len"] % 157 == 0:
            d_model = 157
            h = 1
        else:
            d_model = 80
            h = 5

        N = 2  # number of TCE clones
        d_ff = 120  # dimension of feed forward

        dropout = 0.1
        num_classes = 5
        afr_reduced_cnn_size = 30

        attn = MultiHeadedAttention(h, d_model, afr_reduced_cnn_size)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.tce = TCE(EncoderLayer(d_model, deepcopy(attn), deepcopy(ff), afr_reduced_cnn_size, dropout), N)

    def forward(self, x_feat):
        encoded_features = self.tce(x_feat)
        encoded_features = encoded_features.contiguous()
        return encoded_features

##########################################################################################

class dsn_fe(nn.Module):
    def __init__(self, configs):
        super(dsn_fe, self).__init__()
        self.features_s = nn.Sequential(
            nn.Conv1d(configs.input_channels, 64, 50, 6, padding=24),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=8, stride=8, padding=4),
            nn.Dropout(),
            nn.Conv1d(64, 128, 6, padding=3),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, 6, padding=3),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, 6, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )
        self.features_l = nn.Sequential(
            nn.Conv1d(configs.input_channels, 64, 400, 50, padding=200),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4, padding=2),
            nn.Dropout(),
            nn.Conv1d(64, 128, 8, padding=3),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, 8, padding=3),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, 8, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

    def forward(self, x):
        x_s = self.features_s(x)
        x_l = self.features_l(x)
        x = torch.cat((x_s, x_l), 2)
        return x


class dsn_temporal(nn.Module):  # current one!
    def __init__(self, hparams):
        super(dsn_temporal, self).__init__()
        self.features_seq = nn.LSTM(hparams["features_len"], 512, batch_first=True, bidirectional=True, dropout=0.5,
                                    num_layers=2)
        self.res = nn.Linear(hparams["features_len"], 1024)

    def forward(self, x):
        x = x.flatten(1, 2)
        x_seq = x.unsqueeze(1)
        x_blstm, _ = self.features_seq(x_seq)
        x_blstm = torch.squeeze(x_blstm, 1)
        x_res = self.res(x)
        x = torch.mul(x_res, x_blstm)

        return x
