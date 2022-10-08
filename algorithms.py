import torch
import torch.nn as nn
import numpy as np

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from models.loss import NTXentLoss
import torch.nn.functional as F
from models.TC import TC
from models.helpers import proj_head



def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain adaptation algorithm.
    Subclasses should implement the update() method.
    """

    def __init__(self, configs):
        super(Algorithm, self).__init__()
        self.configs = configs
        self.cross_entropy = nn.CrossEntropyLoss()

    def update(self, *args, **kwargs):
        raise NotImplementedError


class simclr(Algorithm):
    def __init__(self, backbone_fe, backbone_temporal, classifier, configs, hparams, device):
        super(simclr, self).__init__(configs)

        self.feature_extractor = backbone_fe(configs)
        self.temporal_encoder = backbone_temporal(hparams)
        self.proj_head = proj_head(configs, hparams)
        self.network = nn.Sequential(self.feature_extractor, self.proj_head)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"],
            betas=(0.9, 0.99)
        )
        self.hparams = hparams
        self.contrastive_loss = NTXentLoss(device, hparams["batch_size"], 0.2, True)

    def update(self, samples):
        # ====== Data =====================
        aug1 = samples["transformed_samples"][0]
        aug2 = samples["transformed_samples"][1]

        self.optimizer.zero_grad()

        features1 = self.feature_extractor(aug1)
        z1 = self.proj_head(features1)

        features2 = self.feature_extractor(aug2)
        z2 = self.proj_head(features2)

        # normalize projection feature vectors
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # Cross-Entropy loss
        loss = self.contrastive_loss(z1, z2)

        loss.backward()
        self.optimizer.step()

        return {'Total_loss': loss.item()}, \
               [self.feature_extractor, self.temporal_encoder, self.proj_head]


class cpc(Algorithm):
    def __init__(self, backbone_fe, backbone_temporal, classifier, configs, hparams, device):
        super(cpc, self).__init__(configs)

        self.feature_extractor = backbone_fe(configs)
        self.temporal_encoder = backbone_temporal(hparams)
        self.classifier = classifier(configs, hparams)
        self.network = nn.Sequential(self.feature_extractor, self.temporal_encoder, self.classifier)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"],
            betas=(0.9, 0.99)
        )

        self.hparams = hparams

        self.num_channels = hparams["num_channels"]
        self.hid_dim = hparams["hid_dim"]
        self.timestep = hparams["timesteps"]
        self.Wk = nn.ModuleList([nn.Linear(self.hid_dim, self.num_channels) for _ in range(self.timestep)])
        self.lsoftmax = nn.LogSoftmax()
        self.device = device

        self.lstm = nn.LSTM(self.num_channels, self.hid_dim, bidirectional=False, batch_first=True)

    def update(self, samples):
        # ====== Data =====================
        data = samples['sample_ori'].float()

        self.optimizer.zero_grad()

        # Src original features
        features = self.feature_extractor(data)
        seq_len = features.shape[2]
        features = features.transpose(1, 2)

        batch = self.hparams["batch_size"]
        t_samples = torch.randint(seq_len - self.timestep, size=(1,)).long().to(self.device) # randomly pick timesteps

        loss = 0  # average over timestep and batch
        encode_samples = torch.empty((self.timestep, batch, self.num_channels)).float().to(self.device)

        for i in np.arange(1, self.timestep + 1):
            encode_samples[i - 1] = features[:, t_samples + i, :].view(batch, self.num_channels)
        forward_seq = features[:, :t_samples + 1, :]

        output1, _ = self.lstm(forward_seq)
        c_t = output1[:, t_samples, :].view(batch, self.hid_dim)

        pred = torch.empty((self.timestep, batch, self.num_channels)).float().to(self.device)
        for i in np.arange(0, self.timestep):
            linear = self.Wk[i]
            pred[i] = linear(c_t)
        for i in np.arange(0, self.timestep):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))
            loss += torch.sum(torch.diag(self.lsoftmax(total)))
        loss /= -1. * batch * self.timestep

        loss.backward()
        self.optimizer.step()

        return {'Total_loss': loss.item()}, \
               [self.feature_extractor, self.temporal_encoder, self.classifier]


class ts_tcc(Algorithm):
    def __init__(self, backbone_fe, backbone_temporal, classifier, configs, hparams, device):
        super(ts_tcc, self).__init__(configs)

        self.feature_extractor = backbone_fe(configs)
        self.temporal_encoder = backbone_temporal(hparams)
        self.classifier = classifier(configs, hparams)
        self.temporal_contr_model = TC(hparams, device)

        self.network = nn.Sequential(self.feature_extractor, self.temporal_encoder,
                                     self.classifier, self.temporal_contr_model)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"],
            betas=(0.9, 0.99)
        )
        self.hparams = hparams
        self.contrastive_loss = NTXentLoss(device, hparams["batch_size"], 0.2, True)

    def update(self, samples):
        # ====== Data =====================
        aug1 = samples["transformed_samples"][0]
        aug2 = samples["transformed_samples"][1]

        self.optimizer.zero_grad()

        features1 = self.feature_extractor(aug1)
        features2 = self.feature_extractor(aug2)

        # normalize projection feature vectors
        features1 = F.normalize(features1, dim=1)
        features2 = F.normalize(features2, dim=1)

        temp_cont_loss1, temp_cont_lstm_feat1 = self.temporal_contr_model(features1, features2)
        temp_cont_loss2, temp_cont_lstm_feat2 = self.temporal_contr_model(features2, features1)

        # Cross-Entropy loss
        loss = temp_cont_loss1 + temp_cont_loss2 + \
               0.7 * self.contrastive_loss(temp_cont_lstm_feat1, temp_cont_lstm_feat2)

        loss.backward()
        self.optimizer.step()

        return {'Total_loss': loss.item()}, \
               [self.feature_extractor, self.temporal_encoder, self.classifier]


class clsTran(Algorithm):
    def __init__(self, backbone_fe, backbone_temporal, classifier, configs, hparams, device):
        super(clsTran, self).__init__(configs)

        self.feature_extractor = backbone_fe(configs)
        self.temporal_encoder = backbone_temporal(hparams)
        self.classifier = nn.Linear(hparams["clf"], configs.num_clsTran_tasks)
        self.network = nn.Sequential(self.feature_extractor, self.temporal_encoder, self.classifier)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"],
            betas=(0.9, 0.99)
        )

        self.hparams = hparams

    def update(self, samples):
        # ====== Data =====================
        data = samples["transformed_samples"].float()
        labels = samples["aux_labels"].long()

        self.optimizer.zero_grad()

        features = self.feature_extractor(data)
        features = features.flatten(1, 2)
        
        logits = self.classifier(features)

        # Cross-Entropy loss
        loss = self.cross_entropy(logits, labels)

        loss.backward()
        self.optimizer.step()

        return {'Total_loss': loss.item()}, \
               [self.feature_extractor, self.temporal_encoder, self.classifier]


class supervised(Algorithm):
    def __init__(self, backbone_fe, backbone_temporal, classifier, configs, hparams):
        super(supervised, self).__init__(configs)

        self.feature_extractor = backbone_fe
        self.temporal_encoder = backbone_temporal
        self.classifier = classifier(configs, hparams)
        self.network = nn.Sequential(self.feature_extractor, self.temporal_encoder, self.classifier)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"],
            betas=(0.9, 0.99)
        )

        self.hparams = hparams

    def update(self, samples):
        # ====== Data =====================
        data = samples['sample_ori'].float()
        labels = samples['class_labels'].long()

        # ====== Source =====================
        self.optimizer.zero_grad()

        # Src original features
        features = self.feature_extractor(data)
        features = self.temporal_encoder(features)
        logits = self.classifier(features)

        # Cross-Entropy loss
        x_ent_loss = self.cross_entropy(logits, labels)

        x_ent_loss.backward()
        self.optimizer.step()

        return {'Total_loss': x_ent_loss.item()}, \
               [self.feature_extractor, self.temporal_encoder, self.classifier]