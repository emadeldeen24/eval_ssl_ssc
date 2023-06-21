import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms

import os
import numpy as np
from .ts_augmentations import apply_transformation


class Load_Dataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, dataset, normalize, train_mode, ssl_method, augmentation, oversample=False):
        super(Load_Dataset, self).__init__()
        self.train_mode = train_mode
        self.ssl_method = ssl_method
        self.augmentation = augmentation

        X_train = dataset["samples"]
        y_train = dataset["labels"]

        if oversample and "ft" not in train_mode:  # if fine-tuning, it shouldn't be on oversampled data
            X_train, y_train = get_balance_class_oversample(X_train, y_train)

        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)

        if isinstance(X_train, np.ndarray):
            X_train = torch.from_numpy(X_train)
            y_train = torch.from_numpy(y_train).long()

        if X_train.shape.index(min(X_train.shape)) != 1:  # make sure the Channels in second dim
            X_train = X_train.permute(0, 2, 1)

        self.x_data = X_train
        self.y_data = y_train

        self.num_channels = X_train.shape[1]

        # Normalize data
        if normalize:
            data_mean = torch.mean(self.x_data, dim=(0, 2))
            data_std = torch.std(self.x_data, dim=(0, 2))
            self.transform = transforms.Normalize(mean=data_mean, std=data_std)
        else:
            self.transform = None

        self.len = X_train.shape[0]

        self.num_transformations = len(self.augmentation.split("_"))

    def __getitem__(self, index):
        x = self.x_data[index]
        if self.transform:
            x = self.transform(self.x_data[index].reshape(self.num_channels, -1, 1)).reshape(self.x_data[index].shape)
        y = self.y_data[index] if self.y_data is not None else None
        # return x, y

        if self.train_mode == "ssl" and self.ssl_method in ["simclr", "ts_tcc", "cpc"]:
            if self.ssl_method == "ts_tcc":  # TS-TCC has its own augmentations
                self.augmentation = "tsTcc_aug"
            elif self.ssl_method == "clsTran":
                self.augmentation = "permute_timeShift_scale_noise"
            transformed_samples = apply_transformation(x, self.augmentation)
            sample = {
                'transformed_samples': transformed_samples,
                'sample_ori': x.squeeze(-1)
            }

        elif self.train_mode == "ssl" and self.ssl_method == "clsTran":
            transformed_samples = apply_transformation(x, self.augmentation)
            order = np.random.randint(self.num_transformations)
            transformed_sample = transformed_samples[order]
            sample = {
                'transformed_samples': transformed_sample,
                'aux_labels': int(order),
                'sample_ori': x.squeeze(-1)
            }
        else:
            sample = {
                'sample_ori': x.squeeze(-1),
                'class_labels': int(y)
            }

        return sample

    def __len__(self):
        return self.len


def data_generator(data_path, data_type, fold_id, data_percentage, dataset_configs, hparams, train_mode, ssl_method,
                   augmentation, oversample):
    # original
    train_dataset = torch.load(
        os.path.join(data_path, data_type, f"train_{fold_id}_{data_percentage}per.pt"))
    val_dataset = torch.load(os.path.join(data_path, data_type, f"val_{fold_id}.pt"))

    # Loading datasets
    train_dataset = Load_Dataset(train_dataset, dataset_configs.normalize, train_mode, ssl_method, augmentation,
                                 oversample)
    val_dataset = Load_Dataset(val_dataset, dataset_configs.normalize, train_mode, ssl_method, augmentation, oversample)

    # Dataloaders
    batch_size = hparams["batch_size"]
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                               shuffle=True, drop_last=True, num_workers=0)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size,
                                             shuffle=False, drop_last=dataset_configs.drop_last, num_workers=0)
    return train_loader, val_loader


def get_balance_class_oversample(x, y):
    """
    from deepsleepnet https://github.com/akaraspt/deepsleepnet/blob/master/deepsleep/utils.py
    Balance the number of samples of all classes by (oversampling):
        1. Find the class that has the largest number of samples
        2. Randomly select samples in each class equal to that largest number
    """
    class_labels = np.unique(y)
    n_max_classes = -1
    for c in class_labels:
        n_samples = len(np.where(y == c)[0])
        if n_max_classes < n_samples:
            n_max_classes = n_samples

    balance_x = []
    balance_y = []
    for c in class_labels:
        idx = np.where(y == c)[0]
        n_samples = len(idx)
        n_repeats = int(n_max_classes / n_samples)
        tmp_x = np.repeat(x[idx], n_repeats, axis=0)
        tmp_y = np.repeat(y[idx], n_repeats, axis=0)
        n_remains = n_max_classes - len(tmp_x)
        if n_remains > 0:
            sub_idx = np.random.permutation(idx)[:n_remains]
            tmp_x = np.vstack([tmp_x, x[sub_idx]])
            tmp_y = np.hstack([tmp_y, y[sub_idx]])
        balance_x.append(tmp_x)
        balance_y.append(tmp_y)
    balance_x = np.vstack(balance_x)
    balance_y = np.hstack(balance_y)

    return balance_x, balance_y


# For AttnSleep CA loss function
import math


def calc_class_weight(labels_count):
    '''
    from https://github.com/emadeldeen24/AttnSleep/blob/main/data_loader/data_loaders.py
    To generate the CA loss function
    '''
    total = np.sum(labels_count)
    class_weight = dict()
    num_classes = len(labels_count)

    factor = 1 / num_classes
    mu = [factor * 1.5, factor * 2, factor * 1.5, factor, factor * 1.5]  # THESE CONFIGS ARE FOR SLEEP-EDF-20 ONLY

    for key in range(num_classes):
        score = math.log(mu[key] * total / float(labels_count[key]))
        class_weight[key] = score if score > 1.0 else 1.0
        class_weight[key] = round(class_weight[key] * mu[key], 2)

    class_weight = [class_weight[i] for i in range(num_classes)]

    return class_weight
