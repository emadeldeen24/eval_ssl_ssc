import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")
import torch.nn.functional as F
from einops import rearrange


def add_noise(signal, noise_amount):
    """
    adding noise
    """
    signal = signal.cpu().numpy()
    noise = np.random.normal(1, noise_amount, np.shape(signal)[0])
    noised_signal = signal + noise
    return torch.from_numpy(noised_signal)


def negate(signal):
    """
    negate the signal
    """

    negated_signal = signal * (-1)
    return negated_signal

def jitter(x, sigma=0.8):
    if len(x.shape) == 2:
        x = x.unsqueeze(0)
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)

def ts_tcc_scaling(x, sigma=1.1):
    # https://arxiv.org/pdf/1706.00527.pdf
    factor = np.random.normal(loc=2., scale=sigma, size=(x.shape[0], x.shape[2]))
    ai = []
    for i in range(x.shape[1]):
        xi = x[:, i, :]
        ai.append(np.multiply(xi, factor[:, :])[:, np.newaxis, :])
    return torch.from_numpy(np.concatenate((ai), axis=1))

def scaling(x, sigma=1.1):
    return x*sigma

def masking(x, num_splits=10, masking_ratio=0.5):
    num_masked = int(masking_ratio * num_splits)
    patches = rearrange(x, 'b (p l) -> b p l', p=num_splits)
    masked_patches = patches.clone()
    # calculate of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked
    rand_indices = torch.rand(x.shape[0], num_splits).argsort(dim=-1)
    selected_indices = rand_indices[:, :num_masked]
    masked_patches[0, (selected_indices[0, :]), :] = 0
    masked_x = rearrange(masked_patches, 'b p l -> b (p l)', p=num_splits)
    return masked_x


def permutation(x, max_segments=5, seg_mode="random"):
    if len(x.shape) == 2:
        x = x.unsqueeze(0)
    orig_steps = np.arange(x.shape[2])

    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[2] - 2, num_segs[i] - 1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            ret[i] = pat[0,warp]
        else:
            ret[i] = pat
        
    return torch.from_numpy(ret)

def time_shift(x, shift_ratio=0.2):
    if len(x.shape) == 2:
        x = x.unsqueeze(0)
    signal_length = x.shape[2]
    shift = int(signal_length * shift_ratio)
    shifted_sample = np.concatenate((x[:, :, signal_length-shift:], x[:, :, :signal_length-shift]), axis=2)
    return torch.from_numpy(shifted_sample)

def vat_noise(signal, XI=1e-6):
    d = torch.empty(signal.shape).normal_(mean=signal.mean(),std=signal.std())
    d = F.normalize(d.view(d.size(0), -1), p=2, dim=1).reshape(d.size())
    d = XI * d
    return signal + d


def apply_transformation(X_train, aug):
    if not torch.is_tensor(X_train):
        X_train = torch.from_numpy(X_train)
    
    if aug == "noise_permute":
        X0 = vat_noise(X_train,100).unsqueeze(0)
        X1 = permutation(X_train)
    
    elif aug == "noise_timeShift":
        X0 = vat_noise(X_train,100).unsqueeze(0)
        X1 = time_shift(X_train)
        
    elif aug == "noise_negate":
        X0 = vat_noise(X_train,100).unsqueeze(0)
        X1 = negate(X_train).unsqueeze(0)
    
    elif aug == "noise_scale":
        X0 = vat_noise(X_train,100).unsqueeze(0)
        X1 = scaling(X_train, 0.5).unsqueeze(0)
        
    elif aug == "timeShift_negate":
        X0 = time_shift(X_train)
        X1 = negate(X_train).unsqueeze(0)
        
    elif aug == "timeShift_scale":
        X0 = time_shift(X_train)
        X1 = scaling(X_train, 0.5).unsqueeze(0)
        
    elif aug == "timeShift_permute":
        X0 = time_shift(X_train)
        X1 = permutation(X_train)
        
    elif aug == "negate_scale":
        X0 = negate(X_train).unsqueeze(0)
        X1 = scaling(X_train, 0.5).unsqueeze(0)
        
    elif aug == "negate_permute":
        X0 = negate(X_train).unsqueeze(0)
        X1 = permutation(X_train)
        
    elif aug == "scale_permute":
        X0 = permutation(X_train)
        X1 = scaling(X_train, 0.5).unsqueeze(0)
        
    elif aug == "negate":
        X0 = negate(X_train).unsqueeze(0)
        X1 = negate(X_train).unsqueeze(0)
        
    elif aug == "scale":
        X0 = scaling(X_train, 0.5).unsqueeze(0)
        X1 = scaling(X_train, 0.5).unsqueeze(0)
        
    elif aug == "timeShift":
        X0 = time_shift(X_train)
        X1 = time_shift(X_train)
        
    elif aug == "permute":
        X0 = permutation(X_train)
        X1 = permutation(X_train)
        
    elif aug == "noise":
        X0 = vat_noise(X_train,100).unsqueeze(0)
        X1 = vat_noise(X_train,100).unsqueeze(0)
    
    elif aug == "noise_timeShift_permute":
        X0 = vat_noise(X_train,100).unsqueeze(0)
        X1 = permutation(X_train)
        X2 = time_shift(X_train)
    
    elif aug == "noise_timeShift_scale":
        X0 = vat_noise(X_train,100).unsqueeze(0)
        X1 = scaling(X_train, 0.5).unsqueeze(0)
        X2 = time_shift(X_train)
    elif aug == "permute_timeShift_scale":
        X0 = permutation(X_train)
        X1 = scaling(X_train, 0.5).unsqueeze(0)
        X2 = time_shift(X_train)
    elif aug == "permute_timeShift_scale_noise":
        X0 = permutation(X_train)
        X1 = scaling(X_train, 0.5).unsqueeze(0)
        X2 = time_shift(X_train)
        X3 = vat_noise(X_train,100).unsqueeze(0)
    
    elif aug == "tsTcc_aug":
        X0 = jitter(permutation(X_train), 2)
        X1 = ts_tcc_scaling(X_train.unsqueeze(0), 1.5)
    
    elif aug== "mask_noise":
        X0 = vat_noise(X_train,100).unsqueeze(0)
        X1 = masking(X_train).unsqueeze(0)
    
    elif aug== "mask_permute":
        X0 = permutation(X_train)
        X1 = masking(X_train).unsqueeze(0)
    
    elif aug == "mask_Noisepermute":
        X0 =  permutation(vat_noise(X_train,100).unsqueeze(0))
        X1 = masking(X_train).unsqueeze(0)
        
        
        
    X0 = X0.float()
    X1 = X1.float()
    
    if X1.shape != 3:
        X1 = X1.squeeze(X1.shape.index(1))
    if X0.shape != 3:
        X0 = X0.squeeze(X0.shape.index(1))
    
    if len(aug.split("_")) == 3:
        X2 = X2.float()
        if X2.shape != 3:
            X2 = X2.squeeze(X2.shape.index(1))
        return [X0, X1, X2]
    
    if len(aug.split("_")) == 4:
        X2 = X2.float()
        X3 = X3.float()
        if X2.shape != 3:
            X2 = X2.squeeze(X2.shape.index(1))
        if X3.shape != 3:
            X3 = X3.squeeze(X3.shape.index(1))
        return [X0, X1, X2, X3]

    return [X0, X1]
