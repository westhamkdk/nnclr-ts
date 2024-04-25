from turtle import forward
import torch 
import torch.nn as nn
import pandas as pd
import numpy as np
import math
import torch.nn.functional as F
from tqdm import tqdm
import timeit 
from statsmodels.tsa.seasonal import STL
import scipy.interpolate as interpolate
import pickle
import pickle as pkl
import os
from pathlib import Path
from functools import partial




class DataAugmentation(nn.Module):
    def __init__(self, aug_num = 4, jitter = 0.2):
        super().__init__()
        # define jittering here or whatever
        self.aug_num = aug_num
        self.jitter = jitter

    @torch.no_grad()
    def forward(self, x):
        if self.jitter != 0.0:
            x = self.perform_transform(x)
            
        return x

    def perform_transform(self, seq_x):
        seq_x = seq_x.cpu().numpy()
        seq_x = np.expand_dims(seq_x, axis = 0)
        seq_x = seq_x.repeat(self.aug_num, axis=0)
        _, B, L, D = seq_x.shape
        for i in range(self.aug_num -1):
            rand_aug = int(np.rint(np.random.rand(1) * 3)[0])
            if 0 <= rand_aug < 1:
                # Scaling
                seq_x[i+1, :] *= 1 + (np.random.rand(B, L, D) - 0.5) * self.jitter
            elif 1<= rand_aug < 2:
                # Entirety Scaling
                seq_x[i+1, :] *= 1 + (np.random.rand(1) - 0.5)[0] * self.jitter
            else:
                # Jittering
                seq_x[i+1, :] += (np.random.rand(B, L, D) - 0.5) * self.jitter

        B, A, L, D = seq_x.shape
        seq_x = seq_x.reshape(-1, L, D)
        return torch.from_numpy(seq_x).float().cuda()


def window_warping(x, slicing_ratio = 0.1, multiplier = 2):

    if type(x) is np.ndarray:
        x = torch.Tensor(x)

    original_length = x.shape[-1]
    slice_length = math.ceil(slicing_ratio * original_length)
    cutoff_max = slice_length * math.ceil(multiplier - 1)

    mul = np.random.choice([1/multiplier, multiplier])

    rands = torch.randint(0, original_length - cutoff_max, (x.shape[0],))
    rands_end = rands + slice_length
    slices = torch.cat([rands.reshape(-1, 1), rands_end.reshape(-1, 1)], axis = 1)


    mod_list = []
    for i, (a,b) in enumerate(slices.numpy()):
        modified = torch.cat([x[i, :, :a].unsqueeze(0), F.interpolate(torch.unsqueeze(x[i, :, a:b],0), size = math.ceil(mul*slice_length)), x[i, :, b:].unsqueeze(0)], axis = 2)
        mod_list.append(modified)


    mod_cat = torch.cat(mod_list, axis=0)
    interpolated = F.interpolate(mod_cat, size = original_length)

    return interpolated

def window_slicing(x, slicing_ratio = 0.9, leave_last = True):
    if type(x) is np.ndarray:
        x = torch.Tensor(x)


    original_length = x.shape[-1]
    slice_length = math.ceil(slicing_ratio * original_length)

    if leave_last:
        cutoff_max = original_length - math.ceil(slicing_ratio * original_length) - 1
        sliced_x = x[:,:, cutoff_max:]
        interpolated = F.interpolate(sliced_x, size = original_length)
        return interpolated
    else:
        cutoff_max = original_length - math.ceil(slicing_ratio * original_length) - 1

        rands = torch.randint(0, cutoff_max, (x.shape[0],))
        rands1 = rands + slice_length

        slices = torch.cat([rands.reshape(-1, 1), rands1.reshape(-1, 1)], axis = 1)
        all_aranged = []
        for a, b in slices.numpy():
            filled = np.arange(a, b)
            all_aranged.append(filled)
        all_aranged = torch.from_numpy(np.array(all_aranged)).to("cuda")

        x = x.squeeze()
        sliced_x = torch.gather(x, 1, all_aranged)
        sliced_x = torch.unsqueeze(sliced_x, 1)
        interpolated = F.interpolate(sliced_x, size = original_length)
        return interpolated

def jitter(x, sigma=0.03):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)

def scaling(x, sigma=0.1):
    # https://arxiv.org/pdf/1706.00527.pdf
    factor = np.random.normal(loc=1., scale=sigma, size=(x.shape[0],x.shape[-1]))
    return np.multiply(x, factor)

def rotation(x):
    flip = np.random.choice([-1, 1], size=(x.shape[0],x.shape[-1]))
    rotate_axis = np.arange(x.shape[-1])
    np.random.shuffle(rotate_axis)    
    return torch.Tensor(flip) * x

def permutation_original(x, max_segments=5, seg_mode="equal"):
    orig_steps = np.arange(x.shape[0])
    
    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))
    
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[0]-2, num_segs[i]-1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            ret[i] = pat[warp]
        else:
            ret[i] = pat
    return ret

def permutation(tensor, max_segments = 5, seg_mode = "random"):
    # tensor to numpy for easy operation
    arr = tensor.numpy().squeeze()

    # decide number of segments
    num_segments = np.random.randint(1, max_segments + 1)

    # decide the cutting points in the tensor
    cut_points = np.random.choice(np.arange(1, len(arr)), num_segments - 1, replace=False)
    cut_points.sort()

    # split the array and rearrange
    splits = np.split(arr, cut_points)
    np.random.shuffle(splits)

    # join the splits and reshape to original shape
    result = np.concatenate(splits).reshape(tensor.shape)

    # convert back to tensor
    result_tensor = torch.from_numpy(result).float()

    return result_tensor

def magnitude_warp(x, sigma=0.2, knot=4):
    from scipy.interpolate import CubicSpline
    orig_steps = np.arange(x.shape[1])
    
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot+2, x.shape[2]))
    warp_steps = (np.ones((x.shape[2],1))*(np.linspace(0, x.shape[1]-1., num=knot+2))).T
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        warper = np.array([CubicSpline(warp_steps[:,dim], random_warps[i,:,dim])(orig_steps) for dim in range(x.shape[2])]).T
        ret[i] = pat * warper

    return ret

def time_warp(x, sigma=0.2, knot=4):
    from scipy.interpolate import CubicSpline
    orig_steps = np.arange(x.shape[1])
    
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot+2, x.shape[2]))
    warp_steps = (np.ones((x.shape[2],1))*(np.linspace(0, x.shape[1]-1., num=knot+2))).T
    
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            time_warp = CubicSpline(warp_steps[:,dim], warp_steps[:,dim] * random_warps[i,:,dim])(orig_steps)
            scale = (x.shape[1]-1)/time_warp[-1]
            ret[i,:,dim] = np.interp(orig_steps, np.clip(scale*time_warp, 0, x.shape[1]-1), pat[:,dim]).T
    return ret


def fft_addremove(x, y, ratio = 0.1, mode = 'add'):
    fft_transformed = torch.view_as_real(torch.fft.fft(torch.Tensor(x)))[...,0]

    def remove_frequency(x, maskout_ratio=0):
        mask = torch.cuda.FloatTensor(x.shape).uniform_() > maskout_ratio # maskout_ratio are False
        mask = mask.to(x.device)
        return x*mask

    def add_frequency(x, pertub_ratio=0,):
        mask = torch.cuda.FloatTensor(x.shape).uniform_() > (1-pertub_ratio) # only pertub_ratio of all values are True
        mask = mask.to(x.device)
        max_amplitude = x.max()
        random_am = torch.rand(mask.shape)*(max_amplitude*0.1)
        pertub_matrix = mask*random_am
        return x+pertub_matrix
    
    if mode == 'add':
        return torch.view_as_real(torch.fft.ifft(add_frequency(fft_transformed, 0.1)).resolve_conj())[...,0].numpy()
    elif mode == 'remove':
        return torch.view_as_real(torch.fft.ifft(remove_frequency(fft_transformed, 0.1)).resolve_conj())[...,0].numpy()
    else:
        return torch.view_as_real(torch.fft.ifft(add_frequency(fft_transformed, 0.1)).resolve_conj())[...,0].numpy()


def window_slice(x, reduce_ratio=0.9):
    # https://halshs.archives-ouvertes.fr/halshs-01357973/document
    target_len = np.ceil(reduce_ratio*x.shape[1]).astype(int)
    if target_len >= x.shape[1]:
        return x
    starts = np.random.randint(low=0, high=x.shape[1]-target_len, size=(x.shape[0])).astype(int)
    ends = (target_len + starts).astype(int)
    
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            ret[i,:,dim] = np.interp(np.linspace(0, target_len, num=x.shape[1]), np.arange(target_len), pat[starts[i]:ends[i],dim]).T
    return ret

def window_warp(tensor, window_ratio=0.1, scale_range=(0.5, 2)):
    arr = tensor.numpy().squeeze()

    # decide length of the window
    window_length = int(window_ratio * len(arr))

    # decide the start point of the window
    window_start = np.random.randint(0, len(arr) - window_length + 1)

    # decide the scale factor
    scale_factor = np.random.uniform(scale_range[0], scale_range[1])

    # apply the window warping
    arr[window_start:window_start+window_length] *= scale_factor

    # convert back to tensor
    result_tensor = torch.from_numpy(arr).float().view(tensor.shape)

    return result_tensor



def window_warp_original(x, window_ratio=0.1, scales=[0.5, 2.]):
    # https://halshs.archives-ouvertes.fr/halshs-01357973/document
    warp_scales = np.random.choice(scales, x.shape[0])
    warp_size = np.ceil(window_ratio*x.shape[1]).astype(int)
    window_steps = np.arange(warp_size)
        
    window_starts = np.random.randint(low=1, high=x.shape[1]-warp_size-1, size=(x.shape[0])).astype(int)
    window_ends = (window_starts + warp_size).astype(int)
            
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            start_seg = pat[:window_starts[i],dim]
            window_seg = np.interp(np.linspace(0, warp_size-1, num=int(warp_size*warp_scales[i])), window_steps, pat[window_starts[i]:window_ends[i],dim])
            end_seg = pat[window_ends[i]:,dim]
            warped = np.concatenate((start_seg, window_seg, end_seg))                
            ret[i,:,dim] = np.interp(np.arange(x.shape[1]), np.linspace(0, x.shape[1]-1., num=warped.size), warped).T
    return ret

def spawner(x, labels, sigma=0.05, verbose=0):
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6983028/
    # use verbose=-1 to turn off warnings
    # use verbose=1 to print out figures
    
    import utils_dtw.dtw as dtw
    random_points = np.random.randint(low=1, high=x.shape[1]-1, size=x.shape[0])
    window = np.ceil(x.shape[1] / 10.).astype(int)
    orig_steps = np.arange(x.shape[1])
    l = np.argmax(labels, axis=1) if labels.ndim > 1 else labels
    
    ret = np.zeros_like(x)
    pbar = tqdm(x)
    for i, pat in enumerate(pbar):
        pbar.set_description("Processing %s" % "spawner")
        # guarentees that same one isnt selected
        choices = np.delete(np.arange(x.shape[0]), i)
        # remove ones of different classes
        choices = np.where(l[choices] == l[i])[0]
        if choices.size > 0:     
            random_sample = x[np.random.choice(choices)]
            # SPAWNER splits the path into two randomly
            path1 = dtw.dtw(pat[:random_points[i]], random_sample[:random_points[i]], dtw.RETURN_PATH, slope_constraint="symmetric", window=window)
            path2 = dtw.dtw(pat[random_points[i]:], random_sample[random_points[i]:], dtw.RETURN_PATH, slope_constraint="symmetric", window=window)
            combined = np.concatenate((np.vstack(path1), np.vstack(path2+random_points[i])), axis=1)
            if verbose:
                print(random_points[i])
                dtw_value, cost, DTW_map, path = dtw.dtw(pat, random_sample, return_flag = dtw.RETURN_ALL, slope_constraint=slope_constraint, window=window)
                dtw.draw_graph1d(cost, DTW_map, path, pat, random_sample)
                dtw.draw_graph1d(cost, DTW_map, combined, pat, random_sample)
            mean = np.mean([pat[combined[0]], random_sample[combined[1]]], axis=0)
            for dim in range(x.shape[2]):
                ret[i,:,dim] = np.interp(orig_steps, np.linspace(0, x.shape[1]-1., num=mean.shape[0]), mean[:,dim]).T
        else:
            if verbose > -1:
                print("There is only one pattern of class %d, skipping pattern average"%l[i])
            ret[i,:] = pat
    return jitter(ret, sigma=sigma)

def wdba(x, labels, batch_size=6, slope_constraint="symmetric", use_window=True, verbose=0):
    # https://ieeexplore.ieee.org/document/8215569
    # use verbose = -1 to turn off warnings    
    # slope_constraint is for DTW. "symmetric" or "asymmetric"
    
    import utils_dtw.dtw as dtw
    
    if use_window:
        window = np.ceil(x.shape[1] / 10.).astype(int)
    else:
        window = None
        
        
    orig_steps = np.arange(x.shape[1])
    l = np.argmax(labels, axis=1) if labels.ndim > 1 else labels
        
    ret = np.zeros_like(x)
    pbar = tqdm(range(ret.shape[0]))
    for i in pbar:
        pbar.set_description("processing wdba")
        # get the same class as i
        choices = np.where(l == l[i])[0]
        if choices.size > 0:        
            # pick random intra-class pattern
            k = min(choices.size, batch_size)
            random_prototypes = x[np.random.choice(choices, k, replace=False)]
            
            # calculate dtw between all
            dtw_matrix = np.zeros((k, k))
            for p, prototype in enumerate(random_prototypes):
                for s, sample in enumerate(random_prototypes):
                    if p == s:
                        dtw_matrix[p, s] = 0.
                    else:
                        dtw_matrix[p, s] = dtw.dtw(prototype, sample, dtw.RETURN_VALUE, slope_constraint=slope_constraint, window=window)
                        
            # get medoid
            medoid_id = np.argsort(np.sum(dtw_matrix, axis=1))[0]
            nearest_order = np.argsort(dtw_matrix[medoid_id])
            medoid_pattern = random_prototypes[medoid_id]
            
            # start weighted DBA
            average_pattern = np.zeros_like(medoid_pattern)
            weighted_sums = np.zeros((medoid_pattern.shape[0]))
            for nid in nearest_order:
                if nid == medoid_id or dtw_matrix[medoid_id, nearest_order[1]] == 0.:
                    average_pattern += medoid_pattern 
                    weighted_sums += np.ones_like(weighted_sums) 
                else:
                    path = dtw.dtw(medoid_pattern, random_prototypes[nid], dtw.RETURN_PATH, slope_constraint=slope_constraint, window=window)
                    dtw_value = dtw_matrix[medoid_id, nid]
                    warped = random_prototypes[nid, path[1]]
                    weight = np.exp(np.log(0.5)*dtw_value/dtw_matrix[medoid_id, nearest_order[1]])
                    average_pattern[path[0]] += weight * warped
                    weighted_sums[path[0]] += weight
            
            ret[i,:] = average_pattern / weighted_sums[:,np.newaxis]
        else:
            if verbose > -1:
                print("There is only one pattern of class %d, skipping pattern average"%l[i])
            ret[i,:] = x[i]
    return ret

# Proposed

def random_guided_warp(x, labels, slope_constraint="symmetric", use_window=True, dtw_type="normal", verbose=0):
    # use verbose = -1 to turn off warnings
    # slope_constraint is for DTW. "symmetric" or "asymmetric"
    # dtw_type is for shapeDTW or DTW. "normal" or "shape"
    
    import utils_dtw.dtw as dtw
    
    if use_window:
        window = np.ceil(x.shape[1] / 10.).astype(int)
    else:
        window = None
        
    # window 1/10 사이즈로 지정함
    orig_steps = np.arange(x.shape[1])
    l = np.argmax(labels, axis=1) if labels.ndim > 1 else labels
    
    ret = np.zeros_like(x)
    pbar = tqdm(x)
    for i, pat in enumerate(pbar):
        pbar.set_description("processing random guided warp")
        # guarentees that same one isnt selected
        choices = np.delete(np.arange(x.shape[0]), i)
        # remove ones of different classes
        choices = np.where(l[choices] == l[i])[0] # pick up data with same labels
        if choices.size > 0:        
            # pick random intra-class pattern
            random_prototype = x[np.random.choice(choices)]
            
            if dtw_type == "shape":
                path = dtw.shape_dtw(random_prototype, pat, dtw.RETURN_PATH, slope_constraint=slope_constraint, window=window)
            else:
                path = dtw.dtw(random_prototype, pat, dtw.RETURN_PATH, slope_constraint=slope_constraint, window=window)
                            
            # Time warp
            warped = pat[path[1]]
            for dim in range(x.shape[2]):
                ret[i,:,dim] = np.interp(orig_steps, np.linspace(0, x.shape[1]-1., num=warped.shape[0]), warped[:,dim]).T # 여기서 길이가 안 맞더라도 나중에 강제로 길이를 맞춰버림
        else:
            if verbose > -1:
                print("There is only one pattern of class %d, skipping timewarping"%l[i])
            ret[i,:] = pat
    return ret

def random_guided_warp_shape(x, labels, slope_constraint="symmetric", use_window=True):
    return random_guided_warp(x, labels, slope_constraint, use_window, dtw_type="shape")

def discriminative_guided_warp(x, labels, batch_size=6, slope_constraint="symmetric", use_window=True, dtw_type="normal", use_variable_slice=True, verbose=0):
    # use verbose = -1 to turn off warnings
    # slope_constraint is for DTW. "symmetric" or "asymmetric"
    # dtw_type is for shapeDTW or DTW. "normal" or "shape"
    
    import utils_dtw.dtw as dtw
    
    if use_window:
        window = np.ceil(x.shape[1] / 10.).astype(int)
    else:
        window = None
    orig_steps = np.arange(x.shape[1])
    l = np.argmax(labels, axis=1) if labels.ndim > 1 else labels
    
    positive_batch = np.ceil(batch_size / 2).astype(int)
    negative_batch = np.floor(batch_size / 2).astype(int)
        
    ret = np.zeros_like(x)
    warp_amount = np.zeros(x.shape[0])
    pbar = tqdm(x)
    for i, pat in enumerate(pbar):
        pbar.set_description("processing discriminated guided warp")
        # guarentees that same one isnt selected
        choices = np.delete(np.arange(x.shape[0]), i)
        
        # remove ones of different classes
        positive = np.where(l[choices] == l[i])[0]
        negative = np.where(l[choices] != l[i])[0]
        
        if positive.size > 0 and negative.size > 0:
            pos_k = min(positive.size, positive_batch)
            neg_k = min(negative.size, negative_batch)
            positive_prototypes = x[np.random.choice(positive, pos_k, replace=False)]
            negative_prototypes = x[np.random.choice(negative, neg_k, replace=False)]
                        
            # vector embedding and nearest prototype in one
            pos_aves = np.zeros((pos_k))
            neg_aves = np.zeros((pos_k))
            if dtw_type == "shape":
                for p, pos_prot in enumerate(positive_prototypes):
                    for ps, pos_samp in enumerate(positive_prototypes):
                        if p != ps:
                            pos_aves[p] += (1./(pos_k-1.))*dtw.shape_dtw(pos_prot, pos_samp, dtw.RETURN_VALUE, slope_constraint=slope_constraint, window=window)
                    for ns, neg_samp in enumerate(negative_prototypes):
                        neg_aves[p] += (1./neg_k)*dtw.shape_dtw(pos_prot, neg_samp, dtw.RETURN_VALUE, slope_constraint=slope_constraint, window=window)
                selected_id = np.argmax(neg_aves - pos_aves)
                path = dtw.shape_dtw(positive_prototypes[selected_id], pat, dtw.RETURN_PATH, slope_constraint=slope_constraint, window=window)
            else:
                for p, pos_prot in enumerate(positive_prototypes):
                    for ps, pos_samp in enumerate(positive_prototypes):
                        if p != ps:
                            pos_aves[p] += (1./(pos_k-1.))*dtw.dtw(pos_prot, pos_samp, dtw.RETURN_VALUE, slope_constraint=slope_constraint, window=window)
                    for ns, neg_samp in enumerate(negative_prototypes):
                        neg_aves[p] += (1./neg_k)*dtw.dtw(pos_prot, neg_samp, dtw.RETURN_VALUE, slope_constraint=slope_constraint, window=window)
                selected_id = np.argmax(neg_aves - pos_aves)
                path = dtw.dtw(positive_prototypes[selected_id], pat, dtw.RETURN_PATH, slope_constraint=slope_constraint, window=window)
                   
            # Time warp
            warped = pat[path[1]]
            warp_path_interp = np.interp(orig_steps, np.linspace(0, x.shape[1]-1., num=warped.shape[0]), path[1])
            warp_amount[i] = np.sum(np.abs(orig_steps-warp_path_interp))
            for dim in range(x.shape[2]):
                ret[i,:,dim] = np.interp(orig_steps, np.linspace(0, x.shape[1]-1., num=warped.shape[0]), warped[:,dim]).T
        else:
            if verbose > -1:
                print("There is only one pattern of class %d"%l[i])
            ret[i,:] = pat
            warp_amount[i] = 0.
    if use_variable_slice:
        max_warp = np.max(warp_amount)
        if max_warp == 0:
            # unchanged
            ret = window_slice(ret, reduce_ratio=0.9)
        else:
            for i, pat in enumerate(ret):
                # Variable Sllicing
                ret[i] = window_slice(pat[np.newaxis,:,:], reduce_ratio=0.9+0.1*warp_amount[i]/max_warp)[0]
    return ret

def discriminative_guided_warp_shape(x, labels, batch_size=6, slope_constraint="symmetric", use_window=True):
    return discriminative_guided_warp(x, labels, batch_size, slope_constraint, use_window, dtw_type="shape")


def generate_or_get_stl(x, y, dataset, path = None):
    # trend : trend, seasonal : seasonal, residual : residual
    dataset_path = Path(os.path.join(path, dataset))
    dataset_path.mkdir(parents=True, exist_ok= True)
    
    original_path = os.path.join(path, dataset, 'original.pkl')
    trend_path = os.path.join(path, dataset, 'trend.pkl')
    seasonal_path = os.path.join(path, dataset, 'seasonal.pkl')
    residual_path = os.path.join(path, dataset, 'residual.pkl')
    
    trend_conj_path = os.path.join(path, dataset, 'trend_with_conj.pkl')
    seasonal_conj_path = os.path.join(path, dataset, 'seasonal_with_conj.pkl')
    residual_conj_path = os.path.join(path, dataset, 'residual_with_conj.pkl')
    
    if path == None or not os.path.exists(original_path):
        # generate
        fft_matrix_row = x.shape[0]
        fft_matrix_cols = None
        fft_matrix_res, fft_matrix_trend, fft_matrix_seasonal = None, None, None
        
        fft_matrix_res_with_conj, fft_matrix_trend_with_conj, fft_matrix_seasonal_with_conj = None, None, None
        
        # 이거 변환한거 3차원 데이터로 저장하는거 괜찮을 것 같음
        for i, example in enumerate(range(x.shape[0])):
            ucr_example = x[i, :]
                
            to_look = ucr_example[:, 0]
            to_look = pd.Series(to_look, index=pd.date_range("1-1-1980", periods = len(to_look), freq="H"), name="x")
            original_length = len(to_look)
            to_look_clean = to_look.dropna()

            stl = STL(to_look_clean)
            res = stl.fit()
        
            fft_original = np.fft.fft(res.trend+res.seasonal+res.resid)
            fft_res = np.fft.fft(res.resid)
            fft_trend = np.fft.fft(res.trend)
            fft_seasonal = np.fft.fft(res.seasonal)
            fftfreq_res = np.fft.fftfreq(len(res.resid))    
            
            if fft_matrix_cols is None:
                fft_matrix_cols = x.shape[1]
                fft_matrix_original = np.zeros([fft_matrix_row, fft_matrix_cols])
                
                fft_matrix_res = np.zeros([fft_matrix_row, fft_matrix_cols])
                fft_matrix_trend = np.zeros([fft_matrix_row, fft_matrix_cols])
                fft_matrix_seasonal = np.zeros([fft_matrix_row, fft_matrix_cols])
                fft_matrix_res_with_conj = np.zeros([fft_matrix_row, fft_matrix_cols],dtype=np.complex_)
                fft_matrix_trend_with_conj = np.zeros([fft_matrix_row, fft_matrix_cols],dtype=np.complex_)
                fft_matrix_seasonal_with_conj = np.zeros([fft_matrix_row, fft_matrix_cols],dtype=np.complex_)


            fft_matrix_original[i, :] = np.pad(fft_original.real, (0, original_length - len(res.trend)), 'constant', constant_values=np.nan)
            fft_matrix_res[i, :] = np.pad(fft_res.real, (0, original_length - len(res.trend)), 'constant', constant_values=np.nan)
            fft_matrix_trend[i, :] = np.pad(fft_trend.real, (0, original_length - len(res.trend)), 'constant', constant_values=np.nan)
            fft_matrix_seasonal[i, :] = np.pad(fft_seasonal.real, (0, original_length - len(res.trend)), 'constant', constant_values=np.nan)
            
            
            fft_matrix_res_with_conj[i, :] = np.pad(fft_res, (0, original_length - len(res.trend)), 'constant', constant_values=np.nan)
            fft_matrix_trend_with_conj[i, :] = np.pad(fft_trend, (0, original_length - len(res.trend)), 'constant', constant_values=np.nan)
            fft_matrix_seasonal_with_conj[i, :] = np.pad(fft_seasonal, (0, original_length - len(res.trend)), 'constant', constant_values=np.nan)
            
            
            # fft_matrix_original[i, :] = fft_original.real                
            # fft_matrix_res[i, :] = fft_res.real
            # fft_matrix_trend[i, :] = fft_trend.real
            # fft_matrix_seasonal[i, :] = fft_seasonal.real
            
            
            # fft_matrix_res_with_conj[i, :] = fft_res
            # fft_matrix_trend_with_conj[i, :] = fft_trend
            # fft_matrix_seasonal_with_conj[i, :] = fft_seasonal


        original = fft_matrix_original
        trend = fft_matrix_trend
        seasonal = fft_matrix_seasonal
        residual = fft_matrix_res
        
        trend_with_conj = fft_matrix_trend_with_conj
        seasonal_with_conj = fft_matrix_seasonal_with_conj
        residual_with_conj = fft_matrix_res_with_conj
        
        with open(original_path, 'wb') as f:
            pickle.dump(original, f)
        
        with open(trend_path, 'wb') as f:
            pickle.dump(trend, f)
            
        with open(seasonal_path, 'wb') as f:
            pickle.dump(seasonal, f)
            
        with open(residual_path, 'wb') as f:
            pickle.dump(residual, f)
        
        with open(trend_conj_path, 'wb') as f:
            pickle.dump(trend_with_conj, f)
            
        with open(seasonal_conj_path, 'wb') as f:
            pickle.dump(seasonal_with_conj, f)
            
        with open(residual_conj_path, 'wb') as f:
            pickle.dump(residual_with_conj, f)
            
    else:
        # get
        
        with open(original_path, 'rb') as f:
            original = pickle.load(f)

        with open(trend_path, 'rb') as f:
            trend = pickle.load(f)
            
        with open(seasonal_path, 'rb') as f:
            seasonal = pickle.load(f)

        with open(residual_path, 'rb') as f:
            residual = pickle.load(f)
            
        with open(trend_conj_path, 'rb') as f:
            trend_with_conj = pickle.load(f)
            
        with open(seasonal_conj_path, 'rb') as f:
            seasonal_with_conj = pickle.load(f)

        with open(residual_conj_path, 'rb') as f:
            residual_with_conj = pickle.load(f)
        
            
    return trend, seasonal, residual, trend_with_conj, seasonal_with_conj, residual_with_conj, original

        

def generate_stl_and_inv_cdfs(x, y, dataset = None, stl_path = None, sample_mode = 'SAME', hist_edges = None, verbose = 0):
    # sample_mode ['SAME', 'ALL', 'DIFFERENT']
    labels = np.unique(y)
    label_dict = dict()
    # get indexes 

    if dataset in ['nab_machine_10','nab_machine_50', 'nab_machine_100', 'nab_machine_200', 'yahoo', 'kpi']:
        label_dict[0] = list(range(x.shape[0]))
    
    else:
        if sample_mode == 'SAME':
            for label in labels:
                label_dict[label] = np.where(y == label)
        elif sample_mode == 'DIFFERENT':
            for label in labels:
                label_dict[label] = np.where(y != label)
        elif sample_mode == 'ALL':
            label_dict[0] = np.where(y != None)
            
            
    # get fft data
    fft_matrix_trend, fft_matrix_seasonal, fft_matrix_res, fft_matrix_trend_with_conj, fft_matrix_seasonal_with_conj, fft_matrix_res_with_conj, fft_matrix_res_original = generate_or_get_stl(x, y, dataset, path = stl_path)
    fft_matrix_cols = fft_matrix_res.shape[1]
    fft_matrix_row = x.shape[0]
    
    fft_matrices = [fft_matrix_trend, fft_matrix_res, fft_matrix_seasonal, np.fft.fft((np.fft.ifft(fft_matrix_res_with_conj) + np.fft.ifft(fft_matrix_seasonal_with_conj))).real, fft_matrix_res_original]
    # fft_matrices_complex = [fft_matrix_res_with_conj, fft_matrix_seasonal_with_conj, np.fft.fft((np.fft.ifft(fft_matrix_res) + np.fft.ifft(fft_matrix_seasonal))).real]
    
    # numpy save 해야 할듯
    def sturge_rule(observations):
        import math
        res = math.ceil(1 + 3.322 * math.log(observations))
        return res
    
    
    all_inv_cdfs = []
    for matrix in fft_matrices:
        residual_inv_cdfs = dict.fromkeys(label_dict.keys())

        
        for label in label_dict.keys():
            for j in range(fft_matrix_cols):
                
                dt = matrix[label_dict[label]][:, j]
                
                dt = dt[~np.isnan(dt)]

                if hist_edges == None:
                    bins = sturge_rule(dt.shape[0])
                else:
                    bins = hist_edges
                
                
                hist, bin_edges = np.histogram(dt, bins = bins, density= True)
                
                cum_values = np.zeros(bin_edges.shape)
                cum_values[1:] = np.cumsum(hist * np.diff(bin_edges))
                inv_cdf = interpolate.interp1d(cum_values, bin_edges)
                

                if residual_inv_cdfs[label] == None:
                    residual_inv_cdfs[label] = [inv_cdf]
                else:
                    residual_inv_cdfs[label] = residual_inv_cdfs[label] + [inv_cdf]    
                
        all_inv_cdfs.append(residual_inv_cdfs)
                    
                
    return tuple(all_inv_cdfs)


            
        
        
    

def stl_recomposition(x, labels, use_trend = True, use_seasonal = False, combine_resid = True, exclude_resid = False, sample_from_same_class = True, verbose = 0):
    '''
    A - At, As, Ar
    B - Bt, Bs, Br (from same class)

    At+Bs+(Ar+Br)/2
    Bt+As+(Ar+Br)/2
    At+Bs+Ar
    Bt+As+Ar 
    '''
    orig_steps = np.arange(x.shape[1])
    l = np.argmax(labels, axis=1) if labels.ndim > 1 else labels
    ret = np.zeros_like(x)

    pbar = tqdm(x)
    for i, pat in enumerate(pbar):
        pbar.set_description(f"processing stl_recomposition use trend {use_trend} use seasonal {use_seasonal} combine_resid {combine_resid}")
        # guarentees that same one isnt selected
        choices = np.delete(np.arange(x.shape[0]), i)
        # remove ones of different classes
        if sample_from_same_class:
            choices = np.where(l[choices] == l[i])[0] # pick up data with same labels
        else:
            choices = np.where(l[choices] != l[i])[0] # pick up data with different labels
            
        if choices.size > 0:
            random_prototype = x[np.random.choice(choices)]
            random_prototype = pd.Series(random_prototype[:,0], index = pd.date_range("1-1-1980", periods = len(random_prototype), freq="M"), name="choices")
            pat = pd.Series(pat[:,0], index=pd.date_range("1-1-1980", periods = len(pat), freq="M"), name="pat")
            
            pat_stl = STL(pat)
            pat_res = pat_stl.fit()
            choices_stl = STL(random_prototype)
            choices_res = choices_stl.fit()
            
            t = pat_res.trend if use_trend else choices_res.trend
            s = pat_res.seasonal if use_seasonal else choices_res.seasonal
            r = (pat_res.resid + choices_res.resid) / 2 if combine_resid else pat_res.resid
            
            if exclude_resid:
                recomposed = t+s
            else:
                recomposed = t+s+r
            recomposed = recomposed.to_numpy()
            ret[i, :, 0] = recomposed
        else:
            if verbose > -1:
                print("There is only one pattern of class %d, skipping timewarping"%l[i])
            ret[i,:] = pat
    return ret

    
    
    

def random_walk_path(x, labels, use_window=True, verbose=0, ratio = 0.1, prob = [1/3, 1/3, 1/3], mode = 'static', alpha = 2):
    # 3% 5% 10% 20%
    import utils_dtw.dtw as dtw
    
    if use_window:
        window = np.ceil(x.shape[1] * ratio).astype(int)
    else:
        window = None
        
    # window 1/10 사이즈로 지정함
    orig_steps = np.arange(x.shape[1])
    l = np.argmax(labels, axis=1) if labels.ndim > 1 else labels
    
    ret = np.zeros_like(x)
    pbar = tqdm(x)
    for i, pat in enumerate(pbar):
        pbar.set_description("processing random_walk_path")
        # guarentees that same one isnt selected
        choices = np.delete(np.arange(x.shape[0]), i)
        # remove ones of different classes
        choices = np.where(l[choices] == l[i])[0] # pick up data with same labels
        if choices.size > 0:        
            path = dtw._traceback_randomwalk(pat.shape[0], window=window, prob = prob, mode = mode, alpha = alpha)
            # Time warp
            warped = pat[path]
            for dim in range(x.shape[2]):
                ret[i,:,dim] = np.interp(orig_steps, np.linspace(0, x.shape[1]-1., num=warped.shape[0]), warped[:,dim]).T # 여기서 길이가 안 맞더라도 나중에 강제로 길이를 맞춰버림
        else:
            if verbose > -1:
                print("There is only one pattern of class %d, skipping timewarping"%l[i])
            ret[i,:] = pat
    return ret



if __name__ == '__main__':
    # batch,length,dim
    import datautils
    x_tr, y_tr, x_te, y_te = datautils.load_UCR('GunPoint', root_path = '/data1/finance/UCR')
    # x_tr = np.squeeze(x_tr)
    # x_rgw = random_guided_warp(x_tr, y_tr)
    x_rwp = random_walk_path(x_tr, y_tr)
    print(x_rwp.shape) # 50, 150, 1
    

OTHER_AUGS_DICT = {
    "jittering" : jitter,
    "scaling": scaling,
    "rotation": rotation,
    "permutation": partial(permutation, seg_mode = "random"),
    "window_warp": window_warp,
    "noaug": None,
}

def test_augment_all(x, y):
    # batch, length, dim
    original_x = x
    original_y = y
    
    x_jitter = jitter(x)
    x_scaling = scaling(x)
    x_rotation = rotation(x)
    x_permutation = permutation(x)
    x_permutation_random = permutation(x, seg_mode="random")
    x_mag_warp = magnitude_warp(x)
    x_time_warp = time_warp(x)
    x_ws = window_slice(x)
    x_ww = window_warp(x)
    x_spawner = spawner(x, y)
    x_rgw = random_guided_warp(x, y)
    x_rgws = random_guided_warp_shape(x, y)
    x_wdba = wdba(x, y)
    x_dgw = discriminative_guided_warp(x, y)
    x_dgws = discriminative_guided_warp_shape(x, y)


    return x_dgws, y
    # return original_x, x_jitter, x_scaling, x_rotation, x_permutation, x_permutation_random, x_mag_warp, x_time_warp, x_ws, x_ww, x_spawner, x, x_rgw, x_rgws, x_wdba, x_dgw, x_dgws
