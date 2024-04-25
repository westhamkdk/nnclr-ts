import os
import numpy as np
import pandas as pd
import math
import random
from datetime import datetime
import pickle
from utils import pkl_load, pad_nan_to_target
from scipy.io.arff import loadarff
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pathlib import Path

UCR_MISSING_PATH = 'Missing_value_and_variable_length_datasets_adjusted'


def load_UCR_with_val(dataset, root_path = None):
    
    if root_path is None:
        pp = 'datasets/UCR'
    else:
        pp = root_path
        
    adjustment_path = Path(os.path.join(pp, UCR_MISSING_PATH))
    if os.path.exists(adjustment_path):
        datasets_to_adjust = [str(x).split('/')[-1] for x in adjustment_path.iterdir() if x.is_dir]
    else:
        datasets_to_adjust = []
        
    if dataset in datasets_to_adjust:        
        train_file = os.path.join(adjustment_path, dataset, dataset+"_TRAIN.tsv")
        val_file = os.path.join(adjustment_path, dataset, dataset+"_VAL.tsv") 
        test_file = os.path.join(adjustment_path, dataset, dataset+"_TEST.tsv")
    else:
        train_file = os.path.join(pp, dataset, dataset + "_TRAIN.tsv")
        val_file = os.path.join(pp, dataset, dataset+"_VAL.tsv")         
        test_file = os.path.join(pp, dataset, dataset + "_TEST.tsv")

    train_df = pd.read_csv(train_file, sep='\t', header=None)
    test_df = pd.read_csv(test_file, sep='\t', header=None)
    # New validation data loading
    val_df = pd.read_csv(val_file, sep='\t', header=None) 

    train_array = np.array(train_df)
    test_array = np.array(test_df)
    # New validation data conversion to array
    val_array = np.array(val_df)

    # Move the labels to {0, ..., L-1}
    labels = np.unique(train_array[:, 0])
    transform = {}
    for i, l in enumerate(labels):
        transform[l] = i

    train_labels_raw = train_array[:, 0]
    test_labels_raw = test_array[:, 0]
    # New validation labels extraction
    val_labels_raw = val_array[:, 0]

    train = train_array[:, 1:].astype(np.float64)
    train_labels = np.vectorize(transform.get)(train_labels_raw)
    test = test_array[:, 1:].astype(np.float64)
    test_labels = np.vectorize(transform.get)(test_labels_raw)
    # New validation data and labels transformation
    val = val_array[:, 1:].astype(np.float64)
    val_labels = np.vectorize(transform.get)(val_labels_raw)

    # Normalization for non-normalized datasets
    if dataset not in [
        'AllGestureWiimoteX',
        'AllGestureWiimoteY',
        'AllGestureWiimoteZ',
        'BME',
        'Chinatown',
        'Crop',
        'EOGHorizontalSignal',
        'EOGVerticalSignal',
        'Fungi',
        'GestureMidAirD1',
        'GestureMidAirD2',
        'GestureMidAirD3',
        'GesturePebbleZ1',
        'GesturePebbleZ2',
        'GunPointAgeSpan',
        'GunPointMaleVersusFemale',
        'GunPointOldVersusYoung',
        'HouseTwenty',
        'InsectEPGRegularTrain',
        'InsectEPGSmallTrain',
        'MelbournePedestrian',
        'PickupGestureWiimoteZ',
        'PigAirwayPressure',
        'PigArtPressure',
        'PigCVP',
        'PLAID',
        'PowerCons',
        'Rock',
        'SemgHandGenderCh2',
        'SemgHandMovementCh2',
        'SemgHandSubjectCh2',
        'ShakeGestureWiimoteZ',
        'SmoothSubspace',
        'UMD'
    ]:
        # Return tuple is modified to include validation data and labels
        return_tuple = (train[..., np.newaxis], train_labels, val[..., np.newaxis], val_labels, test[..., np.newaxis], test_labels)    
        return return_tuple
    
    mean = np.nanmean(train)
    std = np.nanstd(train)
    adjusted_train = (train - mean) / std
    adjusted_test = (test - mean) / std
    # New validation data normalization
    adjusted_val = (val - mean) / std

    # Return tuple is modified to include adjusted validation data and labels
    return_tuple = (adjusted_train[..., np.newaxis], train_labels, adjusted_val[..., np.newaxis], val_labels, adjusted_test[..., np.newaxis], test_labels)
    return return_tuple
    

def load_UCR(dataset, root_path = None, return_with_raw_values = False):
    
    if root_path is None:
        pp = 'datasets/UCR'
    else:
        pp = root_path
        
    adjustment_path = Path(os.path.join(pp, UCR_MISSING_PATH))
    if os.path.exists(adjustment_path):
        datasets_to_adjust = [str(x).split('/')[-1] for x in adjustment_path.iterdir() if x.is_dir]
    else:
        datasets_to_adjust = []
        
    if dataset in datasets_to_adjust:
        train_file = os.path.join(adjustment_path, dataset, dataset+"_TRAIN.tsv")
        test_file = os.path.join(adjustment_path, dataset, dataset+"_TEST.tsv")
    else:
        train_file = os.path.join(pp, dataset, dataset + "_TRAIN.tsv")
        test_file = os.path.join(pp, dataset, dataset + "_TEST.tsv")


    # train_file = os.path.join('datasets/UCR', dataset, dataset + "_TRAIN.tsv")
    # test_file = os.path.join('datasets/UCR', dataset, dataset + "_TEST.tsv")

    # train_file = os.path.join(pp, dataset, dataset + "_TRAIN.tsv")
    # test_file = os.path.join(pp, dataset, dataset + "_TEST.tsv")
    train_df = pd.read_csv(train_file, sep='\t', header=None)
    test_df = pd.read_csv(test_file, sep='\t', header=None)
    train_array = np.array(train_df)
    test_array = np.array(test_df)

    # Move the labels to {0, ..., L-1}
    labels = np.unique(train_array[:, 0])
    transform = {}
    for i, l in enumerate(labels):
        transform[l] = i

    train_labels_raw = train_array[:, 0]
    test_labels_raw = test_array[:, 0]

    train = train_array[:, 1:].astype(np.float64)
    train_labels = np.vectorize(transform.get)(train_labels_raw)
    test = test_array[:, 1:].astype(np.float64)
    test_labels = np.vectorize(transform.get)(test_labels_raw)

    # Normalization for non-normalized datasets
    # To keep the amplitude information, we do not normalize values over
    # individual time series, but on the whole dataset
    if dataset not in [
        'AllGestureWiimoteX',
        'AllGestureWiimoteY',
        'AllGestureWiimoteZ',
        'BME',
        'Chinatown',
        'Crop',
        'EOGHorizontalSignal',
        'EOGVerticalSignal',
        'Fungi',
        'GestureMidAirD1',
        'GestureMidAirD2',
        'GestureMidAirD3',
        'GesturePebbleZ1',
        'GesturePebbleZ2',
        'GunPointAgeSpan',
        'GunPointMaleVersusFemale',
        'GunPointOldVersusYoung',
        'HouseTwenty',
        'InsectEPGRegularTrain',
        'InsectEPGSmallTrain',
        'MelbournePedestrian',
        'PickupGestureWiimoteZ',
        'PigAirwayPressure',
        'PigArtPressure',
        'PigCVP',
        'PLAID',
        'PowerCons',
        'Rock',
        'SemgHandGenderCh2',
        'SemgHandMovementCh2',
        'SemgHandSubjectCh2',
        'ShakeGestureWiimoteZ',
        'SmoothSubspace',
        'UMD'
    ]:
        return_tuple = (train[..., np.newaxis], train_labels, test[..., np.newaxis], test_labels)    
        if return_with_raw_values:
            return_tuple = (*return_tuple, train, train_labels_raw, test, test_labels_raw)
        return return_tuple
    
    mean = np.nanmean(train)
    std = np.nanstd(train)
    adjusted_train = (train - mean) / std
    adjusted_test = (test - mean) / std

    return_tuple = (adjusted_train[..., np.newaxis], train_labels, adjusted_test[..., np.newaxis], test_labels)
    if return_with_raw_values:
        return_tuple = (*return_tuple, train, train_labels_raw, test, test_labels_raw)
    return return_tuple

def load_UCR_aug(dataset, root_path = None, return_with_raw_values = False):
    if root_path is None:
        pp = 'datasets/UCR'
    else:
        pp = root_path

    train_file = os.path.join(pp, dataset, dataset + "_TRAIN.tsv")

    train_df = pd.read_csv(train_file, sep='\t', header=None)

    train_array = np.array(train_df)

    # Move the labels to {0, ..., L-1}
    labels = np.unique(train_array[:, 0])
    transform = {}
    for i, l in enumerate(labels):
        transform[l] = i

    train_labels_raw = train_array[:, 0]

    train = train_array[:, 1:].astype(np.float64)
    train_labels = np.vectorize(transform.get)(train_labels_raw)

    return_tuple = (train[..., np.newaxis], train_labels)    
    if return_with_raw_values:
        return_tuple = (*return_tuple, train, train_labels_raw)
        
    return return_tuple

def load_Epilepsy(dataset, original_root, return_with_raw_values = False):

    # If not splitted, split to train/val/test
    train_file = os.path.join(original_root, dataset + "_TRAIN.tsv")
    val_file = os.path.join(original_root, dataset + "_VAL.tsv")
    test_file = os.path.join(original_root, dataset + "_TEST.tsv")

    # if os.path.exists(train_file):
    train_df = pd.read_csv(train_file, sep='\t', header=None)
    val_df = pd.read_csv(val_file, sep='\t', header=None)
    test_df = pd.read_csv(test_file, sep='\t', header=None)
    # train_file = os.path.join(original_root, dataset + "_TRAIN.csv")
    # val_file = os.path.join(original_root, dataset + "_TRAIN.csv")
    # test_file = os.path.join(original_root, dataset + "_TEST.csv")

    # # if os.path.exists(train_file):
    # train_df = pd.read_csv(train_file,header=None)
    # val_df = pd.read_csv(val_file, header=None)
    # test_df = pd.read_csv(test_file,header=None)

    # else:
    #     # split to train (60%), val (20%), test (20%)
    #     original_df = pd.read_csv(os.path.join(original_root, dataset+'.csv'))
    #     train_len = int(.6 * len(original_df))
    #     val_len = int(.2 * len(original_df))
    #     test_len = len(original_df) - train_len - val_len
    #     from sklearn.model_selection import train_test_split
        
    #     train_df, temp_df = train_test_split(original_df, test_size=0.4, random_state=42)
    #     val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    #     train_df.to_csv(train_file, index = False)
    #     val_df.to_csv(val_file, index = False)
    #     test_df.to_csv(test_file, index = False)

    train_array = np.array(train_df)
    val_array = np.array(val_df)
    test_array = np.array(test_df)
    
    # As only class 1 have seizure, we change other classes as class 2, making it class 1 aginst the rest.
    train_labels_raw = train_array[:, 0].astype(np.int)
    val_labels_raw = val_array[:, 0].astype(np.int)
    test_labels_raw = test_array[:, 0].astype(np.int)
    
    train_labels_raw[train_labels_raw != 1] = 2
    val_labels_raw[val_labels_raw != 1] = 2
    test_labels_raw[test_labels_raw != 1] = 2


    # Move the labels to {0, ..., L-1}
    labels = np.unique(train_labels_raw)
    transform = {}
    for i, l in enumerate(labels):
        transform[l] = i

    train = train_array[:, 1:].astype(np.float64)
    train_labels = np.vectorize(transform.get)(train_labels_raw)
    val = val_array[:, 1:].astype(np.float64)
    val_labels = np.vectorize(transform.get)(val_labels_raw)
    test = test_array[:, 1:].astype(np.float64)
    test_labels = np.vectorize(transform.get)(test_labels_raw)

    # Normalization for non-normalized datasets
    # To keep the amplitude information, we do not normalize values over
    # individual time series, but on the whole dataset
        
    mean = np.nanmean(train)
    std = np.nanstd(train)
    adjusted_train = (train - mean) / std
    adjusted_val = (val - mean) / std
    adjusted_test = (test - mean) / std

    return_tuple = (adjusted_train[..., np.newaxis], train_labels, adjusted_test[..., np.newaxis], test_labels, adjusted_val[..., np.newaxis], val_labels)
    if return_with_raw_values:
        return_tuple = (*return_tuple, train, train_labels_raw, test, test_labels_raw)
    return return_tuple


    
def load_UEA(dataset):
    train_data = loadarff(f'datasets/UEA/{dataset}/{dataset}_TRAIN.arff')[0]
    test_data = loadarff(f'datasets/UEA/{dataset}/{dataset}_TEST.arff')[0]
    
    def extract_data(data):
        res_data = []
        res_labels = []
        for t_data, t_label in data:
            t_data = np.array([ d.tolist() for d in t_data ])
            t_label = t_label.decode("utf-8")
            res_data.append(t_data)
            res_labels.append(t_label)
        return np.array(res_data).swapaxes(1, 2), np.array(res_labels)
    
    train_X, train_y = extract_data(train_data)
    test_X, test_y = extract_data(test_data)
    
    scaler = StandardScaler()
    scaler.fit(train_X.reshape(-1, train_X.shape[-1]))
    train_X = scaler.transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)
    test_X = scaler.transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)
    
    labels = np.unique(train_y)
    transform = { k : i for i, k in enumerate(labels)}
    train_y = np.vectorize(transform.get)(train_y)
    test_y = np.vectorize(transform.get)(test_y)
    return train_X, train_y, test_X, test_y
    
    
def load_forecast_npy(name, univar=False):
    data = np.load(f'datasets/{name}.npy')    
    if univar:
        data = data[: -1:]
        
    train_slice = slice(None, int(0.6 * len(data)))
    valid_slice = slice(int(0.6 * len(data)), int(0.8 * len(data)))
    test_slice = slice(int(0.8 * len(data)), None)
    
    scaler = StandardScaler().fit(data[train_slice])
    data = scaler.transform(data)
    data = np.expand_dims(data, 0)

    pred_lens = [24, 48, 96, 288, 672]
    return data, train_slice, valid_slice, test_slice, scaler, pred_lens, 0

def _get_time_features(dt):
    return np.stack([
        dt.minute.to_numpy(),
        dt.hour.to_numpy(),
        dt.dayofweek.to_numpy(),
        dt.day.to_numpy(),
        dt.dayofyear.to_numpy(),
        dt.month.to_numpy(),
        dt.weekofyear.to_numpy(),
    ], axis=1).astype(np.float)


def load_forecast_csv(name, univar=False):
    # data = pd.read_csv(f'datasets/{name}.csv', index_col='date', parse_dates=True)
    data = pd.read_csv(f'/data1/finance/{name}.csv', index_col='date', parse_dates=True)

    dt_embed = _get_time_features(data.index)
    n_covariate_cols = dt_embed.shape[-1]
    
    if univar:
        if name in ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'):
            data = data[['OT']]
        elif name == 'electricity':
            data = data[['MT_001']]
        else:
            data = data.iloc[:, -1:]
        
    data = data.to_numpy()
    if name == 'ETTh1' or name == 'ETTh2':
        train_slice = slice(None, 12*30*24)
        valid_slice = slice(12*30*24, 16*30*24)
        test_slice = slice(16*30*24, 20*30*24)
    elif name == 'ETTm1' or name == 'ETTm2':
        train_slice = slice(None, 12*30*24*4)
        valid_slice = slice(12*30*24*4, 16*30*24*4)
        test_slice = slice(16*30*24*4, 20*30*24*4)
    else:
        train_slice = slice(None, int(0.6 * len(data)))
        valid_slice = slice(int(0.6 * len(data)), int(0.8 * len(data)))
        test_slice = slice(int(0.8 * len(data)), None)
    
    scaler = StandardScaler().fit(data[train_slice])
    data = scaler.transform(data)
    if name in ('electricity'):
        data = np.expand_dims(data.T, -1)  # Each variable is an instance rather than a feature
    else:
        data = np.expand_dims(data, 0)
    
    if n_covariate_cols > 0:
        dt_scaler = StandardScaler().fit(dt_embed[train_slice])
        dt_embed = np.expand_dims(dt_scaler.transform(dt_embed), 0)
        data = np.concatenate([np.repeat(dt_embed, data.shape[0], axis=0), data], axis=-1)
    
    if name in ('ETTh1', 'ETTh2', 'electricity'):
        pred_lens = [24, 48, 168, 336, 720]
    else:
        pred_lens = [24, 48, 96, 288, 672]
        
    return data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols


def load_anomaly(original_root, name):
    res = pkl_load(os.path.join(original_root, name)+".pkl")
    return res['all_train_data'], res['all_train_labels'], res['all_train_timestamps'], \
           res['all_test_data'],  res['all_test_labels'],  res['all_test_timestamps'], \
           res['delay']


def gen_ano_train_data(all_train_data):
    maxl = np.max([ len(all_train_data[k]) for k in all_train_data ])
    pretrain_data = []
    for k in all_train_data:
        train_data = pad_nan_to_target(all_train_data[k], maxl, axis=0)
        pretrain_data.append(train_data)
    pretrain_data = np.expand_dims(np.stack(pretrain_data), 2)
    return pretrain_data
