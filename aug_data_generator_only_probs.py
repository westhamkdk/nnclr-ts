import torch
import numpy as np
import argparse
import os
import sys
import time
import datetime
import tasks
import datautils
from utils import init_dl_program, name_with_datetime, pkl_save, data_dropout
import utils_dtw.data_augmentation as aug
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import math
from functools import partial
import pickle as pkl


UCR_MISSING_PATH = 'Missing_value_and_variable_length_datasets_adjusted'
# 0, 1, 42, 1557, 1601, 777, 2453, 1238, 1360, 1843, 1917, 1004, 466, 497, 1317

DTW_AUGS_DICT = {
    # "stl_sampling_sample_mode_ALL_edges_none" : partial(aug.generate_stl_and_inv_cdfs, sample_mode = 'ALL', hist_edges = None),
    # "stl_sampling_sample_mode_SAME_edges_none" : partial(aug.generate_stl_and_inv_cdfs, sample_mode = 'SAME', hist_edges = None),
    # "stl_sampling_sample_mode_DIFFERENT_edges_none" : partial(aug.generate_stl_and_inv_cdfs, sample_mode = 'DIFFERENT', hist_edges = None),
    
    # "stl_sampling_sample_mode_SAME_edges_10" : partial(aug.generate_stl_and_inv_cdfs, sample_mode = 'SAME', hist_edges = 10),
    # "stl_sampling_sample_mode_DIFFERENT_edges_10" : partial(aug.generate_stl_and_inv_cdfs, sample_mode = 'DIFFERENT', hist_edges = 10),
    # "stl_sampling_sample_mode_ALL_edges_10" : partial(aug.generate_stl_and_inv_cdfs, sample_mode = 'ALL', hist_edges = 10),
    
    # "stl_sampling_sample_mode_SAME_edges_5" : partial(aug.generate_stl_and_inv_cdfs, sample_mode = 'SAME', hist_edges = 5),
    # "stl_sampling_sample_mode_DIFFERENT_edges_5" : partial(aug.generate_stl_and_inv_cdfs, sample_mode = 'DIFFERENT', hist_edges = 5),
    "stl_sampling_sample_mode_ALL_edges_5" : partial(aug.generate_stl_and_inv_cdfs, sample_mode = 'ALL', hist_edges = 5),
    
    # "stl_sampling_sample_mode_SAME_edges_20" : partial(aug.generate_stl_and_inv_cdfs, sample_mode = 'SAME', hist_edges = 20),
    # "stl_sampling_sample_mode_DIFFERENT_edges_20" : partial(aug.generate_stl_and_inv_cdfs, sample_mode = 'DIFFERENT', hist_edges = 20),
    "stl_sampling_sample_mode_ALL_edges_20" : partial(aug.generate_stl_and_inv_cdfs, sample_mode = 'ALL', hist_edges = 20),  
    
    # "stl_sampling_sample_mode_SAME_edges_15" : partial(aug.generate_stl_and_inv_cdfs, sample_mode = 'SAME', hist_edges = 15),
    # "stl_sampling_sample_mode_DIFFERENT_edges_15" : partial(aug.generate_stl_and_inv_cdfs, sample_mode = 'DIFFERENT', hist_edges = 15),
    # "stl_sampling_sample_mode_ALL_edges_15" : partial(aug.generate_stl_and_inv_cdfs, sample_mode = 'ALL', hist_edges = 15),
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='/data1/finance/UCR_tvsplit')
    parser.add_argument('--aug_path', type=str, default='/data1/finance/UCR_tvsplit_invcdf')
    parser.add_argument('--loader', type=str, required=True, help='The data loader used to load the experimental data. This can be set to UCR, UEA, forecast_csv, forecast_csv_univar, anomaly, or anomaly_coldstart')
    parser.add_argument('--use_chunk', type=int, default=1)
    parser.add_argument('--data_chunk_id', type=int, default=0)
    parser.add_argument('--max-threads', type=int, default=None, help='The maximum allowed number of threads used by this process')
    parser.add_argument('--dataset', type=str)
    args = parser.parse_args()
    
    
    print("Arguments:", str(args))

    chunk_id = args.data_chunk_id
    args.use_chunk = 0
    aug_root  = args.aug_path
    print('Loading data... ', end='')
    if args.loader == 'UCR':
        root = Path(args.root_path)
        
        all_datasets = [str(x).split('/')[-1] for x in root.iterdir() if x.is_dir]
        if UCR_MISSING_PATH in all_datasets:
            all_datasets.remove(UCR_MISSING_PATH)

        if args.use_chunk:
            chunk_size = math.ceil(len(all_datasets)/10)
            splitted_chunks = [all_datasets[i:i+chunk_size] for i in range(0, len(all_datasets), chunk_size)]
            chunk_dataset = splitted_chunks[chunk_id]
        else:
            chunk_dataset = all_datasets

        adjustment_path = Path(os.path.join(root, UCR_MISSING_PATH))
        if os.path.exists(adjustment_path):
            datasets_to_adjust = [str(x).split('/')[-1] for x in adjustment_path.iterdir() if x.is_dir]
        else:
            datasets_to_adjust = []

        for dataset in tqdm(chunk_dataset):
            if '.DS_Store' in dataset:
                continue
            
            print("Processing Dataset =========> ", dataset)    

            if dataset in datasets_to_adjust:
                new_path = adjustment_path
            else:
                new_path = root
                
            train_data, train_labels, test_data, test_labels, _ , train_labels_raw, _, _ = datautils.load_UCR(dataset, root_path = root, return_with_raw_values=True)

            for aug_name, aug_func in DTW_AUGS_DICT.items():
                print("Augmentation =======> ", aug_name)    
                augment_root_dir = os.path.join(aug_root,aug_name, dataset)
                augment_root_dir_path = Path(augment_root_dir)
                augment_root_dir_path.mkdir(parents = True, exist_ok = True)
                # check if file train / test file exist
                stl_root_path = 'stl_sampling'
                
                inv_cdf_trend, inv_cdf_residual, inv_cdf_seasonal, inv_cdf_remainder, inv_cdf_original = aug_func(train_data, train_labels, dataset, os.path.join(aug_root,stl_root_path))
                inv_cdf_path = os.path.join(aug_root, aug_name, dataset, 'inv_cdfs.pkl')
                with open(inv_cdf_path, 'wb') as f:
                    pkl.dump(inv_cdf_residual, f)
                    
                    
                inv_cdf_path_seasonal = os.path.join(aug_root, aug_name, dataset, 'inv_cdfs_seasonal.pkl')
                with open(inv_cdf_path_seasonal, 'wb') as f:
                    pkl.dump(inv_cdf_seasonal, f)
                    
                inv_cdf_path_trend = os.path.join(aug_root, aug_name, dataset, 'inv_cdfs_trend.pkl')
                with open(inv_cdf_path_trend, 'wb') as f:
                    pkl.dump(inv_cdf_trend, f)
                    
                    
                inv_cdf_path_remainder = os.path.join(aug_root, aug_name, dataset, 'inv_cdfs_remainder.pkl')
                with open(inv_cdf_path_remainder, 'wb') as f:
                    pkl.dump(inv_cdf_remainder, f)

                inv_cdf_path_original = os.path.join(aug_root, aug_name, dataset, 'inv_cdfs_original.pkl')
                with open(inv_cdf_path_original, 'wb') as f:
                    pkl.dump(inv_cdf_original, f)
            
    elif args.loader == "epilepsy":
        task_type = 'classification'
        root = Path(args.root_path)
        dataset = 'epilepsy'

        train_data, train_labels, test_data, test_labels, val_data, val_labels = datautils.load_Epilepsy(dataset, root)
        
        for aug_name, aug_func in DTW_AUGS_DICT.items():
            print("Augmentation =======> ", aug_name)    
            augment_root_dir = os.path.join(aug_root,aug_name, dataset)
            augment_root_dir_path = Path(augment_root_dir)
            augment_root_dir_path.mkdir(parents = True, exist_ok = True)
            # check if file train / test file exist
            stl_root_path = 'stl_sampling'
            
            inv_cdf_trend, inv_cdf_residual, inv_cdf_seasonal, inv_cdf_remainder, inv_cdf_original = aug_func(train_data, train_labels, dataset, os.path.join(aug_root,stl_root_path))
            inv_cdf_path = os.path.join(aug_root, aug_name, dataset, 'inv_cdfs.pkl')
            with open(inv_cdf_path, 'wb') as f:
                pkl.dump(inv_cdf_residual, f)
                
            inv_cdf_path_seasonal = os.path.join(aug_root, aug_name, dataset, 'inv_cdfs_seasonal.pkl')
            with open(inv_cdf_path_seasonal, 'wb') as f:
                pkl.dump(inv_cdf_seasonal, f)
                
            inv_cdf_path_trend = os.path.join(aug_root, aug_name, dataset, 'inv_cdfs_trend.pkl')
            with open(inv_cdf_path_trend, 'wb') as f:
                pkl.dump(inv_cdf_trend, f)
                
            inv_cdf_path_remainder = os.path.join(aug_root, aug_name, dataset, 'inv_cdfs_remainder.pkl')
            with open(inv_cdf_path_remainder, 'wb') as f:
                pkl.dump(inv_cdf_remainder, f)

            inv_cdf_path_original = os.path.join(aug_root, aug_name, dataset, 'inv_cdfs_original.pkl')
            with open(inv_cdf_path_original, 'wb') as f:
                pkl.dump(inv_cdf_original, f)

    elif args.loader == 'UEA':
        task_type = 'classification'
        train_data, train_labels, test_data, test_labels = datautils.load_UEA(args.dataset)

            

        
    elif args.loader == 'forecast_csv':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_csv(args.dataset)
        train_data = data[:, train_slice]
        
    elif args.loader == 'forecast_csv_univar':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_csv(args.dataset, univar=True)
        train_data = data[:, train_slice]
        
    elif args.loader == 'forecast_npy':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_npy(args.dataset)
        train_data = data[:, train_slice]
        
    elif args.loader == 'forecast_npy_univar':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_npy(args.dataset, univar=True)
        train_data = data[:, train_slice]
        
    elif args.loader == 'anomaly':
        task_type = 'anomaly_detection'
        all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay = datautils.load_anomaly(args.root_path, args.dataset)
        train_data = datautils.gen_ano_train_data(all_train_data)

        dataset = args.dataset
        for aug_name, aug_func in DTW_AUGS_DICT.items():
            print("Augmentation =======> ", aug_name)    
            augment_root_dir = os.path.join(aug_root,aug_name, dataset)
            augment_root_dir_path = Path(augment_root_dir)
            augment_root_dir_path.mkdir(parents = True, exist_ok = True)
            # check if file train / test file exist
            stl_root_path = 'stl_sampling'
            
            inv_cdf_trend, inv_cdf_residual, inv_cdf_seasonal, inv_cdf_remainder, inv_cdf_original = aug_func(train_data, all_train_labels, dataset, os.path.join(aug_root,stl_root_path))
            inv_cdf_path = os.path.join(aug_root, aug_name, dataset, 'inv_cdfs.pkl')
            with open(inv_cdf_path, 'wb') as f:
                pkl.dump(inv_cdf_residual, f)
                
            inv_cdf_path_seasonal = os.path.join(aug_root, aug_name, dataset, 'inv_cdfs_seasonal.pkl')
            with open(inv_cdf_path_seasonal, 'wb') as f:
                pkl.dump(inv_cdf_seasonal, f)
                
            inv_cdf_path_trend = os.path.join(aug_root, aug_name, dataset, 'inv_cdfs_trend.pkl')
            with open(inv_cdf_path_trend, 'wb') as f:
                pkl.dump(inv_cdf_trend, f)
                
                
            inv_cdf_path_remainder = os.path.join(aug_root, aug_name, dataset, 'inv_cdfs_remainder.pkl')
            with open(inv_cdf_path_remainder, 'wb') as f:
                pkl.dump(inv_cdf_remainder, f)

            inv_cdf_path_original = os.path.join(aug_root, aug_name, dataset, 'inv_cdfs_original.pkl')
            with open(inv_cdf_path_original, 'wb') as f:
                pkl.dump(inv_cdf_original, f)
            

        
    elif args.loader == 'anomaly_coldstart':
        task_type = 'anomaly_detection_coldstart'
        all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay = datautils.load_anomaly(args.dataset)
        train_data, _, _, _ = datautils.load_UCR('FordA')
        
    else:
        raise ValueError(f"Unknown loader {args.loader}.")
        
    print('done')