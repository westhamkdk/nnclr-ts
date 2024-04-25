import torch
import numpy as np
import argparse
import os
import sys
import time
import datetime
from ts2vec import TS2Vec
# from nnclr import NNCLR
from nnclr_working import NNCLR
from nnclr_anomaly import NNCLR_anomaly_self_supervised_version
import tasks
import datautils
from utils import init_dl_program, name_with_datetime, pkl_save, data_dropout
import utils_dtw.data_augmentation as aug
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from mixup_contrastive_example import MixUp
import mixup_contrastive_example
import pickle as pkl
import random
from utils_dtw.data_augmentation import OTHER_AUGS_DICT
from statsmodels.tsa.seasonal import STL 
import pickle


# UCR_minisets = ['ElectricDevices', 'ECGFiveDays', 'FaceAll', 'Crop', 'CricketX', 'GunPoint', 'ChlorineConcentration', 'InsectWingbeatSound','ItalyPowerDemand', 'Phoneme', 'StarLightCurves', 'CBF', 'SyntheticControl','Strawberry', 'Chinatown','MelbournePedestrian', 'ArrowHead' , 'PhalangesOutlinesCorrect' ,'AllGestureWiimoteZ']
UCR_minisets = ['ECGFiveDays', 'FaceAll', 'CricketX', 'GunPoint', 'ChlorineConcentration', 'InsectWingbeatSound','ItalyPowerDemand', 'Phoneme', 'StarLightCurves', 'CBF', 'SyntheticControl','Strawberry', 'Chinatown','MelbournePedestrian', 'ArrowHead' , 'PhalangesOutlinesCorrect' ,'AllGestureWiimoteZ']
UCR_hardsets = ['GestureMidAirD3', 'Phoneme', 'ScreenType', 'InlineSkate', 'EthanolLevel','GestureMidAirD2', 'EOGVerticalSignal','Haptics','EOGHorizontalSignal','PLAID','DodgerLoopDay','MiddlePhalanxTW','RefrigerationDevices','GestureMidAirD1','InsectWingbeatSound','PigAirwayPressure','MiddlePhalanxOutlineAgeGroup','Herring']
UCR_easysets = ['CBF','Coffee','ECGFiveDays','Plane','Trace','TwoPatterns','GunPointMaleVersusFemale','GunPointOldVersusYoung','InsectEPGRegularTrain','InsectEPGSmallTrain','UMD','SyntheticControl','Wafer','GunPointAgeSpan','ShapeletSim','FreezerRegularTrain','DiatomSizeReduction','TwoLeadECG','GunPoint']
UCR_less_than_10 = ['MelbournePedestrian', 'Chinatown', 'FacesUCR', 'GunPointAgeSpan', 'GunPointOldVersusYoung', 'GunPointMaleVersusFemale', 'FaceAll', 'Fungi', 'SmoothSubspace', 'BME', 'MiddlePhalanxOutlineAgeGroup', 'PowerCons','ProximalPhalanxTW', 'ChlorineConcentration', 'MoteStrain', 'CBF', 'GestureMidAirD2', 'DistalPhalanxOutlineCorrect', 'SonyAIBORobotSurface2', 'DistalPhalanxOutlineAgeGroup', 'ArrowHead', 'ECG200', 'WordSynonyms', 'FreezerSmallTrain', 'DistalPhalanxTW', 'ProximalPhalanxOutlineAgeGroup', 'PickupGestureWiimoteZ', 'MedicalImages', 'GestureMidAirD1', 'GestureMidAirD3']
UCR_mediocre = list(pd.read_csv('DataSummary_with_baselines.csv').iloc[13:64]['Name'])
few_shot_df = pd.read_csv('ts2vec_with_label_info.csv')
UCR_few_shot = few_shot_df[few_shot_df['Use_for_paper_if_few_shot'] == 'Y']['dataset'].tolist()
UCR_ratio_ablation = ['Crop', 'DistalPhalanxOutlineCorrect', 'FordA', 'FordB', 'HandOutlines', 'MiddlePhalanxOutlineCorrect', 'PhalangesOutlinesCorrect', 'Strawberry', 'TwoPatterns', 'Wafer']



def save_checkpoint_callback(
    save_every=1,
    unit='epoch'
):
    assert unit in ('epoch', 'iter')
    def callback(model, loss):
        n = model.n_epochs if unit == 'epoch' else model.n_iters
        if n % save_every == 0:
            model.save(f'{run_dir}/model_{n}.pkl')
    return callback

def save_args(args, run_dir_root):
    with open(f'{run_dir_root}/args.pkl', 'wb') as file:
        pkl.dump(args, file)
        print(f'Args saved at : {run_dir_root}/args.pkl')


def load_args(run_dir_root):
    with open(f'{run_dir_root}/args.pkl', 'rb') as file:
        return pkl.load(file)


def select_labels_for_requirements_with_biasensss_not_random(label_ratio, train_labels, biasness=0.8, dataset = None, seed_number = None):
    N = len(train_labels)
    num_to_select = int(label_ratio * N)
    unique_labels, counts = np.unique(train_labels, return_counts=True)
    # 최소 선택 요구사항 설정
    if len(unique_labels) * 2 <= num_to_select:
        min_required = 2
    else:
        min_required = 1

    selected_indices = []
    for label in unique_labels:
        label_indices = np.where(train_labels == label)[0]
        random.shuffle(label_indices)
        selected_indices.extend(label_indices[:min_required])

    remaining_num = num_to_select - len(selected_indices)
    all_indices = np.arange(N)
    unselected_indices = np.setdiff1d(all_indices, selected_indices)
    unselected_labels = train_labels[unselected_indices]

    
    if len(unique_labels) == 2:
        biased_label = np.min(unique_labels)
    else:
    # biasness에 해당될 label 1개 선발
        biased_label_index = np.random.choice(np.where(unselected_labels == np.random.choice(unselected_labels))[0])
        biased_label = unselected_labels[biased_label_index]
    
    # biasness 계수만큼 해당 label 데이터 선발
    biased_label_indices = np.where(unselected_labels == biased_label)[0]
    not_biased_label_indices = np.where(unselected_labels != biased_label)[0]
    num_biased = int(remaining_num * biasness)
    if num_biased > len(biased_label_indices):
        num_biased = len(biased_label_indices)
    selected_biased_indices = np.random.choice(biased_label_indices, size=num_biased, replace=False)
    selected_indices.extend(unselected_indices[selected_biased_indices])

    
    # 1 - biasness 계수만큼 나머지 label 데이터 선발
    num_remaining = remaining_num - num_biased
    if num_remaining > len(not_biased_label_indices):
        num_remaining = len(not_biased_label_indices)
    selected_remaining_indices = np.random.choice(not_biased_label_indices, size=num_remaining, replace=False)
    selected_indices.extend(unselected_indices[selected_remaining_indices])

    
    random.shuffle(selected_indices)

    if dataset is not None:
        file_name = f"bias_indexes/{dataset}_seed{seed_number}_ratio{label_ratio:.2f}_biasness{biasness:.1f}.pkl"
        with open(file_name, "wb") as f:
            pickle.dump(selected_indices, f)

    selected_labels = train_labels[selected_indices]
    unique_labels, counts = np.unique(selected_labels, return_counts=True)
    print(f"Selected labels count:{len(selected_indices)}")
    for label, count in zip(unique_labels, counts):
        print(f"Label {label}: {count}")

    return selected_indices



def select_labels_for_requirements_with_biasensss(label_ratio, train_labels, biasness=0.8, dataset = None, seed_number = None):
    N = len(train_labels)
    num_to_select = int(label_ratio * N)
    unique_labels, counts = np.unique(train_labels, return_counts=True)
    # 최소 선택 요구사항 설정
    if len(unique_labels) * 2 <= num_to_select:
        min_required = 2
    else:
        min_required = 1

    selected_indices = []
    for label in unique_labels:
        label_indices = np.where(train_labels == label)[0]
        random.shuffle(label_indices)
        selected_indices.extend(label_indices[:min_required])

    remaining_num = num_to_select - len(selected_indices)
    all_indices = np.arange(N)
    unselected_indices = np.setdiff1d(all_indices, selected_indices)
    
    # biasness에 해당될 label 1개 선발
    unselected_labels = train_labels[unselected_indices]
    biased_label_index = np.random.choice(np.where(unselected_labels == np.random.choice(unselected_labels))[0])
    biased_label = unselected_labels[biased_label_index]
    
    # biasness 계수만큼 해당 label 데이터 선발
    biased_label_indices = np.where(unselected_labels == biased_label)[0]
    not_biased_label_indices = np.where(unselected_labels != biased_label)[0]
    num_biased = int(remaining_num * biasness)
    if num_biased > len(biased_label_indices):
        num_biased = len(biased_label_indices)
    selected_biased_indices = np.random.choice(biased_label_indices, size=num_biased, replace=False)
    selected_indices.extend(unselected_indices[selected_biased_indices])

    
    # 1 - biasness 계수만큼 나머지 label 데이터 선발
    num_remaining = remaining_num - num_biased
    if num_remaining > len(not_biased_label_indices):
        num_remaining = len(not_biased_label_indices)
    selected_remaining_indices = np.random.choice(not_biased_label_indices, size=num_remaining, replace=False)
    selected_indices.extend(unselected_indices[selected_remaining_indices])

    
    random.shuffle(selected_indices)

    if dataset is not None:
        file_name = f"bias_indexes/{dataset}_seed{seed_number}_ratio{label_ratio:.2f}_biasness{biasness:.1f}.pkl"
        with open(file_name, "wb") as f:
            pickle.dump(selected_indices, f)

    selected_labels = train_labels[selected_indices]
    unique_labels, counts = np.unique(selected_labels, return_counts=True)
    print(f"Selected labels count:{len(selected_indices)}")
    for label, count in zip(unique_labels, counts):
        print(f"Label {label}: {count}")

    return selected_indices
    
def select_labels_for_requirements(label_ratio, train_labels):
    N = len(train_labels)
    num_to_select = int(label_ratio * N)
    unique_labels, counts = np.unique(train_labels, return_counts = True)

    if len(unique_labels) * 2 <= num_to_select:
        min_required = 2
    else:
        min_required = 1

    selected_indices = []
    for label in unique_labels:
        label_indices = np.where(train_labels == label)[0]
        random.shuffle(label_indices)
        selected_indices.extend(label_indices[:min_required])

    remaining_num = num_to_select - len(selected_indices)
    all_indices = np.arange(N)
    unselected_indices = np.setdiff1d(all_indices, selected_indices)
    random.shuffle(unselected_indices)
    selected_indices.extend(unselected_indices[:remaining_num])

    random.shuffle(selected_indices)
    return selected_indices




if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='/data1/finance/UCR')
    parser.add_argument('--aug_path', type=str, default='/data1/finance/UCR_tvsplit_invcdf')
    parser.add_argument('--loader', type=str, required=True, help='The data loader used to load the experimental data. This can be set to UCR, UEA, forecast_csv, forecast_csv_univar, anomaly, or anomaly_coldstart')
    parser.add_argument('--rundir_root', type=str, default='training/')
    parser.add_argument('--gpu', type=int, default=0, help='The gpu no. used for training and inference (defaults to 0)')
    parser.add_argument('--batch-size', type=int, default=8, help='The batch size (defaults to 8)')
    parser.add_argument('--lr', type=float, default=0.001, help='The learning rate (defaults to 0.001)')
    parser.add_argument('--repr-dims', type=int, default=320, help='The representation dimension (defaults to 320)')
    parser.add_argument('--max-train-length', type=int, default=3000, help='For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length> (defaults to 3000)')
    parser.add_argument('--iters', type=int, default=None, help='The number of iterations')
    parser.add_argument('--epochs', type=int, default=None, help='The number of epochs')
    parser.add_argument('--save-every', type=int, default=None, help='Save the checkpoint every <save_every> iterations/epochs')
    parser.add_argument('--seed', type=int, default=None, help='The random seed')
    parser.add_argument('--train_seed', type = int, default = None , help='train_exclusive seed')
    parser.add_argument('--max-threads', type=int, default=None, help='The maximum allowed number of threads used by this process')
    parser.add_argument('--eval', action="store_true", help='Whether to perform evaluation after training')
    parser.add_argument('--irregular', type=float, default=0, help='The ratio of missing observations (defaults to 0)')
    parser.add_argument('--label_ratio', type=float, default=1.0, help="rabel ratio for semi-supervised learning")
    parser.add_argument('--miniset', type=str, default='miniset', help='training with minisets')
    parser.add_argument('--aug_name', type = str, default = 'dgw_dtw')
    parser.add_argument('--model_name', type=str, default='ts2vec', help='Model name')
    parser.add_argument('--aug_as_pos', action = "store_true", help='Whether to perform evaluation after training')
    parser.add_argument('--exclude_original', default= True, help='discard original data for training')
    parser.add_argument('--part_to_sample', type=str, default='seasonal', help = 'seasonal, residual, remainder')
    parser.add_argument('--use_label_info', action="store_true", help='Whether to perform evaluation after training')
    parser.add_argument('--use_augment', action="store_true", help='Whether to perform evaluation after training')
    parser.add_argument('--module_list', nargs = '+', default=['spectral', 'temporal'])
    parser.add_argument('--loss', type=str, default='ts2vec', help='ts2vec loss or infonce')
    parser.add_argument('--nearest_selection_mode', type=str, default='cossim', help='cossim/dot')
    parser.add_argument('--train_longer', action="store_true", help='Whether to perform evaluation after training')
    parser.add_argument('--pool_support_embed', action="store_true", help='Whether to perform evaluation after training')
    parser.add_argument('--use_knn_loss', action="store_true", help='Whether to perform evaluation after training')
    parser.add_argument('--use_pseudo_labeling', action="store_true", help='Whether to perform evaluation after training')
    parser.add_argument('--knn_loss_negative_lambda', type=float, default=0, help='The ratio of missing observations (defaults to 0)')
    parser.add_argument('--alpha', type=float, default = 0.5)
    parser.add_argument('--max_supset_size', type=float, default = 0.8)
    parser.add_argument('--UCR_test_dataset', type=str, default='ArrowHead')
    parser.add_argument('--topk', type=int, default = 5)
    parser.add_argument('--limit_support_set', action="store_true", help='Whether to perform evaluation after training')
    parser.add_argument('--biasness', type=float, default = 0.8)

    
    parser.add_argument('--other_aug', type=str, default="noaug", help="other augmentation mode")
    

    args = parser.parse_args()
    
    print("Arguments:", str(args))
    
    UCR_test = [args.UCR_test_dataset]
    
    if args.train_seed is None:
        args.train_seed = args.seed
    
    device = init_dl_program(args.gpu, seed=args.train_seed, max_threads=args.max_threads)

    if args.miniset  == 'UCR_hard':
        minisets = UCR_hardsets
    elif args.miniset == 'UCR_easy':
        minisets = UCR_easysets
    elif args.miniset == 'UCR_mediocre':
        minisets = UCR_mediocre
    elif args.miniset == 'UCR_less_than_10':
        minisets = UCR_less_than_10
    elif args.miniset == 'UCR_few_shot':
        minisets = UCR_few_shot
    elif args.miniset == 'epilepsy':
        minisets = ['epilepsy']
    elif args.miniset == 'UCR_ratio_ablation':
        minisets = UCR_ratio_ablation
    elif args.miniset == 'UCR_few_shot_except_crop':
        UCR_few_shot.remove('Crop')
        minisets = UCR_few_shot
    elif args.miniset == 'UCR_few_shot_less_than_one':
        remove_lst = ['Crop', 'ElectricDevices', 'FordA', 'FordB', 'HandOutlines', 'PhalangesOutlinesCorrect','StarLightCurves', 'TwoPatterns', 'Wafer']
        # remove_lst = ['Crop', 'ElectricDevices', 'FordA', 'FordB', 'HandOutlines', 'PhalangesOutlinesCorrect', 'StarLightCurves','TwoPatterns', 'Wafer']

        minisets = list(filter(lambda value: value not in remove_lst, UCR_few_shot))
    else:
        minisets = UCR_test
        
    print(f'Miniset length : {len(minisets)}')
    
    os.makedirs(args.rundir_root, exist_ok=True)
    save_args(args, args.rundir_root)

    
    for dataset in minisets:
        if args.loader == 'UCR':
            task_type = 'classification'
            aug_name = args.aug_name
            original_root = Path(args.root_path)
            augment_root = Path(args.aug_path)
            print('Augmentation mode ...', aug_name)
            print('Loading data... ', dataset)
            train_data, train_labels, test_data, test_labels = datautils.load_UCR(dataset, original_root)
            p = np.random.permutation(train_data.shape[0])
            train_data = train_data[p]
            train_labels = train_labels[p]
            
            original_root = Path(args.root_path)
    
            aug_joined_root = os.path.join(augment_root, aug_name)
            ori_joined_root = os.path.join(original_root)

            stl_path = os.path.join(augment_root, 'stl_sampling')
            stl_data = aug.generate_or_get_stl(train_data, train_labels, dataset, path = stl_path)
            with open(os.path.join(augment_root, aug_name, dataset, 'inv_cdfs.pkl'), 'rb') as f:
                inv_cdf = pkl.load(f)
            with open(os.path.join(augment_root, aug_name, dataset, 'inv_cdfs_seasonal.pkl'), 'rb') as f:
                inv_cdf_seasonal = pkl.load(f)
            with open(os.path.join(augment_root, aug_name, dataset, 'inv_cdfs_remainder.pkl'), 'rb') as f:
                inv_cdf_remainder = pkl.load(f)
            with open(os.path.join(augment_root, aug_name, dataset, 'inv_cdfs_trend.pkl'), 'rb') as f:
                inv_cdf_trend = pkl.load(f)                


        elif args.loader == "epilepsy":
            task_type = 'classification'
            original_root = Path(args.root_path)
            aug_name = args.aug_name

            train_data, train_labels, test_data, test_labels, val_data, val_labels  = datautils.load_Epilepsy(dataset, original_root)
            p = np.random.permutation(train_data.shape[0])
            train_data = train_data[p]
            train_labels = train_labels[p]
            
            original_root = Path(args.root_path)
            augment_root = Path(args.aug_path)
            
            aug_joined_root = os.path.join(augment_root, aug_name)
            ori_joined_root = os.path.join(original_root)

            stl_path = os.path.join(augment_root, 'stl_sampling')
            stl_data = aug.generate_or_get_stl(train_data, train_labels, dataset, path = stl_path)
            with open(os.path.join(augment_root, aug_name, dataset, 'inv_cdfs.pkl'), 'rb') as f:
                inv_cdf = pkl.load(f)
            with open(os.path.join(augment_root, aug_name, dataset, 'inv_cdfs_seasonal.pkl'), 'rb') as f:
                inv_cdf_seasonal = pkl.load(f)
            with open(os.path.join(augment_root, aug_name, dataset, 'inv_cdfs_remainder.pkl'), 'rb') as f:
                inv_cdf_remainder = pkl.load(f)
            with open(os.path.join(augment_root, aug_name, dataset, 'inv_cdfs_trend.pkl'), 'rb') as f:
                inv_cdf_trend = pkl.load(f)                

            
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

            original_root = Path(args.root_path)
            augment_root = Path(args.aug_path)
            aug_name = args.aug_name
            print('Augmentation mode ...', aug_name)
            print('Loading data... ', dataset)
            all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay = datautils.load_anomaly(args.root_path, dataset)
            train_data = datautils.gen_ano_train_data(all_train_data)

            p = np.random.permutation(train_data.shape[0])
            train_data = train_data[p]
            
            original_root = Path(args.root_path)
    
            aug_joined_root = os.path.join(augment_root, aug_name)

            stl_path = os.path.join(augment_root, 'stl_sampling')
            stl_data = aug.generate_or_get_stl(train_data, all_train_labels, dataset, path = stl_path)
            with open(os.path.join(augment_root, aug_name, dataset, 'inv_cdfs.pkl'), 'rb') as f:
                inv_cdf = pkl.load(f)
            with open(os.path.join(augment_root, aug_name, dataset, 'inv_cdfs_seasonal.pkl'), 'rb') as f:
                inv_cdf_seasonal = pkl.load(f)
            with open(os.path.join(augment_root, aug_name, dataset, 'inv_cdfs_remainder.pkl'), 'rb') as f:
                inv_cdf_remainder = pkl.load(f)
            with open(os.path.join(augment_root, aug_name, dataset, 'inv_cdfs_trend.pkl'), 'rb') as f:
                inv_cdf_trend = pkl.load(f)                


        elif args.loader == 'anomaly_coldstart':
            task_type = 'anomaly_detection_coldstart'
            original_root = Path(args.root_path)
            augment_root = Path(args.aug_path)
            aug_name = args.aug_name
            print('Augmentation mode ...', aug_name)
            print('Loading data... ', dataset)
            all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay = datautils.load_anomaly(args.root_path, dataset)

            train_data, train_labels, test_data, test_labels = datautils.load_UCR('TwoPatterns', '/data1/finance/UCR_tvsplit')
            p = np.random.permutation(train_data.shape[0])
            train_data = train_data[p]
            train_labels = train_labels[p]
            
            original_root = Path(args.root_path)
    
            aug_joined_root = os.path.join(augment_root, aug_name)
            ori_joined_root = os.path.join(original_root)

            stl_path = os.path.join(augment_root, 'stl_sampling')
            stl_data = aug.generate_or_get_stl(train_data, train_labels, dataset, path = stl_path)
            with open(os.path.join(augment_root, aug_name, dataset, 'inv_cdfs.pkl'), 'rb') as f:
                inv_cdf = pkl.load(f)
            with open(os.path.join(augment_root, aug_name, dataset, 'inv_cdfs_seasonal.pkl'), 'rb') as f:
                inv_cdf_seasonal = pkl.load(f)
            with open(os.path.join(augment_root, aug_name, dataset, 'inv_cdfs_remainder.pkl'), 'rb') as f:
                inv_cdf_remainder = pkl.load(f)
            with open(os.path.join(augment_root, aug_name, dataset, 'inv_cdfs_trend.pkl'), 'rb') as f:
                inv_cdf_trend = pkl.load(f)                
            
        else:
            raise ValueError(f"Unknown loader {args.loader}.")
        
        run_name = 'UCR'
        #run_dir = args.rundir_root'training_randomwalk_pequal_allset/' + dataset + '__' + name_with_datetime(run_name)+'__'+str(args.train_seed)+ '__' + str(args.seed) + '__' +aug_name + '__' + str(args.iters)
        run_dir = args.rundir_root+ '/'+ dataset + '__' + name_with_datetime(run_name)+'__'+str(args.train_seed)+ '__' + str(args.seed) + '__' +aug_name + '__' + str(args.iters)
        print('run_dir : ', run_dir)
        os.makedirs(run_dir, exist_ok=True)

        if args.irregular > 0:
            if task_type == 'classification':
                train_data = data_dropout(train_data, args.irregular)
                test_data = data_dropout(test_data, args.irregular)
            else:
                raise ValueError(f"Task type {task_type} is not supported when irregular>0.")
        print('done')

        if not (args.loader == 'anomaly' or args.loader =='anomaly_coldstart'):
            if args.biasness == 0:
                label_mask_index = select_labels_for_requirements(args.label_ratio, train_labels)
            else:
            # label_mask_index = select_labels_for_requirements(args.label_ratio, train_labels)
                label_mask_index = select_labels_for_requirements_with_biasensss_not_random(args.label_ratio, train_labels, biasness= args.biasness, dataset = dataset, seed_number = args.seed)

            
        else:
            label_mask_index = []
        
        config = dict(
            batch_size=args.batch_size,
            lr=args.lr,
            output_dims=args.repr_dims,
            max_train_length=args.max_train_length,
            pool_support_embed = args.pool_support_embed,
            train_data_L = train_data.shape[1],
            nearest_selection_mode = args.nearest_selection_mode,
            use_knn_loss = args.use_knn_loss,
            knn_loss_negative_lambda = args.knn_loss_negative_lambda,
            label_ratio = args.label_ratio,
            use_pseudo_labeling = args.use_pseudo_labeling,
            alpha = args.alpha,
            max_supset_size = args.max_supset_size,
            label_mask_index = label_mask_index,
            top_k = args.topk,
            limit_support_set = args.limit_support_set,
        )
        
        if args.save_every is not None:
            unit = 'epoch' if args.epochs is not None else 'iter'
            config[f'after_{unit}_callback'] = save_checkpoint_callback(args.save_every, unit)

        t = time.time()
        
        if args.model_name == 'ts2vec':
            model = TS2Vec(
                input_dims=train_data.shape[-1],
                device=device,
                **config
            )
            
            loss_log = model.fit_with_stl_augs(
                train_data,
                stl_data,
                train_labels,
                inv_cdf = inv_cdf,
                inv_cdf_seasonal = inv_cdf_seasonal,
                inv_cdf_remainder = inv_cdf_remainder,
                part_to_sample = args.part_to_sample,
                n_epochs=args.epochs,
                n_iters=args.iters,
                verbose=True,
                aug_as_pos = args.aug_as_pos
            )
            
            
        elif args.model_name == 'nnclr':
            model = NNCLR(
                input_dims = train_data.shape[-1],
                device = device,
                use_label_info= args.use_label_info,
                use_augment = args.use_augment,
                module_list = args.module_list,
                loss = args.loss,
                train_longer = args.train_longer,
                **config
            )
            
            loss_log = model.fit(
                train_data,
                stl_data,
                train_labels,
                inv_cdf = inv_cdf,
                inv_cdf_seasonal = inv_cdf_seasonal,
                inv_cdf_remainder = inv_cdf_remainder,
                inv_cdf_trend = inv_cdf_trend,                
                part_to_sample = args.part_to_sample,
                n_epochs=args.epochs,
                n_iters=args.iters,
                verbose=True,
                aug_as_pos = args.aug_as_pos
            )
            
        elif args.model_name == 'nnclr_anomaly':
            model = NNCLR_anomaly_self_supervised_version(
                input_dims = train_data.shape[-1],
                device = device,
                use_augment = args.use_augment,
                module_list = args.module_list,
                loss = args.loss,
                train_longer = args.train_longer,
                **config
            )
            
            loss_log = model.fit(
                train_data,
                stl_data,
                all_train_labels,
                inv_cdf = inv_cdf,
                inv_cdf_seasonal = inv_cdf_seasonal,
                inv_cdf_remainder = inv_cdf_remainder,
                inv_cdf_trend = inv_cdf_trend,                
                part_to_sample = args.part_to_sample,
                n_epochs=args.epochs,
                n_iters=args.iters,
                verbose=True,
                aug_as_pos = args.aug_as_pos
            )


        else:
            model = TS2Vec(
                input_dims=train_data.shape[-1],
                device=device,
                **config
            )
            
        # loss_log = model.fit(
        #     train_data,
        #     n_epochs=args.epochs,
        #     n_iters=args.iters,
        #     verbose=True
        # )

        model.save(f'{run_dir}/model.pkl')

        t = time.time() - t
        print(f"\nTraining time: {datetime.timedelta(seconds=t)}\n")

        if args.eval:
            if args.model_name == 'ts2vec':
                if task_type == 'classification':
                    out, eval_res = tasks.eval_classification(model, train_data, train_labels, test_data, test_labels, eval_protocol='svm')                    

                elif task_type == 'forecasting':
                    out, eval_res = tasks.eval_forecasting(model, data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols)
                elif task_type == 'anomaly_detection':
                    out, eval_res = tasks.eval_anomaly_detection(model, all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay)
                elif task_type == 'anomaly_detection_coldstart':
                    out, eval_res = tasks.eval_anomaly_detection_coldstart(model, all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay)
                else:
                    assert False                
            elif args.model_name == 'nnclr' or args.model_name == 'nnclr_other_augs':
                if task_type == 'classification':
                    out, eval_res = tasks.eval_classification(model, train_data, train_labels, test_data, test_labels, eval_protocol='svm', run_dir=run_dir, plot_tsne=True, label_mask_index = label_mask_index)                    

            elif args.model_name == 'nnclr_anomaly':
                if task_type == 'anomaly_detection':
                    out, eval_res, labels_log, test_data_log = tasks.eval_anomaly_detection(model, all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay)
                elif task_type == 'anomaly_detection_coldstart':
                    out, eval_res = tasks.eval_anomaly_detection_coldstart(model, all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay)

                pkl_save(f'{run_dir}/labels_log.pkl', labels_log)
                pkl_save(f'{run_dir}/test_data_log.pkl', test_data_log)

                
            pkl_save(f'{run_dir}/out.pkl', out)
            pkl_save(f'{run_dir}/eval_res.pkl', eval_res)
            print('Evaluation result:', eval_res)

        print("Finished.")
