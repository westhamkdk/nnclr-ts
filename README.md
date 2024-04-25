# Semi-supervised Contrastive Learning with Decomposition-based Data Augmentation for Time Series Classification (NNCLR-TS)

This repository contains the implementation of the NNCLR-TS model for time series classification using semi-supervised contrastive learning and decomposition-based data augmentation.

## Requirements

The recommended requirements for NNCLR-TS are as follows:
* Python 3.8
* torch==1.8.1
* scipy==1.6.1
* numpy==1.19.2
* pandas==1.0.1
* scikit_learn==0.24.2
* statsmodels==0.12.2
* Bottleneck==1.3.2

The dependencies can be installed by running:
```bash
pip install -r requirements.txt
```

## Datasets

Please download the following datasets:

* UCR Archive: Download the UCR classification datasets from [this link](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/).
* Epilepsy: Download the Epilepsy Seizure prediction dataset from [this link](https://www.kaggle.com/datasets/harunshimanto/epileptic-seizure-recognition).

## Data Preprocessing

Before training the model, the data needs to be preprocessed. Run the following command for data preprocessing:

```
python aug_data_generator_only_probs.py <root_path> <aug_path> --loader <loader> --use_chunk <use_chunk> --max-threads <max_threads>
```

The detailed descriptions of the arguments are as follows:
| Parameter name | Description of parameter |
| --- | --- |
| root_path | The root path of the dataset |
| aug_path | The path to save the augmented data |
| loader | The data loader used to load the experimental data. This can be set to `UCR`, `epilepsy`, or `anomaly` |
| use_chunk | Whether to use chunk-based processing (defaults to 0) |
| max-threads | The maximum number of threads to use (defaults to 8) |


## Usage

python train_online_augs.py --gpu <gpu> --loader <loader> --model_name <model_name> --root_path <root_path> --miniset <miniset> --aug_name <aug_name> --part_to_sample <part_to_sample> --rundir_root <rundir_root> --batch-size <batch_size> --repr-dims <repr_dims> --max-threads <max_threads> --seed <seed> --eval --label_ratio <label_ratio> --use_augment --use_label_info --loss <loss> --pool_support_embed --module_list <module_list> --nearest_selection_mode <nearest_selection_mode> --use_knn_loss --use_pseudo_labeling --knn_loss_negative_lambda <knn_loss_negative_lambda> --UCR_test_dataset <UCR_test_dataset> --alpha <alpha> --topk <topk> --max_supset_size <max_supset_size>
```

The detailed descriptions of the arguments are as follows:
| Parameter name | Description of parameter |
| --- | --- |
| gpu | The GPU number used for training and inference (defaults to 0) |
| loader | The data loader used to load the experimental data. This can be set to `UCR` or `epilepsy` |
| model_name | The name of the model to be used |
| root_path | The root path of the dataset |
| miniset | The name of the miniset to be used |
| aug_name | The name of the augmentation method to be used |
| part_to_sample | The part of the time series to be sampled |
| rundir_root | The root directory to save the training results |
| batch_size | The batch size (defaults to 8) |
| repr_dims | The representation dimensions (defaults to 40) |
| max-threads | The maximum number of threads to use (defaults to 8) |
| seed | The random seed (defaults to 0) |
| eval | Whether to perform evaluation after training |
| label_ratio | The ratio of labeled data to be used |
| use_augment | Whether to use data augmentation |
| use_label_info | Whether to use label information |
| loss | The loss function to be used (defaults to "infonce") |
| pool_support_embed | Whether to pool support embeddings |
| module_list | The list of modules to be used (defaults to "temporal" and "spectral") |
| nearest_selection_mode | The mode for selecting nearest neighbors (defaults to "dot") |
| use_knn_loss | Whether to use KNN loss |
| use_pseudo_labeling | Whether to use pseudo-labeling |
| knn_loss_negative_lambda | The negative lambda value for KNN loss (defaults to 2.0) |
| UCR_test_dataset | The UCR test dataset to be used |
| alpha | The alpha value (defaults to 0.2) |
| topk | The number of top-k neighbors to consider (defaults to 5) |
| max_supset_size | The maximum size of the support set (defaults to 1.0) |

After training and evaluation, the trained model, output, and evaluation metrics can be found in the directory specified by `rundir_root`.
```

## Execution Examples

Refer to the `launch.json` file for examples of how to run the NNCLR-TS model with different configurations. 

## Acknowledgments

This project is based on the code from the [TS2Vec repository](https://github.com/yuezhihan/ts2vec) by Zhihan Yue. We would like to express our gratitude to the authors of the TS2Vec paper for their excellent research and open-source code, which served as a valuable reference for this project. The original code has been modified and adapted to implement the NNCLR-TS model for semi-supervised contrastive learning with decomposition-based data augmentation for time series classification. We are committed to contributing to the advancement of time series analysis.


