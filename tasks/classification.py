import numpy as np
from . import _eval_protocols as eval_protocols
from sklearn.preprocessing import label_binarize
from sklearn.metrics import average_precision_score, f1_score,accuracy_score
from sklearn.linear_model import RidgeClassifierCV
import time
import datetime
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch as th
import torch


def eval_classification(model, train_data, train_labels, test_data, test_labels, eval_protocol='linear', run_dir=None, plot_tsne=True, label_mask_index = None):
    assert train_labels.ndim == 1 or train_labels.ndim == 2

    t = time.time()

    if label_mask_index is not None or train_data.shape[0] != len(label_mask_index):
        

        if train_data.shape[0] != len(label_mask_index):
            original_train_data = train_data.copy()
            unlabeled_train_data = np.delete(train_data, label_mask_index, axis=0)
            unlabeled_train_labels = np.delete(train_labels, label_mask_index)
            train_data = train_data[label_mask_index, ...]
            train_labels = train_labels[label_mask_index]

            train_repr = model.encode(train_data, encoding_window='full_series' if train_labels.ndim == 1 else None)
            test_repr = model.encode(test_data, encoding_window='full_series' if train_labels.ndim == 1 else None)


            p = np.random.permutation(unlabeled_train_data.shape[0])
            
            unlabeled_train_data = unlabeled_train_data[p]
            unlabeled_train_labels = unlabeled_train_labels[p]
            unlabeled_repr = model.encode(unlabeled_train_data, encoding_window='full_series' if train_labels.ndim == 1 else None)

            dist_matrix = np.sum((train_repr[:, np.newaxis] - unlabeled_repr) ** 2, axis=-1)

            top_k = min(4, train_data.shape[0])
            top_6_indices = np.argsort(dist_matrix, axis=0)[:top_k]
            top_6_labels = train_labels[top_6_indices]
            most_frequent_labels = np.zeros((len(unlabeled_train_labels),))
            for i in range(top_6_labels.shape[1]):
                unique, counts = np.unique(top_6_labels[:, i], return_counts=True)
                max_count = np.max(counts)
                max_indices = np.where(counts == max_count)[0]
                most_frequent_labels[i] = np.random.choice(unique[max_indices])

            original_train_data = np.concatenate((train_data, unlabeled_train_data), axis=0)
            train_repr = np.concatenate((train_repr, unlabeled_repr), axis=0)
            train_labels = np.concatenate((train_labels, most_frequent_labels), axis=0)
            
        else:
            train_repr = model.encode(train_data, encoding_window='full_series' if train_labels.ndim == 1 else None)
            test_repr = model.encode(test_data, encoding_window='full_series' if train_labels.ndim == 1 else None)
    else:
        train_repr = model.encode(train_data, encoding_window='full_series' if train_labels.ndim == 1 else None)
        test_repr = model.encode(test_data, encoding_window='full_series' if train_labels.ndim == 1 else None)


    if eval_protocol == 'linear':
        fit_clf = eval_protocols.fit_lr
    elif eval_protocol == 'svm':
        fit_clf = eval_protocols.fit_svm
    elif eval_protocol == 'knn':
        fit_clf = eval_protocols.fit_knn
    elif eval_protocol == "ridge":
        fit_clf = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10))
    else:
        assert False, 'unknown evaluation protocol'

    def merge_dim01(array):
        return array.reshape(array.shape[0]*array.shape[1], *array.shape[2:])

    if train_labels.ndim == 2:
        train_repr = merge_dim01(train_repr)
        train_labels = merge_dim01(train_labels)
        test_repr = merge_dim01(test_repr)
        test_labels = merge_dim01(test_labels)

    if not eval_protocol == "ridge":
        clf = fit_clf(train_repr, train_labels)        
    else:
        clf = fit_clf.fit(train_repr, train_labels)
        

    acc = clf.score(test_repr, test_labels)
    
    if plot_tsne:
        import pandas as pd
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        
        X_embedded = TSNE(n_components= 2, learning_rate='auto', init='random').fit_transform(np.array(train_repr))
        tsne_df = pd.DataFrame({'x': X_embedded[:, 0], 'y':X_embedded[:, 1], 'classes':train_labels})
        plt.scatter(tsne_df['x'], tsne_df['y'], c = tsne_df['classes'])
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        plt.savefig(f'{run_dir}/train_repr.png')
        plt.clf()        
        
        
        X_embedded = TSNE(n_components= 2, learning_rate='auto', init='random').fit_transform(np.array(test_repr))
        tsne_df = pd.DataFrame({'x': X_embedded[:, 0], 'y':X_embedded[:, 1], 'classes':test_labels})
        plt.scatter(tsne_df['x'], tsne_df['y'], c = tsne_df['classes'])
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        plt.savefig(f'{run_dir}/test_repr.png')
        plt.clf()        
        
        X_original = TSNE(n_components= 2, learning_rate='auto', init='random').fit_transform(np.array(test_data.squeeze()))
        tsne_df_original = pd.DataFrame({'x': X_original[:, 0], 'y':X_original[:, 1], 'classes':test_labels})
        plt.scatter(tsne_df_original['x'], tsne_df_original['y'], c = tsne_df_original['classes'])
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        plt.savefig(f'{run_dir}/test_data.png')
        plt.clf()        
        
        X_original = TSNE(n_components= 2, learning_rate='auto', init='random').fit_transform(np.array(original_train_data.squeeze()))
        tsne_df_original = pd.DataFrame({'x': X_original[:, 0], 'y':X_original[:, 1], 'classes':train_labels})
        plt.scatter(tsne_df_original['x'], tsne_df_original['y'], c = tsne_df_original['classes'])
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        plt.savefig(f'{run_dir}/train_data.png')
        plt.clf()        

        
    
    if eval_protocol == 'linear' or eval_protocol == 'knn':
        y_score = clf.predict_proba(test_repr)
    else:
        y_score = clf.decision_function(test_repr)
    test_labels_onehot = label_binarize(test_labels, classes=np.arange(train_labels.max()+1))
    
    if eval_protocol == 'knn':
        auprc = average_precision_score(test_labels_onehot, y_score[:, 1])
    else:
        auprc = average_precision_score(test_labels_onehot, y_score)
        mf1 = f1_score(test_labels, clf.predict(test_repr), average='macro')

    
    t = time.time() - t
    print(f"\nClassification time: {datetime.timedelta(seconds=t)}\n")

    # y_score가 0보다 크거나 같으면 label 1, 미만이면 label 0으로 할당됨
    return y_score, { 'acc': acc, 'auprc': auprc , 'mf1':mf1}


def test_model(model, test_data, test_labels):    
    model.net.eval()
    model.projection.eval()
    model.projection.to('cpu')
    # model.net.to('cpu')
    
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        logits = model.projection(model.encode_test(torch.Tensor(test_data), encoding_window='full_series' if test_labels.ndim == 1 else None))
        # logits = model.projection(model.net(torch.Tensor(test_data)))
        test_labels = torch.Tensor(test_labels)
        y_score, predicted = torch.max(logits, 1)
        
        
        total += test_labels.size(0)
        print(np.unique(predicted.cpu().numpy(), return_counts = True))
        correct += (predicted == test_labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(test_labels.cpu().numpy())
            
        accuracy = correct / total
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
    
    
    return y_score, { 'acc': accuracy , 'mf1':macro_f1}
 
def test_mixup(model, test_data, test_labels):    
    model.encoder.eval()
    model.encoder.to('cpu')
    model.projection.eval()
    model.projection.to('cpu')
    # model.net.to('cpu')
    
    print("test labels : ", np.unique(test_labels, return_counts = True))
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        logits = model.projection(model.encode(torch.Tensor(test_data).permute([0,2,1])))
        test_labels = torch.Tensor(test_labels)
        y_score, predicted = torch.max(logits, 1)
        
        total += test_labels.size(0)
        print(np.unique(predicted.cpu().numpy(), return_counts = True))
        correct += (predicted == test_labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(test_labels.cpu().numpy())
            
        accuracy = correct / total
        print("preds : ", np.unique(all_preds, return_counts = True))

        macro_f1 = f1_score(all_labels, all_preds, average='macro')
    
    
    return y_score, { 'acc': accuracy , 'mf1':macro_f1}
