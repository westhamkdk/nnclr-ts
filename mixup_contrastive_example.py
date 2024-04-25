import torch
import torch as th
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from tasks import _eval_protocols as eval_protocols
from models import LinearProjection
import tempfile
import os
from utils import take_per_row, split_with_nan, centerize_vary_length_series, torch_pad_nan




from sktime.datasets import load_gunpoint
from IPython.display import clear_output
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.neighbors import KNeighborsClassifier

from utils import split_with_nan, centerize_vary_length_series

def to_np(x):
    return x.cpu().detach().numpy()

# print(f"x_tr: {x_tr.shape}, y_tr : {y_tr.shape}, x_te : {x_te.shape}, y_te : {y_te.shape}")
# x_tr: (50, 1, 150), y_tr : (50, 1), x_te : (150, 1, 150), y_te : (150, 1)



class MixUp(nn.Module):
    def __init__(self, n_in, device='cuda', lr=0.001,max_train_length=None, num_classes = 2):
        super(MixUp, self).__init__()
        self.device = device
        self.lr = lr
        self.batch_size = 8
        self.max_train_length = max_train_length
        self.n_epochs = 0
        self.n_iters = 0    
        
        self.projection = LinearProjection(128, num_classes).to(self.device)
        
        self.criterion = MixUpLoss(self.device, self.batch_size)
        
        self.encoder = nn.Sequential(
            nn.Conv1d(n_in, 128, kernel_size=7, padding=6, dilation=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=5, padding=8, dilation=4),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=3, padding=8, dilation=8),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )

        self.proj_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )        
        self.net = torch.optim.swa_utils.AveragedModel(self)
        self.net.update_parameters(self)

        self.to(self.device)
        

    def forward(self, x):
        h = self.encoder(x)
        out = self.proj_head(h)
        return out, h
    


    def _eval_with_pooling(self, x, mask=None, slicing=None, encoding_window=None):
        out = self.encoder(x)
        if encoding_window == 'full_series':
            if slicing is not None:
                out = out[:, slicing]
            
        elif isinstance(encoding_window, int):
            if encoding_window % 2 == 0:
                out = out[:, :-1]
            if slicing is not None:
                out = out[:, slicing]            
        else:
            if slicing is not None:
                out = out[:, slicing]
            
        return out.cpu()
    
    def encode(self, data, mask=None, encoding_window=None, casual=False, sliding_length=None, sliding_padding=0, batch_size=None):
        # n_samples, ts_l, _ = data.shape
        # self.eval()

        # dataset = TensorDataset(torch.from_numpy(data).to(torch.float))
        # loader = DataLoader(dataset, batch_size=batch_size)

        # with torch.no_grad():
        #     output = []
        #     for batch in loader:
        #         x = batch[0]
        #         if sliding_length is not None:
        #             reprs = []
        #             if n_samples < batch_size:
        #                 calc_buffer = []
        #                 calc_buffer_l = 0
        #             for i in range(0, ts_l, sliding_length):
        #                 l = i - sliding_padding
        #                 r = i + sliding_length + (sliding_padding if not casual else 0)
        #                 x_sliding = torch_pad_nan(
        #                     x[:, max(l, 0) : min(r, ts_l)],
        #                     left=-l if l<0 else 0,
        #                     right=r-ts_l if r>ts_l else 0,
        #                     dim=1
        #                 )
        #                 if n_samples < batch_size:
        #                     if calc_buffer_l + n_samples > batch_size:
        #                         out = self._eval_with_pooling(
        #                             torch.cat(calc_buffer, dim=0),
        #                             mask,
        #                             slicing=slice(sliding_padding, sliding_padding+sliding_length),
        #                             encoding_window=encoding_window
        #                         )
        #                         reprs += torch.split(out, n_samples)
        #                         calc_buffer = []
        #                         calc_buffer_l = 0
        #                     calc_buffer.append(x_sliding)
        #                     calc_buffer_l += n_samples
        #                 else:
        #                     out = self._eval_with_pooling(
        #                         x_sliding,
        #                         mask,
        #                         slicing=slice(sliding_padding, sliding_padding+sliding_length),
        #                         encoding_window=encoding_window
        #                     )
        #                     reprs.append(out)

        #             if n_samples < batch_size:
        #                 if calc_buffer_l > 0:
        #                     out = self._eval_with_pooling(
        #                         torch.cat(calc_buffer, dim=0),
        #                         mask,
        #                         slicing=slice(sliding_padding, sliding_padding+sliding_length),
        #                         encoding_window=encoding_window
        #                     )
        #                     reprs += torch.split(out, n_samples)
        #                     calc_buffer = []
        #                     calc_buffer_l = 0
                    
        #             out = torch.cat(reprs, dim=1)
        #             if encoding_window == 'full_series':
        #                 out = F.max_pool1d(
        #                     out.transpose(1, 2).contiguous(),
        #                     kernel_size = out.size(1),
        #                 ).squeeze(1)
        #         else:
        #             out = self._eval_with_pooling(x, mask, encoding_window=encoding_window)
        #             if encoding_window == 'full_series':
        #                 out = out.squeeze(1)
                        
        #         output.append(out)
                
        #     output = torch.cat(output, dim=0)
            



        return self.encoder(data)
    
    
    def fit(self, train_data, n_epochs=None, n_iters=None, verbose=False, alpha = 0.2):
        self.train()

        assert train_data.ndim == 3
        
        if n_iters is None and n_epochs is None:
            n_iters = 200 if train_data.size <= 100000 else 600  # default param for n_iters
            print(f"n_iters : {n_iters}")

        if self.max_train_length is not None:
            sections = train_data.shape[1] // self.max_train_length
            if sections >= 2:
                train_data = np.concatenate(split_with_nan(train_data, sections, axis=1), axis=0)

        temporal_missing = np.isnan(train_data).all(axis=-1).any(axis=0)
        if temporal_missing[0] or temporal_missing[-1]:
            train_data = centerize_vary_length_series(train_data)
                
        train_data = train_data[~np.isnan(train_data).all(axis=2).all(axis=1)]
        train_data = np.transpose(train_data, (0,2,1))
        
        train_dataset = TensorDataset(torch.from_numpy(train_data).to(torch.float))
        train_loader = DataLoader(train_dataset, batch_size=min(self.batch_size, len(train_dataset)), shuffle=True, drop_last=True)
        
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        
        loss_log = []
        
        while True:
            if n_epochs is not None and self.n_epochs >= n_epochs:
                break
            
            cum_loss = 0
            n_epoch_iters = 0
            
            interrupted = False
            # for batch in train_loader:
            for batch in train_loader:
                if n_iters is not None and self.n_iters >= n_iters:
                    interrupted = True
                    break
                
                optimizer.zero_grad()

                x = batch[0]
                if self.max_train_length is not None and x.size(2) > self.max_train_length:
                    window_offset = np.random.randint(x.size(2) - self.max_train_length + 1)
                    x = x[:, window_offset : window_offset + self.max_train_length]
                x = x.to(self.device)
                
                x_1 = x
                x_2 = x_1[torch.randperm(len(x))]
                lam = np.random.beta(alpha,alpha)
                x_aug = lam * x_1 + (1-lam) * x_2
                
                x_1 = x_1.to(self.device)
                x_2 = x_2.to(self.device)
                x_aug = x_aug.to(self.device)
                
                z_1, _ = self.forward(x_1)
                z_2, _ = self.forward(x_2)
                z_aug, _ = self.forward(x_aug)
                
                loss = self.criterion(z_aug, z_1, z_2, lam)
                loss.backward()
                
                optimizer.step()
                self.net.update_parameters(self)
                    
                cum_loss += loss.item()
                n_epoch_iters += 1
                
                self.n_iters += 1
            
            if interrupted:
                break
            
            cum_loss /= n_epoch_iters
            loss_log.append(cum_loss)
            if verbose:
                print(f"Epoch #{self.n_epochs}: loss={cum_loss}")
            self.n_epochs += 1
                            
            
        return loss_log
    

    def fit_linear(self, train_data, train_labels, val_data, val_labels):
        print(f'fit linear train data size : {train_data.shape[0]}')
        print(np.unique(train_labels, return_counts = True))
        train_data = torch.tensor(train_data, dtype=torch.float32)
        train_labels = torch.tensor(train_labels, dtype=torch.long)
        train_dataset = TensorDataset(train_data, train_labels)
        train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        
        val_data = torch.tensor(val_data, dtype=torch.float32)
        val_labels = torch.tensor(val_labels, dtype=torch.long)
        val_dataset = TensorDataset(val_data, val_labels)
        val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)
        
        optimizer = torch.optim.Adam(self.projection.parameters(), lr=3e-3)
        criterion = torch.nn.CrossEntropyLoss()

        self.freeze_encoders()
        self.projection.train()

        best_val_accuracy = 0.0  # Keep track of the best validation accuracy
        _, best_model_path = tempfile.mkstemp()
        
        for epoch in range(40):
            running_loss = 0.0
            for batch_data, batch_labels in train_dataloader:
                batch_data, batch_labels = batch_data.to(self.device), batch_labels.to(self.device)
                
                batch_data = batch_data.permute([0,2,1])

                repr = self.encode(batch_data)
                logits = self.projection(repr)
                loss = criterion(logits, batch_labels)
                optimizer.zero_grad()

                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
            epoch_loss = running_loss / len(train_dataloader)
            print(f"Linear Training Epoch #{epoch}: loss={epoch_loss}")

            # Evaluate on the validation set
            self.projection.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_data, batch_labels in val_dataloader:
                    batch_data, batch_labels = batch_data.to(self.device), batch_labels.to(self.device)
                    batch_data = batch_data.permute([0,2,1])

                    repr = self.encode(batch_data)
                    logits = self.projection(repr)
                    
                    _, predicted = torch.max(logits.data, 1)
                    total += batch_labels.size(0)
                    correct += (predicted == batch_labels).sum().item()
            
            val_accuracy = correct / total
            print(f"Validation accuracy: {val_accuracy}")
            
            # If this epoch gives us the best validation accuracy so far, save the model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save(self.projection.state_dict(), best_model_path)

            self.projection.train()

        # Load the best model
        self.projection.load_state_dict(torch.load(best_model_path))
        os.remove(best_model_path)

            
    def freeze_encoders(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
            

    def save(self, fn):
        ''' Save the model to a file.
        
        Args:
            fn (str): filename.
        '''
        torch.save(self.state_dict(), fn)
    
    def load(self, fn):
        ''' Load the model from a file.
        
        Args:
            fn (str): filename.
        '''
        state_dict = torch.load(fn, map_location=self.device)
        self.load_state_dict(state_dict)

    

class MixUpLoss(th.nn.Module):

    def __init__(self, device, batch_size):
        super(MixUpLoss, self).__init__()
        
        self.tau = 0.5
        self.device = device
        self.batch_size = batch_size
        self.logsoftmax = nn.LogSoftmax(dim=1)
        
    def forward(self, z_aug, z_1, z_2, lam):

        z_1 = nn.functional.normalize(z_1) # [8,128]
        z_2 = nn.functional.normalize(z_2)
        z_aug = nn.functional.normalize(z_aug)

        labels_lam_0 = lam*th.eye(self.batch_size, device=self.device)
        labels_lam_1 = (1-lam)*th.eye(self.batch_size, device=self.device) # [8, 16]

        labels = th.cat((labels_lam_0, labels_lam_1), 1)

        logits = th.cat((th.mm(z_aug, z_1.T),
                         th.mm(z_aug, z_2.T)), 1) # 8, 16

        loss = self.cross_entropy(logits / self.tau, labels)

        return loss

    def cross_entropy(self, logits, soft_targets):
        return th.mean(th.sum(- soft_targets * self.logsoftmax(logits), 1))


def test_model(model, training_set, test_set):
    class MyDataset(Dataset):
        def __init__(self, x, y):
            device = 'cuda'
            self.x = th.tensor(x, dtype=th.float, device=device)
            self.y = th.tensor(y, dtype=th.long, device=device)

        def __len__(self):
            return len(self.x)

        def __getitem__(self, idx):
            return self.x[idx], self.y[idx]

    model.eval()

    x_trr, y_trr = np.transpose(training_set[0], (0,2,1)), training_set[1]
    x_tee, y_tee = np.transpose(test_set[0], (0,2,1)), test_set[1]
    N_tr = len(x_trr)
    N_te = len(x_tee)
    
    training_set = MyDataset(x_trr, y_trr)
    test_set = MyDataset(x_tee, y_tee)

    training_generator = DataLoader(training_set, batch_size=1,
                                    shuffle=True, drop_last=False)
    test_generator = DataLoader(test_set, batch_size= 1,
                                    shuffle=True, drop_last=False)

    H_tr = th.zeros((N_tr, 128))
    y_tr = th.zeros((N_tr), dtype=th.long)

    H_te = th.zeros((N_te, 128))
    y_te = th.zeros((N_te), dtype=th.long)

    for idx_tr, (x_tr, y_tr_i) in enumerate(training_generator):
        with th.no_grad():
            _, H_tr_i = model(x_tr)
            H_tr[idx_tr] = H_tr_i
            y_tr[idx_tr] = y_tr_i

    H_tr = to_np(nn.functional.normalize(H_tr))
    y_tr = to_np(y_tr)

    for idx_te, (x_te, y_te_i) in enumerate(test_generator):
        with th.no_grad():
            _, H_te_i = model(x_te)
            H_te[idx_te] = H_te_i
            y_te[idx_te] = y_te_i

    H_te = to_np(nn.functional.normalize(H_te))
    y_te = to_np(y_te)

    # clf = KNeighborsClassifier(n_neighbors=1).fit(H_tr, y_tr)
    clf = eval_protocols.fit_knn(H_tr, y_tr)
    acc = clf.score(H_te, y_te)
    y_score = clf.predict_proba(H_te)
    
    return y_score, {'acc': acc}
