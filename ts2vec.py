import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset, Dataset
import numpy as np
from models import TSEncoder
from models.losses import hierarchical_contrastive_loss
from utils import take_per_row, split_with_nan, centerize_vary_length_series, torch_pad_nan, EarlyStopping
import math


class STLEmpiricalDistDataset(TensorDataset):
    def __init__(self, trend, seasonal, residual, original_labels, inv_cdf, inv_cdf_seasonal, inv_cdf_remainder, part_to_sample):
        self.trend = torch.from_numpy(trend).to(torch.float)
        self.seasonal = torch.from_numpy(seasonal).to(torch.float)
        self.residual = torch.from_numpy(residual).to(torch.float)
        self.original_labels = original_labels
        self.inv_cdf = inv_cdf
        self.inv_cdf_seasonal = inv_cdf_seasonal
        self.inv_cdf_remainder = inv_cdf_remainder
        self.part_to_sample = part_to_sample
        
    def __getitem__(self, index):
        x_trend = self.trend[index] # data_len, 1
        x_seasonal = self.seasonal[index] # data_len, 1
        x_residual = self.residual[index] # data_len, 1
        label = self.original_labels[index]    
        
        
        if self.part_to_sample == 'seasonal':
            inv_cdf_to_look = self.inv_cdf_seasonal
        elif self.part_to_sample == 'residual':
            inv_cdf_to_look = self.inv_cdf
        else:
            inv_cdf_to_look = self.inv_cdf_remainder


        if len(inv_cdf_to_look.keys()) == 1:
            cdf = inv_cdf_to_look[0]
        else:
            cdf = inv_cdf_to_look[label]

        tt = np.zeros(len(cdf))
        for i in range(len(cdf)):
            r = np.random.rand(1)
            tt[i] = cdf[i](r)
            
        augmented = torch.from_numpy(tt).to(torch.float)
        
        if self.part_to_sample == 'seasonal':
            summed = torch.fft.ifft(x_trend.T) + torch.fft.ifft(augmented) + torch.fft.ifft(x_residual.T) 
        elif self.part_to_sample == 'residual':
            summed = torch.fft.ifft(x_trend.T) + torch.fft.ifft(x_seasonal.T) + torch.fft.ifft(augmented) 
        else:
            summed = torch.fft.ifft(x_trend.T) + torch.fft.ifft(augmented)

        return summed.real.T, 

    def __len__(self):
        return len(self.trend)
    
    
class STLEmpiricalDistDatasetWithOriginal(STLEmpiricalDistDataset):
    def __init__(self, trend, seasonal, residual, original_labels, inv_cdf, inv_cdf_seasonal, inv_cdf_remainder, part_to_sample, original_data):
        self.original_data = torch.from_numpy(original_data).to(torch.float)
        super().__init__(trend, seasonal, residual, original_labels, inv_cdf, inv_cdf_seasonal, inv_cdf_remainder, part_to_sample)
        
    def __getitem__(self, index):
        return self.original_data[index], super().__getitem__(index)
    
    def __len__(self):
        return super().__len__()

    


class AugCustomDataset(TensorDataset):
    def __init__(self, original_dataset, augmented_dataset):
        self.original_dataset = original_dataset
        self.augmented_dataset = augmented_dataset
        
    def __getitem__(self, index):
        x_original = self.original_dataset[index]
        x_aug = self.augmented_dataset[index]
        return x_original, x_aug

    def __len__(self):
        return len(self.original_dataset)
    
    
    

class TS2Vec:
    '''The TS2Vec model'''
    def __init__(
        self,
        input_dims,
        output_dims=320,
        hidden_dims=64,
        depth=10,
        device='cuda',
        lr=0.001,
        batch_size=16,
        max_train_length=None,
        temporal_unit=0,
        after_iter_callback=None,
        after_epoch_callback=None
    ):
        ''' Initialize a TS2Vec model.
        
        Args:
            input_dims (int): The input dimension. For a univariate time series, this should be set to 1.
            output_dims (int): The representation dimension.
            hidden_dims (int): The hidden dimension of the encoder.
            depth (int): The number of hidden residual blocks in the encoder.
            device (int): The gpu used for training and inference.
            lr (int): The learning rate.
            batch_size (int): The batch size.
            max_train_length (Union[int, NoneType]): The maximum allowed sequence length for training. For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length>.
            temporal_unit (int): The minimum unit to perform temporal contrast. When training on a very long sequence, this param helps to reduce the cost of time and memory.
            after_iter_callback (Union[Callable, NoneType]): A callback function that would be called after each iteration.
            after_epoch_callback (Union[Callable, NoneType]): A callback function that would be called after each epoch.
        '''
        
        super().__init__()
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.max_train_length = max_train_length
        self.temporal_unit = temporal_unit
        
        self._net = TSEncoder(input_dims=input_dims, output_dims=output_dims, hidden_dims=hidden_dims, depth=depth).to(self.device)
        self.net = torch.optim.swa_utils.AveragedModel(self._net)
        self.net.update_parameters(self._net)
        
        self.after_iter_callback = after_iter_callback
        self.after_epoch_callback = after_epoch_callback
        
        self.n_epochs = 0
        self.n_iters = 0
    
    def fit_only_with_augs(self, train_data, n_epochs=None, n_iters=None, verbose=False):
        ''' Training the TS2Vec model.
        Args:
            train_data (numpy.ndarray): The training data. It should have a shape of (n_instance, n_timestamps, n_features). All missing data should be set to NaN.
            n_epochs (Union[int, NoneType]): The number of epochs. When this reaches, the training stops.
            n_iters (Union[int, NoneType]): The number of iterations. When this reaches, the training stops. If both n_epochs and n_iters are not specified, a default setting would be used that sets n_iters to 200 for a dataset with size <= 100000, 600 otherwise.
            verbose (bool): Whether to print the training loss after each epoch.
            
        Returns:
            loss_log: a list containing the training losses on each epoch.
        '''
        assert train_data.ndim == 3
        
        early_stopping = EarlyStopping(patience=500, verbose=True, delta=10e-2)
        
        if self.max_train_length is not None:
            sections = train_data.shape[1] // self.max_train_length
            if sections >= 2:
                train_data = np.concatenate(split_with_nan(train_data, sections, axis=1), axis=0)

        temporal_missing = np.isnan(train_data).all(axis=-1).any(axis=0)
        if temporal_missing[0] or temporal_missing[-1]:
            train_data = centerize_vary_length_series(train_data)
                
        train_data = train_data[~np.isnan(train_data).all(axis=2).all(axis=1)]
        
        # 아 몰라 반 쪼개
        train_original = train_data[:train_data.shape[0]//2, ...]
        train_augmented = train_data[train_data.shape[0]//2:,...]
        
        if n_iters is None and n_epochs is None:
            n_iters = 200 if train_original.size <= 100000 else 600  # default param for n_iters
            print(f"n_iters : {n_iters}")
        
        
        train_org_dataset = TensorDataset(torch.from_numpy(train_original).to(torch.float))
        train_aug_dataset = TensorDataset(torch.from_numpy(train_augmented).to(torch.float))
        
        train_custom_dataset = AugCustomDataset(train_org_dataset, train_aug_dataset)
        train_custom_loader = DataLoader(train_custom_dataset, batch_size=min(self.batch_size, len(train_org_dataset)), shuffle=True, drop_last=True)
        
        optimizer = torch.optim.AdamW(self._net.parameters(), lr=self.lr)
        
        loss_log = []
        
        while True:
            if n_epochs is not None and self.n_epochs >= n_epochs:
                break
            
            cum_loss = 0
            n_epoch_iters = 0
            
            interrupted = False
            # for batch in train_loader:
            for batch, batch_aug in train_custom_loader:
                if n_iters is not None and self.n_iters >= n_iters:
                    interrupted = True
                    break
                
                x_org = batch[0]
                x_aug = batch_aug[0]
                
                if self.max_train_length is not None and x_org.size(1) > self.max_train_length:
                    window_offset = np.random.randint(x_org.size(1) - self.max_train_length + 1)
                    x_org = x_org[:, window_offset : window_offset + self.max_train_length]
                    x_aug = x_aug[:, window_offset : window_offset + self.max_train_length]
                    
                x_org = x_org.to(self.device)
                x_aug = x_aug.to(self.device)
                
                ts_l = x_org.size(1)
                crop_l = np.random.randint(low=2 ** (self.temporal_unit + 1), high=ts_l+1)
                crop_left = np.random.randint(ts_l - crop_l + 1)
                crop_right = crop_left + crop_l
                crop_eleft = np.random.randint(crop_left + 1)
                crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
                crop_offset = np.random.randint(low=-crop_eleft, high=ts_l - crop_eright + 1, size=x_org.size(0))
                
                optimizer.zero_grad()
                
                # out1 = self._net(take_per_row(x_org, crop_offset + crop_eleft, crop_right - crop_eleft))
                # out1 = out1[:, -crop_l:] # 오른쪽 남기고
                # out2 = self._net(take_per_row(x_aug, crop_offset + crop_left, crop_eright - crop_left))
                # out2 = out2[:, :crop_l] # 왼쪽 남기고
                
                # out1 = self._net(x_org)
                # out2 = self._net(x_aug)                
                
                out1 = self._net(take_per_row(x_org, crop_offset + crop_eleft, crop_right - crop_eleft))
                out1 = out1[:, -crop_l:] # 오른쪽 남기고
                out2 = self._net(take_per_row(x_aug, crop_offset + crop_left, crop_eright - crop_left))
                out2 = out2[:, :crop_l] # 왼쪽 남기고
                
                loss = hierarchical_contrastive_loss(
                    out1,
                    out2,
                    temporal_unit=self.temporal_unit
                )
                
                loss.backward()
                optimizer.step()
                self.net.update_parameters(self._net)
                    
                cum_loss += loss.item()
                n_epoch_iters += 1
                self.n_iters += 1
                
                if self.after_iter_callback is not None:
                    self.after_iter_callback(self, loss.item())
            
            if interrupted:
                break
            
            cum_loss /= n_epoch_iters
            loss_log.append(cum_loss)
            if verbose:
                print(f"Epoch #{self.n_epochs}: loss={cum_loss}")
            self.n_epochs += 1
            
            # early_stopping(cum_loss)
            if early_stopping.early_stop:
                print(f"Early stopping at {self.n_epochs} and {self.n_iters}")
                break
            
            if self.after_epoch_callback is not None:
                self.after_epoch_callback(self, cum_loss)
            
        return loss_log
    
    def fit_with_stl_augs(self, train_data, stl_data, train_labels, inv_cdf,inv_cdf_seasonal, inv_cdf_remainder,part_to_sample = 'seasonal', n_epochs = None, n_iters = None, verbose = False, aug_as_pos = False):
        # https://stackoverflow.com/questions/51444059/how-to-iterate-over-two-dataloaders-simultaneously-using-pytorch 내일 이거 보고 aug as pos 부분 처리 할것
        trend, seasonal, residual, _, _, _ = stl_data
        trend = trend[..., np.newaxis]
        seasonal = seasonal[..., np.newaxis]
        residual = residual[..., np.newaxis]
        assert trend.ndim == 3
        
        if n_iters is None and n_epochs is None:
            n_iters = 200 if train_data.size <= 100000 else 600  # default param for n_iters
            # n_iters = 400 if train_data.size <= 100000 else 1000  # default param for n_iters
            print(f"n_iters : {n_iters}")

        tr_data_list = [trend, seasonal, residual, train_data]
        for i, data in enumerate(tr_data_list):
            if self.max_train_length is not None:
                sections = data.shape[1] // self.max_train_length
                if sections >= 2:
                    data = np.concatenate(split_with_nan(data, sections, axis=1), axis=0)
                    
            tr_data_list[i] = data[~np.isnan(data).all(axis=2).all(axis=1)]

        
        original_train_dataset = TensorDataset(torch.from_numpy(train_data).to(torch.float))
        stl_aug_dataset = STLEmpiricalDistDataset(trend, seasonal, residual, train_labels, inv_cdf, inv_cdf_seasonal, inv_cdf_remainder, part_to_sample)
        
        if not aug_as_pos:
            dt_set = ConcatDataset([original_train_dataset, stl_aug_dataset])
            dt_loader = DataLoader(dataset = dt_set, batch_size=min(self.batch_size, len(original_train_dataset) + len(stl_aug_dataset)), shuffle=True, drop_last=True)
        else:
            stl_aug_dataset_with_original = STLEmpiricalDistDatasetWithOriginal(trend, seasonal, residual, train_labels, inv_cdf, inv_cdf_seasonal, inv_cdf_remainder, part_to_sample, train_data)            
            dt_loader = DataLoader(dataset = stl_aug_dataset_with_original, batch_size=min(self.batch_size, len(original_train_dataset)), shuffle=True, drop_last=True)
                
        optimizer = torch.optim.AdamW(self._net.parameters(), lr=self.lr)
        
        loss_log = []
        
        while True:
            if n_epochs is not None and self.n_epochs >= n_epochs:
                break
            
            cum_loss = 0
            n_epoch_iters = 0
            
            interrupted = False
            # for batch in train_loader:
            for batch in dt_loader:
                if n_iters is not None and self.n_iters >= n_iters:
                    interrupted = True
                    break
                
                if not aug_as_pos:
                    x, y = batch[0], batch[0]
                else:
                    x, y = batch[0], batch[1][0]
                    
                if self.max_train_length is not None and x.size(1) > self.max_train_length:
                    window_offset = np.random.randint(x.size(1) - self.max_train_length + 1)
                    x = x[:, window_offset : window_offset + self.max_train_length]
                    y = y[:, window_offset : window_offset + self.max_train_length]

                x = x.to(self.device)
                y = y.to(self.device)
                
                ts_l = x.size(1)
                crop_l = np.random.randint(low=2 ** (self.temporal_unit + 1), high=ts_l+1)
                crop_left = np.random.randint(ts_l - crop_l + 1)
                crop_right = crop_left + crop_l
                crop_eleft = np.random.randint(crop_left + 1)
                crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
                crop_offset = np.random.randint(low=-crop_eleft, high=ts_l - crop_eright + 1, size=x.size(0))
                
                
                # crop_l은 crop_length
                # crop_l만큼의 길이를 가지는 crop_left, crop_right 선정
                # crop_left보다 작은 값을 가지는 crop_eleft를 선정
                # crop_right보다는 큰 값을 가지는 crop_eright를 선정
                # -crop_eleft <-> tsl-crop_right 사이를 움직이는 offset값들을 선정
                
                optimizer.zero_grad()
                
                out1 = self._net(take_per_row(x, crop_offset + crop_eleft, crop_right - crop_eleft))
                out1 = out1[:, -crop_l:] # 오른쪽 남기고
                
                out2 = self._net(take_per_row(y, crop_offset + crop_left, crop_eright - crop_left))
                out2 = out2[:, :crop_l] # 왼쪽 남기고
                
                loss = hierarchical_contrastive_loss(
                    out1,
                    out2,
                    temporal_unit=self.temporal_unit
                )
                
                loss.backward()
                optimizer.step()
                self.net.update_parameters(self._net)
                    
                cum_loss += loss.item()
                n_epoch_iters += 1
                
                self.n_iters += 1
                
                if self.after_iter_callback is not None:
                    self.after_iter_callback(self, loss.item())
            
            if interrupted:
                break
            
            cum_loss /= n_epoch_iters
            loss_log.append(cum_loss)
            if verbose:
                print(f"Epoch #{self.n_epochs}: loss={cum_loss}")
            self.n_epochs += 1
            
            if self.after_epoch_callback is not None:
                self.after_epoch_callback(self, cum_loss)
                
            
        return loss_log
        
    
    
    def fit(self, train_data, n_epochs=None, n_iters=None, verbose=False):
        ''' Training the TS2Vec model.
        
        Args:
            train_data (numpy.ndarray): The training data. It should have a shape of (n_instance, n_timestamps, n_features). All missing data should be set to NaN.
            n_epochs (Union[int, NoneType]): The number of epochs. When this reaches, the training stops.
            n_iters (Union[int, NoneType]): The number of iterations. When this reaches, the training stops. If both n_epochs and n_iters are not specified, a default setting would be used that sets n_iters to 200 for a dataset with size <= 100000, 600 otherwise.
            verbose (bool): Whether to print the training loss after each epoch.
            
        Returns:
            loss_log: a list containing the training losses on each epoch.
        '''
        assert train_data.ndim == 3
        
        if n_iters is None and n_epochs is None:
            n_iters = 200 if train_data.size <= 100000 else 600  # default param for n_iters
            # n_iters = 400 if train_data.size <= 100000 else 1000  # default param for n_iters
            print(f"n_iters : {n_iters}")

        
        if self.max_train_length is not None:
            sections = train_data.shape[1] // self.max_train_length
            if sections >= 2:
                train_data = np.concatenate(split_with_nan(train_data, sections, axis=1), axis=0)

        temporal_missing = np.isnan(train_data).all(axis=-1).any(axis=0)
        if temporal_missing[0] or temporal_missing[-1]:
            train_data = centerize_vary_length_series(train_data)
                
        train_data = train_data[~np.isnan(train_data).all(axis=2).all(axis=1)]
                
        
        train_dataset = TensorDataset(torch.from_numpy(train_data).to(torch.float))
        train_loader = DataLoader(train_dataset, batch_size=min(self.batch_size, len(train_dataset)), shuffle=True, drop_last=True)
        
        optimizer = torch.optim.AdamW(self._net.parameters(), lr=self.lr)
        
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
                
                x = batch[0]
                if self.max_train_length is not None and x.size(1) > self.max_train_length:
                    window_offset = np.random.randint(x.size(1) - self.max_train_length + 1)
                    x = x[:, window_offset : window_offset + self.max_train_length]
                x = x.to(self.device)
                
                ts_l = x.size(1)
                crop_l = np.random.randint(low=2 ** (self.temporal_unit + 1), high=ts_l+1)
                crop_left = np.random.randint(ts_l - crop_l + 1)
                crop_right = crop_left + crop_l
                crop_eleft = np.random.randint(crop_left + 1)
                crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
                crop_offset = np.random.randint(low=-crop_eleft, high=ts_l - crop_eright + 1, size=x.size(0))
                
                
                # crop_l은 crop_length
                # crop_l만큼의 길이를 가지는 crop_left, crop_right 선정
                # crop_left보다 작은 값을 가지는 crop_eleft를 선정
                # crop_right보다는 큰 값을 가지는 crop_eright를 선정
                # -crop_eleft <-> tsl-crop_right 사이를 움직이는 offset값들을 선정
                
                optimizer.zero_grad()
                
                out1 = self._net(take_per_row(x, crop_offset + crop_eleft, crop_right - crop_eleft))
                out1 = out1[:, -crop_l:] # 오른쪽 남기고
                
                out2 = self._net(take_per_row(x, crop_offset + crop_left, crop_eright - crop_left))
                out2 = out2[:, :crop_l] # 왼쪽 남기고
                
                loss = hierarchical_contrastive_loss(
                    out1,
                    out2,
                    temporal_unit=self.temporal_unit
                )
                
                loss.backward()
                optimizer.step()
                self.net.update_parameters(self._net)
                    
                cum_loss += loss.item()
                n_epoch_iters += 1
                
                self.n_iters += 1
                
                if self.after_iter_callback is not None:
                    self.after_iter_callback(self, loss.item())
            
            if interrupted:
                break
            
            cum_loss /= n_epoch_iters
            loss_log.append(cum_loss)
            if verbose:
                print(f"Epoch #{self.n_epochs}: loss={cum_loss}")
            self.n_epochs += 1
            
            if self.after_epoch_callback is not None:
                self.after_epoch_callback(self, cum_loss)
                
            
        return loss_log
    
    def _eval_with_pooling(self, x, mask=None, slicing=None, encoding_window=None):
    
        out = self.net(x.to(self.device, non_blocking=True), mask)
        # out = self._net(x.to(self.device, non_blocking=True), mask)
        

        if encoding_window == 'full_series':
            if slicing is not None:
                out = out[:, slicing]
            out = F.max_pool1d(
                out.transpose(1, 2),
                kernel_size = out.size(1),
            ).transpose(1, 2)
                        
        elif isinstance(encoding_window, int):
            out = F.max_pool1d(
                out.transpose(1, 2),
                kernel_size = encoding_window,
                stride = 1,
                padding = encoding_window // 2
            ).transpose(1, 2)
            if encoding_window % 2 == 0:
                out = out[:, :-1]
            if slicing is not None:
                out = out[:, slicing]
            
        elif encoding_window == 'multiscale':
            p = 0
            reprs = []
            while (1 << p) + 1 < out.size(1):
                t_out = F.max_pool1d(
                    out.transpose(1, 2),
                    kernel_size = (1 << (p + 1)) + 1,
                    stride = 1,
                    padding = 1 << p
                ).transpose(1, 2)
                if slicing is not None:
                    t_out = t_out[:, slicing]
                reprs.append(t_out)
                p += 1
            out = torch.cat(reprs, dim=-1)
            
        else:
            if slicing is not None:
                out = out[:, slicing]
            
        return out.cpu()
    
    def encode(self, data, mask=None, encoding_window=None, casual=False, sliding_length=None, sliding_padding=0, batch_size=None):
        ''' Compute representations using the model.
        Args:
            data (numpy.ndarray): This should have a shape of (n_instance, n_timestamps, n_features). All missing data should be set to NaN.
            mask (str): The mask used by encoder can be specified with this parameter. This can be set to 'binomial', 'continuous', 'all_true', 'all_false' or 'mask_last'.
            encoding_window (Union[str, int]): When this param is specified, the computed representation would the max pooling over this window. This can be set to 'full_series', 'multiscale' or an integer specifying the pooling kernel size.
            casual (bool): When this param is set to True, the future informations would not be encoded into representation of each timestamp.
            sliding_length (Union[int, NoneType]): The length of sliding window. When this param is specified, a sliding inference would be applied on the time series.
            sliding_padding (int): This param specifies the contextual data length used for inference every sliding windows.
            batch_size (Union[int, NoneType]): The batch size used for inference. If not specified, this would be the same batch size as training.
            
        Returns:
            repr: The representations for data.
        '''
        assert self.net is not None, 'please train or load a net first'
        # assert self._net is not None, 'please train or load a net first'

        assert data.ndim == 3
        if batch_size is None:
            batch_size = self.batch_size
        n_samples, ts_l, _ = data.shape

        org_training = self.net.training
        # org_training = self._net.training

        self.net.eval()
        # self._net.eval()

        
        dataset = TensorDataset(torch.from_numpy(data).to(torch.float))
        loader = DataLoader(dataset, batch_size=batch_size)
        
        with torch.no_grad():
            output = []
            for batch in loader:
                x = batch[0]
                if sliding_length is not None:
                    reprs = []
                    if n_samples < batch_size:
                        calc_buffer = []
                        calc_buffer_l = 0
                    for i in range(0, ts_l, sliding_length):
                        l = i - sliding_padding
                        r = i + sliding_length + (sliding_padding if not casual else 0)
                        x_sliding = torch_pad_nan(
                            x[:, max(l, 0) : min(r, ts_l)],
                            left=-l if l<0 else 0,
                            right=r-ts_l if r>ts_l else 0,
                            dim=1
                        )
                        if n_samples < batch_size:
                            if calc_buffer_l + n_samples > batch_size:
                                out = self._eval_with_pooling(
                                    torch.cat(calc_buffer, dim=0),
                                    mask,
                                    slicing=slice(sliding_padding, sliding_padding+sliding_length),
                                    encoding_window=encoding_window
                                )
                                reprs += torch.split(out, n_samples)
                                calc_buffer = []
                                calc_buffer_l = 0
                            calc_buffer.append(x_sliding)
                            calc_buffer_l += n_samples
                        else:
                            out = self._eval_with_pooling(
                                x_sliding,
                                mask,
                                slicing=slice(sliding_padding, sliding_padding+sliding_length),
                                encoding_window=encoding_window
                            )
                            reprs.append(out)

                    if n_samples < batch_size:
                        if calc_buffer_l > 0:
                            out = self._eval_with_pooling(
                                torch.cat(calc_buffer, dim=0),
                                mask,
                                slicing=slice(sliding_padding, sliding_padding+sliding_length),
                                encoding_window=encoding_window
                            )
                            reprs += torch.split(out, n_samples)
                            calc_buffer = []
                            calc_buffer_l = 0
                    
                    out = torch.cat(reprs, dim=1)
                    if encoding_window == 'full_series':
                        out = F.max_pool1d(
                            out.transpose(1, 2).contiguous(),
                            kernel_size = out.size(1),
                        ).squeeze(1)
                else:
                    out = self._eval_with_pooling(x, mask, encoding_window=encoding_window)
                    if encoding_window == 'full_series':
                        out = out.squeeze(1)
                        
                output.append(out)
                
            output = torch.cat(output, dim=0)
            
        self.net.train(org_training)
        # self._net.train(org_training)

        return output.numpy()
    
    def save(self, fn):
        ''' Save the model to a file.
        
        Args:
            fn (str): filename.
        '''
        # torch.save(self._net.state_dict(), fn)
        torch.save(self.net.state_dict(), fn)

    
    def load(self, fn):
        ''' Load the model from a file.
        
        Args:
            fn (str): filename.
        '''
        state_dict = torch.load(fn, map_location=self.device)
        self.net.load_state_dict(state_dict)
        # self._net.load_state_dict(state_dict)

    
