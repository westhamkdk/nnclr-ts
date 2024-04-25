import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset, Dataset
import numpy as np
from models import TSEncoder
from models.losses import *
from utils import take_per_row, split_with_nan, centerize_vary_length_series, torch_pad_nan, EarlyStopping
import math
from collections import deque

LABEL_UNDEFINED = -1        

class STLEmpiricalDistDataset(TensorDataset):
    def __init__(self, trend, seasonal, residual, original_labels, inv_cdf, inv_cdf_seasonal, inv_cdf_remainder, inv_cdf_trend,part_to_sample):
        self.trend = torch.from_numpy(trend).to(torch.float)
        self.seasonal = torch.from_numpy(seasonal).to(torch.float)
        self.residual = torch.from_numpy(residual).to(torch.float)
        self.original_labels = original_labels
        self.inv_cdf = inv_cdf
        self.inv_cdf_seasonal = inv_cdf_seasonal
        self.inv_cdf_remainder = inv_cdf_remainder
        self.inv_cdf_trend = inv_cdf_trend
        self.part_to_sample = part_to_sample
        
    def __getitem__(self, index):
        x_trend = self.trend[index] # data_len, 1
        x_seasonal = self.seasonal[index] # data_len, 1
        x_residual = self.residual[index] # data_len, 1


        
        if self.part_to_sample == 'seasonal':
            inv_cdf_to_look = self.inv_cdf_seasonal
        elif self.part_to_sample == 'residual':
            inv_cdf_to_look = self.inv_cdf
        elif self.part_to_sample == 'trend':
            inv_cdf_to_look = self.inv_cdf_trend
        else:
            inv_cdf_to_look = self.inv_cdf_remainder

        cdf = inv_cdf_to_look[0]

        # tt = np.zeros(len(cdf))
        tt = []
        for i in range(len(cdf)):
            r = np.random.rand(2)
            # tt[i] = cdf[i](r)
            tt.append(cdf[i](r))
            
        augmented = torch.from_numpy(np.array(tt)).to(torch.float)

        x_trend = np.array(x_trend)
        x_seasonal = np.array(x_seasonal)
        x_residual = np.array(x_residual)
        original_length = x_trend.size

        x_trend = x_trend[~np.isnan(x_trend)]
        x_seasonal = x_seasonal[~np.isnan(x_seasonal)]
        x_residual = x_residual[~np.isnan(x_residual)]

        x_trend = torch.from_numpy(np.pad(torch.fft.ifft(torch.from_numpy(x_trend)).to(torch.float), (0, original_length - len(x_trend)), 'constant', constant_values = np.nan)).to(torch.float)
        x_seasonal = torch.from_numpy(np.pad(torch.fft.ifft(torch.from_numpy(x_seasonal)).to(torch.float), (0, original_length - len(x_seasonal)), 'constant', constant_values = np.nan)).to(torch.float)
        x_residual = torch.from_numpy(np.pad(torch.fft.ifft(torch.from_numpy(x_residual)).to(torch.float), (0, original_length - len(x_residual)), 'constant', constant_values = np.nan)).to(torch.float)

        x_augmented = torch.fft.ifft(augmented.T)

        # shape가 1, 840이 되면 됨
        if self.part_to_sample == 'seasonal':
            summed = x_trend + x_augmented + x_residual
        elif self.part_to_sample == 'residual':
            summed = x_trend + x_augmented + x_seasonal 
        elif self.part_to_sample == 'trend':
            summed = x_seasonal + x_augmented + x_residual 
        else:
            summed = x_trend + x_augmented

        

        # if self.part_to_sample == 'seasonal':
        #     summed = torch.fft.ifft(x_trend.T) + torch.fft.ifft(augmented.T) + torch.fft.ifft(x_residual.T) 
        # elif self.part_to_sample == 'residual':
        #     summed = torch.fft.ifft(x_trend.T) + torch.fft.ifft(x_seasonal.T) + torch.fft.ifft(augmented.T) 
        # elif self.part_to_sample == 'trend':
        #     summed = torch.fft.ifft(augmented.T) + torch.fft.ifft(x_seasonal.T) + torch.fft.ifft(x_residual.T) 
        # else:
        #     summed = torch.fft.ifft(x_trend.T) + torch.fft.ifft(augmented.T)

        summed = summed.real.T
        return summed[:, 0], summed[:, 1], 

    def __len__(self):
        return len(self.trend)
    
class STLEmpiricalDistDatasetWithOriginal(STLEmpiricalDistDataset):
    def __init__(self, trend, seasonal, residual, original_labels, inv_cdf, inv_cdf_seasonal, inv_cdf_remainder,inv_cdf_trend, part_to_sample, original_data, label_mask):
        self.original_data = torch.from_numpy(original_data).to(torch.float)
        self.label_mask = label_mask
        super().__init__(trend, seasonal, residual, original_labels, inv_cdf, inv_cdf_seasonal, inv_cdf_remainder,inv_cdf_trend, part_to_sample)
        
    def __getitem__(self, index):
        return self.original_data[index], super().__getitem__(index), index
    
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
    
    
    
class CustomNets:
    def __init__(self, name, input_dims, output_dims, hidden_dims, depth, device, lr, train_data_L):
        self.name = name
        self._net = TSEncoder(input_dims=input_dims, output_dims=output_dims, hidden_dims=hidden_dims, depth=depth, train_data_L = train_data_L, mask_mode = 'all_true').to(device)
        self.net = torch.optim.swa_utils.AveragedModel(self._net)
        self.optimizer = torch.optim.AdamW(self._net.parameters(), lr=lr)

        
    def update_parameters(self):
        self.net.update_parameters(self._net)
        
        

class NNCLR_anomaly_self_supervised_version:
    '''The Nearest-neighbor contrastive learning model'''
    def __init__(
        self,
        train_data_L,
        input_dims,
        output_dims=320,
        hidden_dims=64,
        depth=10,
        device='cuda',
        pool_support_embed = True,
        use_augment = True,
        module_list = ['spectral', 'temporal'],
        loss = 'ts2vec',
        train_longer = True,
        nearest_selection_mode = 'cossim',
        knn_loss_negative_lambda = 0.5,
        alpha = 0.5,
        lr=0.001,
        batch_size=16,
        max_train_length=None,
        temporal_unit=0,
        after_iter_callback=None,
        after_epoch_callback=None,
        max_supset_size = 0.8,
        label_mask_index = None,
        limit_support_set = True,
        use_knn_loss = False,
        top_k = 0,
        use_pseudo_labeling = False,
        label_ratio = False

    ):
        ''' Initialize a NNCLR model.
        
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
        
        self.train_data_L = train_data_L
        
        self.all_networks = []
        for module in module_list:
            cnet = CustomNets(module, input_dims, output_dims, hidden_dims, depth, device, lr, self.train_data_L)
            self.all_networks.append(cnet)
                
        self.update_all_networks()
        
        self.after_iter_callback = after_iter_callback
        self.after_epoch_callback = after_epoch_callback
        
        self.n_epochs = 0
        self.n_iters = 0
        
        self.use_augment = use_augment
        self.loss = loss
        self.train_longer = train_longer
        
        self.pool_support_embed = pool_support_embed
        self.nearest_selection_mode = nearest_selection_mode
        self.knn_loss_negative_lambda = knn_loss_negative_lambda
        self.alpha = alpha
        self.max_supset_size = max_supset_size

        self.label_mask_index = label_mask_index
        self.limit_support_set = limit_support_set
        
    def update_all_networks(self):
        for network in self.all_networks:
            network.optimizer.step()
            network.update_parameters()
     
    def fit(self, train_data, stl_data, train_labels, inv_cdf,inv_cdf_seasonal, inv_cdf_remainder,inv_cdf_trend, part_to_sample = 'seasonal', n_epochs = None, n_iters = None, verbose = False, aug_as_pos = False):
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
        
        # Support set buffer initialization
        FLAG_USE_NN_FROM_SUPPORT_SET = True
        data_size = train_data.size
        LIMIT_SIZE = 1000000
        SAFE_SIZE = int(self.max_supset_size * LIMIT_SIZE)
        if self.limit_support_set:
            if data_size >= LIMIT_SIZE:
                supset_maxlen = int(SAFE_SIZE // train_data.shape[1])
                print(f"data size exceedes {LIMIT_SIZE}, actual size : {data_size}. Support set size adjusted to {supset_maxlen}. Train data size is {train_data.shape[0]}")
            else:
                supset_maxlen = train_data.shape[0]
        else:
            supset_maxlen = train_data.shape[0]

        print(f"Support set size set to {supset_maxlen}. Train data size is {train_data.shape[0]}")
            
        self.support_set = deque(maxlen=supset_maxlen)
                
        trend, seasonal, residual, _, _, _, original= stl_data
        trend = trend[..., np.newaxis]
        seasonal = seasonal[..., np.newaxis]
        residual = residual[..., np.newaxis]
        original = original[..., np.newaxis]


        # Label masking for semi-supervised learning
        assert trend.ndim == 3
        
        # set train iters
        if n_iters is None and n_epochs is None:
            if self.train_longer:
                n_iters = 500 if train_data.size <= 100000 else 1200  # default param for n_iters                
            else:
                n_iters = 200 if train_data.size <= 100000 else 600  # default param for n_iters
                # n_iters = 50 if train_data.size <= 100000 else 50  # default param for n_iters

            
            print(f"n_iters : {n_iters}")

        # split sections
        tr_data_list = [trend, seasonal, residual, train_data]
        for i, data in enumerate(tr_data_list):
            if self.max_train_length is not None:
                sections = data.shape[1] // self.max_train_length
                if sections >= 2:
                    data = np.concatenate(split_with_nan(data, sections, axis=1), axis=0)
                    
            tr_data_list[i] = data[~np.isnan(data).all(axis=2).all(axis=1)]
                
        original_dataset = TensorDataset(torch.from_numpy(train_data).to(torch.float))
        '''
        TODO: Create stratified sampling (reference : https://discuss.pytorch.org/t/how-to-enable-the-dataloader-to-sample-from-each-class-with-equal-probability/911)
        dataset에서 index 가져오기
        '''
        # shuffle only once
        train_dataset = STLEmpiricalDistDatasetWithOriginal(trend, seasonal, residual, train_labels, inv_cdf, inv_cdf_seasonal, inv_cdf_remainder,inv_cdf_trend, part_to_sample, train_data, _)            
        # train_dataset = torch.utils.data.Subset(train_dataset, rand_conditional)
        # original_dataset = torch.utils.data.Subset(original_dataset, rand_conditional)
        train_loader = DataLoader(train_dataset, batch_size=min(self.batch_size, len(train_dataset)), shuffle=False, drop_last=False)
        
        nx_xent_criterion = NTXentLoss_poly(self.device, self.batch_size, 0.2, True)
        
        early_stopping = EarlyStopping(patience=3, verbose=True, delta=10e-2)
        # early_stopping = EarlyStopping(patience=3, verbose=True)        
        
        loss_log = []
        while True:
            if n_epochs is not None and self.n_epochs >= n_epochs:
                break
            cum_loss = 0
            n_epoch_iters = 0
            
            interrupted = False

            for b_index, batch in enumerate(train_loader):
                if batch[0].shape[0] >= 0:
                    if n_iters is not None and self.n_iters >= n_iters:
                        interrupted = True
                        break
                    # label_mask가 0이면 pseudo-label을 해야 함
                    x_org, x_pos1_org, x_pos2_org, index = batch[0], batch[1][0].unsqueeze(-1), batch[1][1].unsqueeze(-1), batch[2]
                    total_out_x, total_out_pos1, total_out_pos2 = [],[],[]
                    np.arange(b_index*self.batch_size, b_index*self.batch_size+x_org.shape[0])
                    current_index_start = b_index * self.batch_size
                    current_index_end = b_index*self.batch_size+x_org.shape[0]
                    
                    '''
                    pseudo-labeling 방법
                    support set 내에 저장할 정보
                    label_info (이건 귀찮으니까 일단 냅두고) 
                    label_info를 바탕으로 masking을 생성함
                    
                    label_info
                    pseudolabel_info

                    2. label_info는 다 들고 있게 시키고
                    label_info, pseudolabel_info, is_label_fixed
                    이후에 마스크를 생성할때
                    label_info[is_label_fixed] + pseudolabel_info[~is_label_fixed]
                    '''                        
                    for network in self.all_networks:
                        x = x_org.clone()
                        x_pos1 = x_pos1_org.clone()
                        x_pos2 = x_pos2_org.clone()
                        
                        x = x.to(self.device)                
                        
                        if network.name == 'spectral':
                            # apply rFFT transform
                            x = torch.fft.rfft(x, norm = 'ortho').real
                        
                        if not self.use_augment:
                            if network.name == 'temporal':
                                x_pos1 = x.clone()
                                x_pos2 = x.clone()
                            elif network.name == 'spectral':
                                x_pos1 = x.clone()
                                x_pos2 = x.clone()
                        
                        x_pos1 = x_pos1.to(self.device)
                        x_pos2 = x_pos2.to(self.device)
                                                
                        network.optimizer.zero_grad()
                        
                        total_out_x.append(network._net(x))
                        total_out_pos1.append(network._net(x_pos1))
                        total_out_pos2.append(network._net(x_pos2))
                        
                    out_x = torch.cat(total_out_x, dim = 2).to(self.device)
                    
                    out_pos1 = torch.cat(total_out_pos1, dim = 2).to(self.device)
                    out_pos2 = torch.cat(total_out_pos2, dim = 2).to(self.device)
                    
                    # initialize NN-positive, NN-negative
                    closest_x = torch.zeros_like(out_x).to(self.device)
                    closest_x_negative = torch.zeros_like(out_x).to(self.device)
                    

                    if FLAG_USE_NN_FROM_SUPPORT_SET:
                    # if len(self.support_set) > 0.2 * train_data.shape[0]:
                        if len(self.support_set) != 0:
                            # Find nearest neighbor of out_pos1
                            sup_set_temporal = torch.cat([x for x,_,_,_,_ in self.support_set], dim=0)
                            sup_set_temporal = sup_set_temporal.to(self.device)                            
                            sup_set_temporal_indexes = torch.Tensor([z for _,_,z,_,_ in self.support_set])
                            
                            current_indexes = torch.Tensor(np.arange(current_index_start, current_index_end))

                            x_to_clone = out_pos1.clone()
                                
                            if self.pool_support_embed:
                                b = F.max_pool1d(x_to_clone.detach().transpose(1, 2),kernel_size = x_to_clone.size(1),).transpose(1, 2)
                                a = F.max_pool1d(sup_set_temporal.transpose(1,2), kernel_size = sup_set_temporal.size(1),).transpose(1,2)                            
                            else:
                                b = x_to_clone.detach()
                                a = sup_set_temporal                            
                            
                            sup_set_temporal = sup_set_temporal.to(self.device)
                            
                            a = sup_set_temporal.unsqueeze(dim=1)                        
                            a = sup_set_temporal.unsqueeze(dim=1).cpu()                        
                            b = b.detach().cpu()
                            
                            c = (a-b) ** 2
                            
                            index_mask = (sup_set_temporal_indexes.unsqueeze(-1) != current_indexes).float().to(self.device)
                            index_mask[index_mask == 0.] = float('inf')

                            if self.nearest_selection_mode == 'cossim':
                                cos = torch.nn.CosineSimilarity(dim=-1)
                                dist_matrix = cos(a.squeeze().unsqueeze(1), b.squeeze()).to(self.device)
                            else:
                                dist_matrix = torch.sum(c, dim=[-1,-2]).to(self.device)
                                
                            dist_matrix  = dist_matrix * index_mask
                            
                            knn_index = dist_matrix.topk(1, largest = False, dim=0).indices.squeeze()
                            closest_x = sup_set_temporal[knn_index, :]
                                    
                        else:
                            closest_x = out_pos1
                    else:
                        closest_x = out_pos1
                            
                    closest_x = closest_x.to(self.device)
          
                    for x_item, z_item in zip(torch.split(out_x.detach().cpu(), 1), np.arange(current_index_start, current_index_end)):
                        self.support_set.append([x_item,_,z_item, _, _])
                    
                    
                    if out_x.dim() != 2:  
                        if self.loss == 'ts2vec':
                            loss = hierarchical_contrastive_loss(
                                closest_x,
                                out_pos2,
                                temporal_unit=self.temporal_unit
                            )

                            # loss = hierarchical_contrastive_loss(
                            #     out_pos1,
                            #     out_pos2,
                            #     temporal_unit=self.temporal_unit
                            # )

                            

                        elif self.loss == 'infonce':
                            if torch.max(closest_x_negative) != 0:
                                loss_pos = instance_contrastive_loss(
                                    closest_x,
                                    out_pos2,
                                    # temporal_unit=self.temporal_unit                                    
                                )
                                
                                # loss = self.alpha * loss_pos
                                loss = loss_pos
                                
                            else:
                                loss_pos = instance_contrastive_loss(
                                    closest_x,
                                    out_pos2,
                                    # temporal_unit=self.temporal_unit
                                )
                                
                                loss = loss_pos
                        
                        loss.backward()
                        self.update_all_networks()                
                            
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
    
    def _eval_with_pooling(self, x, mask=None, slicing=None, encoding_window=None):
        outs = []
        for network in self.all_networks:
            out = network.net(x.to(self.device, non_blocking=True), mask)
            # if slicing is not None:
            #     out = out[:, slicing]
            # out = F.max_pool1d(
            #     out.transpose(1, 2),
            #     kernel_size = out.size(1),
            # ).transpose(1, 2)
            outs.append(out)
         
        out = torch.cat(outs, dim=2)

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
        assert self.all_networks[0].net is not None, 'please train or load a net first'
        # assert self._net is not None, 'please train or load a net first'

        assert data.ndim == 3
        if batch_size is None:
            batch_size = self.batch_size
        n_samples, ts_l, _ = data.shape

        for network in self.all_networks:
            org_training = network.net.training
            network.net.eval()
        
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
            
            
        for network in self.all_networks:
            network.net.train(org_training)

        return output.numpy()
    
    def save(self, fn):
        ''' Save the model to a file.
        
        Args:
            fn (str): filename.
        '''
        model_keys = [m.name for m in self.all_networks]
        model_states = [m.net.state_dict() for m in self.all_networks]
        dd = dict(zip(model_keys, model_states))        
        torch.save(dd, fn)

    
    def load(self, fn):
        ''' Load the model from a file.
        
        Args:
            fn (str): filename.
        '''
        ckpt = torch.load(fn, map_location=self.device)
        for network in self.all_networks:
            network.net.load_state_dict(ckpt[network.name])

    
