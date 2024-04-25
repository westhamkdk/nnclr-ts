import torch
from torch import nn
import torch.nn.functional as F

def hierarchical_contrastive_loss(z1, z2, alpha=0.5, temporal_unit=0):
    # z1: Batch * Dimension * Time
    loss = torch.tensor(0., device=z1.device)
    d = 0
    while z1.size(1) > 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2)
        if d >= temporal_unit:
            if 1 - alpha != 0:
                loss += (1 - alpha) * temporal_contrastive_loss(z1, z2)
        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)
    if z1.size(1) == 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2)
        d += 1
    return loss / d


def instance_contrastive_loss(z1, z2):
    B, T = z1.size(0), z1.size(1)
    if B == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=0)  # 2B x T x C
    z = z.transpose(0, 1)  # T x 2B x C
    sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B [251, 16, 16] <-- 모든 timestamp에 대한 batch간의 similarity matrix
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # T x 2B x (2B-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    
     
    logits = -F.log_softmax(logits, dim=-1)
    
    i = torch.arange(B, device=z1.device)
    loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
    return loss

def instance_contrastive_loss_minus(z1, z2):
    B, T = z1.size(0), z1.size(1)
    if B == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=0)  # 2B x T x C
    z = z.transpose(0, 1)  # T x 2B x C
    sim = - torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B [251, 16, 16] <-- 모든 timestamp에 대한 batch간의 similarity matrix
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # T x 2B x (2B-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    
    i = torch.arange(B, device=z1.device)
    loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
    return loss



def temporal_contrastive_loss(z1, z2):
    B, T = z1.size(0), z1.size(1)
    if T == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=1)  # B x 2T x C
    sim = torch.matmul(z, z.transpose(1, 2))  # B x 2T x 2T
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # B x 2T x (2T-1) <-- 모든 batch에 대한 timestamp 간의 similarity matrix
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    
    t = torch.arange(T, device=z1.device)
    loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
    return loss


import torch
import numpy as np

class NTXentLoss(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        """Criterion has an internal one-hot function. Here, make all positives as 1 while all negatives as 0. """
        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss    


class NTXentLoss_poly(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss_poly, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        zis = zis.reshape(zis.shape[0], -1)
        zjs = zjs.reshape(zis.shape[0], -1)
        
        self.batch_size = zis.shape[0]
        

        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        
        representations = torch.cat([zjs, zis], dim=0)

        # 3d data에 대해서 SIMILARITY를 평가할 수는 없을까 ??
        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        try:
            
            l_pos = torch.diag(similarity_matrix, self.batch_size)
            r_pos = torch.diag(similarity_matrix, -self.batch_size)
            positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)
            negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)
            # 여기 자리에 negative sample 을 뽑게 되는데 z_neg랑 zis간의 similarity matrix를 negative로 취급해야 하겠네

            logits = torch.cat((positives, negatives), dim=1)
            logits /= self.temperature

            """Criterion has an internal one-hot function. Here, make all positives as 1 while all negatives as 0. """
            labels = torch.zeros(2 * self.batch_size).to(self.device).long()
            CE = self.criterion(logits, labels)

            onehot_label = torch.cat((torch.ones(2 * self.batch_size, 1),torch.zeros(2 * self.batch_size, negatives.shape[-1])),dim=-1).to(self.device).long()
            # Add poly loss
            pt = torch.mean(onehot_label* torch.nn.functional.softmax(logits,dim=-1))

            epsilon = self.batch_size
            # loss = CE/ (2 * self.batch_size) + epsilon*(1-pt) # replace 1 by 1/self.batch_size
            loss = CE / (2 * self.batch_size) + epsilon * (1/self.batch_size - pt)
            # loss = CE / (2 * self.batch_size)

            return loss
        except Exception as e:
            print(e)
            
class NTXentLoss_triplet(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss_triplet, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)
    
    def _get_same_label_mask(self, y):
        
        yy = torch.cat([y.unsqueeze(0)]*len(y))
        mask = yy == y.unsqueeze(1)
        mask = mask.type(torch.bool)
        return mask.to(self.device)
        
        
        
        

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs, y):
        # anchor, augmented
        # anchor <-> augmented (strong)
        # minibatch 내에서 다른 애들을 전부 negative로 처리해버리기 때문에 이렇게 할 수 있는거임
        # label정보를 갖고 있지 않으니 index가 다르면 넌 무조건 나랑 달라
        
        '''
        Case A.
        index label --> repr (hidden_dim=4) --> similarity
        0       0                     
        1       0                               
        2       1
        3       2
        4       0
        1. 이 상황에서는 1은 positive, 나머지는 negative
        
        2. 같은 레이블이 여러개 들어왔으면 (랜덤하게) 하나만 positive 선정. 나머지는 negative로 쓰면 ㅄ이니까 아예 계산에서 제외를 시켜버려야 한다.
        similarity
        [0 0.8 0.2 0.1]
        
        random하게 할 수도 있고 similarity가장 높은 애를 positive로 할 수도 있고.
        [0 0.8 0.2 0.1 0.9-->0] --> softmax[0.8, 0.2, 0.1]
        [0 1 1 1 0] --> 이 때 mask out 해야 할 대상은 자기자신 + 같은 label이지만 positive로 선정되지 않았던 모든 같은 label을 가진 데이터
        
        3. 0.8 0.2 0.1 --> CE(softmax([0.8,0,2,0.1]), [1,0,0]) 계산
        
        4. try 할 수 있는것 
        (positive에 augmentation을 넣어본다. negative에 augmentation을 넣어본다.)
        index 4의 데이터에 엄청나게 강한 augmentation을 준다. 그리고 얘를 다른 라벨인 데이터처럼 negative로 취급해버린다.
        
        Case B.
        index 0의 라벨을 모를 때
        index label --> repr (hidden_dim=4) --> similarity
        0       ?                    
        1       0                               
        2       1
        3       2
        4       0
        가장 쉬운 방법은 다른 애들을 전부 negative (augmentatation 강하게 주고)
        자기 자신은 augmentation 약하게 줘서 positive로 삼아버리면 됨
        
        loss 계산 시에 얘네를 index별로 취사선택 할 수 있게
        original  
        weakly augmented data
        strong augmented data
        '''
        # 우리는 label 정보를 갖고 있기 때문에
        # 우리가 아닌 positive sample을 이미 알고 있음
        # 우리는 진짜 negative sample을 원하는 만큼 뽑을 수 있다
        
        zis = zis.reshape(zis.shape[0], -1)
        zjs = zjs.reshape(zjs.shape[0], -1)
        self.batch_size = zis.shape[0]

        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        
        mask_from_same_label = self._get_same_label_mask(y).type(torch.bool)
        
        representations = torch.cat([zis, zjs], dim=0)

        # 3d data에 대해서 SIMILARITY를 평가할 수는 없을까 ??
        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        try:
            l_pos = torch.diag(similarity_matrix, self.batch_size)
            r_pos = torch.diag(similarity_matrix, -self.batch_size)
            positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)
            negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)
            
            # 여기 자리에 negative sample 을 뽑게 되는데 z_neg랑 zis간의 similarity matrix를 negative로 취급해야 하겠네

            # positive 값을 그래서 0번째 column에 몰아넣는 식으로 하고
            # 나머지를 지금은 전부 negative로 때려박고 있는 상황임
            logits = torch.cat((positives, negatives), dim=1)
            logits /= self.temperature

            """Criterion has an internal one-hot function. Here, make all positives as 1 while all negatives as 0. """
            labels = torch.zeros(2 * self.batch_size).to(self.device).long()
            CE = self.criterion(logits, labels)

            onehot_label = torch.cat((torch.ones(2 * self.batch_size, 1),torch.zeros(2 * self.batch_size, negatives.shape[-1])),dim=-1).to(self.device).long()
            # Add poly loss
            pt = torch.mean(onehot_label* torch.nn.functional.softmax(logits,dim=-1))

            epsilon = self.batch_size
            # loss = CE/ (2 * self.batch_size) + epsilon*(1-pt) # replace 1 by 1/self.batch_size
            loss = CE / (2 * self.batch_size) + epsilon * (1/self.batch_size - pt)
            # loss = CE / (2 * self.batch_size)

            return loss
        except Exception as e:
            print(e)
      

def info_nce_loss(z1, z2, y = None):
    
    def _get_same_label_mask(y):
        yy = torch.cat([y.unsqueeze(0)]*len(y))
        mask = yy == y.unsqueeze(1)
        mask = mask.type(torch.bool)
        return mask
    
    z1 = z1.reshape(z1.shape[0], -1)
    z2 = z2.reshape(z2.shape[0], -1)
    
    # Calculate cosine similarity
    cos_sim = F.cosine_similarity(z1[:,None,:], z2[None,:,:], dim=-1)
    # Mask out cosine similarity to itself
    self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=z1.device)
    
    if y != None:
        label_mask = _get_same_label_mask(y).to(z1.device)
    cos_sim.masked_fill_(self_mask, -9e15)
    pos_mask = self_mask.roll(shifts=cos_sim.shape[0]//2, dims=0)
    # InfoNCE loss
    cos_sim = cos_sim / 0.07
    nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
    nll = nll.mean()

    return nll

def rowwise_posneg_loss(out_x, out_pos, out_neg, temperature = 1):
    out_x = out_x.reshape(out_x.shape[0], -1)
    out_pos = out_pos.reshape(out_pos.shape[0], -1)
    out_neg = out_neg.reshape(out_neg.shape[0], -1)
    
    # 여기서 while 문으로 해보면 달라지지 않을까 ??
    
    cos_pos = F.cosine_similarity(out_x, out_pos, dim=-1).to(out_x.device)
    cos_neg = F.cosine_similarity(out_x, out_neg, dim=-1).to(out_x.device)
    
    logits = torch.stack([cos_pos, cos_neg], dim=1).to(out_x.device)
    
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    return criterion((logits/temperature), torch.zeros(out_x.shape[0]).type(torch.LongTensor).to(out_x.device))

# MAX_VALUE = torch.Tensor([1]).cuda()

def knn_nn_loss(out_x, y):
    out_x = F.max_pool1d(out_x.transpose(1, 2),kernel_size = out_x.size(1),).transpose(1, 2)
    
    dist = (out_x.unsqueeze(1)-out_x) ** 2
    dist_matrix = torch.sum(dist, dim=[-1,-2]).to(out_x.device)
    
    # global MAX_VALUE
    
    # if MAX_VALUE == 0:
        # MAX_VALUE = torch.max(dist_matrix).clone().detach()
    
    label_mask = (y.unsqueeze(-1) != y).float().to(out_x.device)
    label_mask[label_mask == 0.] = float('inf')
    
    dist_matrix[label_mask==float('inf')] = float('inf')
    
    knn_values = dist_matrix.topk(1, largest = False, dim=0).values
    knn_indexes = dist_matrix.topk(1, largest = False, dim=0).indices
    knn_values[knn_values == float('inf')] = 0
    
    # knn_values = torch.minimum(MAX_VALUE, knn_values)
    return - knn_values.mean()

def knn_nn_loss_normalized(out_x, y):
    out_x = F.max_pool1d(out_x.transpose(1, 2),kernel_size = out_x.size(1),).transpose(1, 2)
    
    dist = (out_x.unsqueeze(1)-out_x) ** 2
    dist_matrix = torch.nn.functional.normalize(torch.sum(dist, dim=[-1,-2]), p=2.0, dim=0).to(out_x.device)
    
    # global MAX_VALUE
    
    # if MAX_VALUE == 0:
        # MAX_VALUE = torch.max(dist_matrix).clone().detach()
    
    label_mask = (y.unsqueeze(-1) != y).float().to(out_x.device)
    label_mask[label_mask == 0.] = float('inf')
    
    dist_matrix[label_mask==float('inf')] = float('inf')
    
    knn_values = dist_matrix.topk(1, largest = False, dim=0).values
    knn_indexes = dist_matrix.topk(1, largest = False, dim=0).indices
    knn_values[knn_values == float('inf')] = 0
    
    # knn_values = torch.minimum(MAX_VALUE, knn_values)
    return - knn_values.sum()

    

def knn_nn_loss_positive(out_x, y):
    out_x = F.max_pool1d(out_x.transpose(1, 2),kernel_size = out_x.size(1),).transpose(1, 2)
    
    dist = (out_x.unsqueeze(1)-out_x) ** 2
    dist_matrix = torch.sum(dist, dim=[-1,-2]).to(out_x.device)
    
    label_mask = (y.unsqueeze(-1) == y).float().to(out_x.device)
    # label_mask[label_mask == 0.] = - float('inf')
    
    dist_matrix[label_mask== 0] = float('-inf')
    dist_matrix = dist_matrix.fill_diagonal_(float('-inf'))
    
    knn_values = dist_matrix.topk(1, largest = True, dim=0).values
    knn_indexes = dist_matrix.topk(1, largest = True, dim=0).indices
    
    # dist_matrix에 torch.nn.functional.normalize(input, p=2.0, dim = 0) 이렇게 한 다음에 knn_values.sum() 요렇게 퉁쳐버릴까 ??
    
    knn_values[knn_values ==  - float('inf')] = 0
    return knn_values.mean()

def knn_nn_loss_positive_normalized(out_x, y):
    out_x = F.max_pool1d(out_x.transpose(1, 2),kernel_size = out_x.size(1),).transpose(1, 2)
    
    dist = (out_x.unsqueeze(1)-out_x) ** 2
    dist_matrix = torch.nn.functional.normalize(torch.sum(dist, dim=[-1,-2]), p=2.0, dim=0).to(out_x.device)
    
    label_mask = (y.unsqueeze(-1) == y).float().to(out_x.device)
    # label_mask[label_mask == 0.] = - float('inf')
    
    dist_matrix[label_mask== 0] = float('-inf')
    dist_matrix = dist_matrix.fill_diagonal_(float('-inf'))
    
    knn_values = dist_matrix.topk(1, largest = True, dim=0).values
    knn_indexes = dist_matrix.topk(1, largest = True, dim=0).indices
    
    # dist_matrix에 torch.nn.functional.normalize(input, p=2.0, dim = 0) 이렇게 한 다음에 knn_values.sum() 요렇게 퉁쳐버릴까 ??
    
    knn_values[knn_values ==  - float('inf')] = 0
    return knn_values.sum()

    
    
    


    
