from torch import nn

class CombinedNet(nn.Module):
    def __init__(self, net, projection):
        super(CombinedNet, self).__init__()
        self.net = net
        self.projection = projection

    def forward(self, x):
        x = self.net(x)
        x = self.projection(x)
        return x

class LinearProjection(nn.Module):
    def __init__(self, model_output_dim, num_class):
        super(LinearProjection, self).__init__()
        # self.logits = nn.Linear(model_output_dim * train_data_L, num_class)
        self.logits = nn.Linear(model_output_dim, num_class)

    def forward(self, x_in):
        x = x_in
        x_flat = x.reshape(x.shape[0], -1)
        logits = self.logits(x_flat)
        return logits
