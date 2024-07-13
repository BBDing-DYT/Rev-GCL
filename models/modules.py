import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from torch.nn.parameter import Parameter
import numpy as np


class UpSampleConvnext(nn.Module):
    def __init__(self, inchannel, outchannel):
        super().__init__()
        self.channel_reschedule = nn.Sequential(  
                                        # LayerNorm(inchannel, eps=1e-6, data_format="channels_last"),
                                        nn.Linear(inchannel, outchannel),
                                        # nn.LayerNorm(outchannel, eps=1e-5, elementwise_affine=True)
                                        nn.ReLU()
        )
    def forward(self, x):
        x = self.channel_reschedule(x)
        return x

class DownSampleConvnext(nn.Module):
    def __init__(self, inchannel, outchannel):
        super().__init__()
        self.channel_reschedule = nn.Sequential(
                                        # LayerNorm(inchannel, eps=1e-6, data_format="channels_last"),
                                        nn.Linear(inchannel, outchannel),
                                        # nn.LayerNorm(outchannel, eps=1e-5, elementwise_affine=True)
                                        nn.ReLU()
        )
    def forward(self, x):
        x = self.channel_reschedule(x)
        return x

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first", elementwise_affine = True):
        super().__init__()
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            if self.elementwise_affine:
                x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ConvNextBlock(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, in_channel, hidden_dim, out_channel, kernel_size=3, layer_scale_init_value=1e-6, drop_path= 0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(in_channel, in_channel, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, groups=in_channel) # depthwise conv
        self.norm = nn.LayerNorm(in_channel, eps=1e-6)
        self.pwconv1 = nn.Linear(in_channel, hidden_dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(hidden_dim, out_channel)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((out_channel)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        # print(f"x min: {x.min()}, x max: {x.max()}, input min: {input.min()}, input max: {input.max()}, x mean: {x.mean()}, x var: {x.var()}, ratio: {torch.sum(x>8)/x.numel()}")
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class Decoder(nn.Module):
    def __init__(self, depth=[2,2,2,2], dim=[112, 72, 40, 24], dropout=0.0) -> None:
        super().__init__()
        self.depth = depth
        self.dim = dim
        # self.gc2 = GraphConvolution(dim[0], dim[0]//4, dropout, act=lambda x: x)
        # self.gc3 = GraphConvolution(dim[0], dim[0]//4, dropout, act=lambda x: x)
        self.gc2 = nn.Linear(dim[0], dim[0]//4)
        self.gc3 = nn.Linear(dim[0], dim[0]//4)
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)

    def encode(self, x, adj):
        return self.gc2(x), self.gc3(x)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj):
        mu, logvar = self.encode(x, adj)
        # print(mu)
        # print("=============")
        # print(logvar)
        # print("=============")
        z = self.reparameterize(mu, logvar)
        # print(z)
        return self.dc(z), mu, logvar

# class LinDecoder(nn.Module):
#     def __init__(self, depth=[2,2,2,2], dim=[112, 72, 40, 24], dropout=0.0) -> None:
#         super().__init__()
#         proj_layers = nn.ModuleList()
#         for i in range(1, len(dim)):
#             proj_layers.append(nn.Sequential(
#                 nn.LayerNorm(dim[i-1], eps=1e-5, elementwise_affine=True),
#                 nn.Linear(dim[i-1], dim[i]),
#                 # nn.LayerNorm(1433, eps=1e-5, elementwise_affine=True)
#             ))
#         self.proj_layers = proj_layers
#         self.projback = nn.Sequential(
#             nn.LayerNorm(dim[-1], eps=1e-5, elementwise_affine=True),
#             nn.Linear(dim[-1],1433),
#         )
#     def forward(self, c3):
#         x = self.proj_layers[0](c3)
#         x = self.proj_layers[1](x)
#         x = self.proj_layers[2](x)
#         x = self.projback(x)
#         return x

class LinDecoder(nn.Module):
    def __init__(self, depth=[2,2,2,2], dim=[112, 72, 40, 24], dropout=0.0) -> None:
        super().__init__()
        self.projback = nn.Sequential(
            nn.LayerNorm(dim[0], eps=1e-5, elementwise_affine=True),
            nn.Linear(dim[0], dim[3]),
            nn.LayerNorm(dim[3], eps=1e-5, elementwise_affine=True),
            nn.Linear(dim[3],1433),
        )
    def forward(self, c3):
        return self.projback(c3)

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = Parameter(torch.FloatTensor(in_features, out_features).to('cuda:0'))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training)
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        output = self.act(output)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj

class SparseInputLinear(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(SparseInputLinear, self).__init__()
        weight = np.zeros((inp_dim, out_dim), dtype=np.float32)
        weight = nn.Parameter(torch.from_numpy(weight))

        bias = np.zeros(out_dim, dtype=np.float32)
        bias = nn.Parameter(torch.from_numpy(bias))
        self.inp_dim, self.out_dim = inp_dim, out_dim
        self.weight, self.bias = weight, bias
        self.reset_parameters()

    def reset_parameters(self):
        # pre-init
        stdv = 1. / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        return torch.mm(x, self.weight) + self.bias

class Classifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.LayerNorm(in_channels, eps=1e-5, elementwise_affine=True), # final norm layer
            nn.Linear(in_channels, num_classes)
        )

    def forward(self, x):
        x = self.classifier(x)
        return x

class GCNModelVAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(GCNModelVAE, self).__init__()
        self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.gc3 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)

    def encode(self, x, adj):
        hidden1 = self.gc1(x, adj)
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj):
        mu, logvar = self.encode(x, adj)
        # print(mu)
        # print(logvar)
        z = self.reparameterize(mu, logvar)
        # print(z)
        # print(self.dc(z))
        return self.dc(z), mu, logvar

