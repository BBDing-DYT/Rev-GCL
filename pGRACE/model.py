from models.modules import *
from models.revcol_function import ReverseFunction
from timm.models.layers import trunc_normal_
import numpy as np
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
import math
from typing import Optional
from torch.nn import Parameter
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class Encoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation, base_model=GCNConv, k: int = 2, skip=False):
        super(Encoder, self).__init__()
        self.base_model = base_model

        assert k >= 2
        self.k = k
        self.skip = skip
        if not self.skip:
            self.conv = [base_model(in_channels, 2 * out_channels).jittable()]
            for _ in range(1, k - 1):
                self.conv.append(base_model(2 * out_channels, 2 * out_channels))
            self.conv.append(base_model(2 * out_channels, out_channels))
            self.conv = nn.ModuleList(self.conv)

            self.activation = activation
        else:
            self.fc_skip = nn.Linear(in_channels, out_channels)
            self.conv = [base_model(in_channels, out_channels)]
            for _ in range(1, k):
                self.conv.append(base_model(out_channels, out_channels))
            self.conv = nn.ModuleList(self.conv)

            self.activation = activation

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        if not self.skip:
            for i in range(self.k):
                x = self.activation(self.conv[i](x, edge_index))
            return x
        else:
            h = self.activation(self.conv[0](x, edge_index))
            hs = [self.fc_skip(x), h]
            for i in range(1, self.k):
                u = sum(hs)
                hs.append(self.activation(self.conv[i](u, edge_index)))
            return hs[-1]


class GRACE(torch.nn.Module):
    def __init__(self, encoder: Encoder, num_hidden: int, num_proj_hidden: int, tau: float = 0.5):
        super(GRACE, self).__init__()
        self.encoder: Encoder = encoder
        self.tau: float = tau

        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)

        self.num_hidden = num_hidden

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, batch_size: int):
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))
            between_sim = f(self.sim(z1[mask], z2))

            losses.append(-torch.log(between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                                     / (refl_sim.sum(1) + between_sim.sum(1)
                                        - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)

    def loss(self, z1: torch.Tensor, z2: torch.Tensor, mean: bool = True, batch_size: Optional[int] = None):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size is None:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret

def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)
def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)



class NewGConv(MessagePassing):
    r"""The graph convolutional operator from the `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        normalize (bool, optional): Whether to add self-loops and apply
            symmetric normalization. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, in_channels, out_channels, improved=False, cached=False,
                 bias=True, normalize=True, **kwargs):
        super(NewGConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
      
        self.cached = cached
        self.normalize = normalize

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        self.cached_result = None
        self.cached_num_edges = None


    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, improved=False,
             dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

        return edge_index, edge_weight


    def forward(self, x, edge_index, c, edge_weight=None):
        """"""
        x = torch.matmul(x, self.weight)
    
        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}. Please '
                    'disable the caching behavior of this layer by removing '
                    'the `cached=True` argument in its constructor.'.format(
                        self.cached_num_edges, edge_index.size(1)))

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            if self.normalize:
                edge_index, norm = self.norm(edge_index, x.size(
                    self.node_dim), edge_weight, self.improved, x.dtype)
            else:
                norm = edge_weight
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result
        if c == 0:
            x = self.propagate(edge_index, x=x, norm=norm)
        for _ in range(c):
            x = 1 * x + 1 * self.propagate(edge_index, x=x, norm=norm)
            x = 0.5 * x
        return x


    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j if norm is not None else x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

class NewEncoder(nn.Module):
    def  __init__(self, nfeat_channels, level_channels, num_subnet, save_memory):
        super(NewEncoder, self).__init__()
        self.fullnet = FullNet(channels=level_channels, num_subnet=num_subnet, nfeat=nfeat_channels, save_memory=save_memory)
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, R = [1,2], final = False):
        if final == False:
            K1 = np.random.randint(R[0], R[1])
            K2 = np.random.randint(R[0], R[1])
            # K3 = np.random.randint(R[0], R[1])
            # K4 = np.random.randint(R[0], R[1])
        if final: 
            K1 = 2
            K2 = 2
            # K3 = 2
            # K4 = 2
        x = self.fullnet(x, edge_index, [K1, K2], final)
        # x = self.fullnet(x, edge_index, [K1, K2, K3, K4], final)
        return x

class NewGRACE(torch.nn.Module):
    def __init__(self, encoder: NewEncoder, tau: float = 0.5):
        super(NewGRACE, self).__init__()
        self.encoder: NewEncoder = encoder
        self.tau: float = tau

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, R = [1,2], final = False) -> torch.Tensor:
        return self.encoder(x, edge_index, R, final)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)

        return torch.mm(z1, z2.t())

    def recLoss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        between_sim = f(self.sim(z1, z2))
        iden = torch.tensor(np.eye(between_sim.shape[0]))
        ref_sim = f(self.sim(z1, z1) - iden)
        ret = -torch.log(between_sim.diag() / ref_sim.sum(1))
        ret = ret.mean()
        return ret
    """   
    def GAELoss(self, z: torch.Tensor):
        act = nn.Sigmoid()
        return self.norm * self.BCE(act(torch.mm(z, z.t())), self.A)
    """

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        between_sim = f(torch.mm(z1, z2.t()))
        return -torch.log(between_sim.diag() / between_sim.sum(1))

    def loss(self, z1: torch.Tensor, z2: torch.Tensor, mean: bool = True, batch_size: Optional[int] = None):
        h1 = z1
        h2 = z2
        if batch_size is None:
            l = self.semi_loss(h1, h2)
        else:
            l = self.semi_loss(h1, h2)

        ret = l
        #ret = l1
        ret = ret.mean() if mean else ret.sum()
        #ret = ret# + 0.1 * self.GAELoss(z1) + 0.1 * self.GAELoss(z2)
        return ret

class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret

class Level(nn.Module):
    def __init__(self, level, channels, first_col, nfeat) -> None:
        super().__init__()
        self.fusion = Fusion(level, channels, first_col, nfeat)
        if level == 0:
            self.block = NewGConv(nfeat, channels[level])
        else:
            self.block = NewGConv(channels[level], channels[level])

    def forward(self, *args):
        x = self.fusion(*args)
        x = self.block(x, args[2], args[3])
        return x

#two levels
class Fusion(nn.Module):
    def __init__(self, level, channels, first_col, nfeat) -> None:
        super().__init__()

        self.level = level
        self.first_col = first_col
        self.down = DownSampleConvnext(channels[level - 1], channels[level]) if level in [1] else nn.Identity()
        if not first_col:
            if level == 0:
                self.up = UpSampleConvnext(channels[level + 1], nfeat)

    def forward(self, *args):
        c_down, c_up, adj, _= args
        if self.first_col:
            x = self.down(c_down)
            return x
        if self.level == 1:
            x = self.down(c_down)
        else:
            x = self.up(c_up) + self.down(c_down)
        return x

#four levels
# class Fusion(nn.Module):
#     def __init__(self, level, channels, first_col, nfeat) -> None:
#         super().__init__()
#         self.level = level
#         self.first_col = first_col
#         self.down = DownSampleConvnext(channels[level - 1], channels[level]) if level in [1, 2, 3] else nn.Identity()
#         if not first_col:
#             if level == 0:
#                 self.up = UpSampleConvnext(channels[level + 1], nfeat)
#             else:
#                 self.up = UpSampleConvnext(channels[level + 1], channels[level]) if level in [1, 2] else nn.Identity()
#
#     def forward(self, *args):
#         c_down, c_up, adj, _= args
#         if self.first_col:
#             x = self.down(c_down)
#             return x
#         if self.level == 3:
#             x = self.down(c_down)
#         else:
#             x = self.up(c_up) + self.down(c_down)
#         return x

# two levels
class SubNet(nn.Module):
    def __init__(self, channels, first_col, save_memory, nfeat) -> None:
        super().__init__()
        shortcut_scale_init_value = 0.5
        self.save_memory = save_memory
        self.alpha0 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[0])),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha1 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[1])),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.level0 = Level(0, channels, first_col, nfeat)

        self.level1 = Level(1, channels, first_col, nfeat)
    def _forward_nonreverse(self, *args):
        x, c0, c1, adj, stack = args
        c0 = (self.alpha0) * c0 + self.level0(x, c1, adj, stack[0])
        c1 = (self.alpha1) * c1 + self.level1(c0, None, adj, stack[1])
        return c0, c1
    def _forward_reverse(self, *args):
        local_funs = [self.level0, self.level1]
        alpha = [self.alpha0, self.alpha1]
        _, c0, c1 = ReverseFunction.apply(
            local_funs, alpha, *args)
        return c0, c1

    def forward(self, *args):
        self._clamp_abs(self.alpha0.data, 1e-3)
        self._clamp_abs(self.alpha1.data, 1e-3)
        if self.save_memory:
            return self._forward_reverse(*args)
        else:
            return self._forward_nonreverse(*args)

    def _clamp_abs(self, data, value):
        with torch.no_grad():
            sign = data.sign()
            data.abs_().clamp_(value)
            data *= sign


#four levels
# class SubNet(nn.Module):
#     def __init__(self, channels, first_col, save_memory, nfeat) -> None:
#         super().__init__()
#         shortcut_scale_init_value = 0.5
#         self.save_memory = save_memory
#         self.alpha0 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[0])),
#                                    requires_grad=True) if shortcut_scale_init_value > 0 else None
#         self.alpha1 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[1])),
#                                    requires_grad=True) if shortcut_scale_init_value > 0 else None
#         self.alpha2 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[2])),
#                                    requires_grad=True) if shortcut_scale_init_value > 0 else None
#         self.alpha3 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[3])),
#                                    requires_grad=True) if shortcut_scale_init_value > 0 else None
#         self.level0 = Level(0, channels, first_col, nfeat)
#
#         self.level1 = Level(1, channels, first_col, nfeat)
#
#         self.level2 = Level(2, channels, first_col, nfeat)
#
#         self.level3 = Level(3, channels, first_col, nfeat)
#     def _forward_nonreverse(self, *args):
#         x, c0, c1, c2, c3, adj, stack = args
#         c0 = (self.alpha0) * c0 + self.level0(x, c1, adj, stack[0])
#         c1 = (self.alpha1) * c1 + self.level1(c0, c2, adj, stack[1])
#         c2 = (self.alpha2) * c2 + self.level2(c1, c3, adj)
#         c3 = (self.alpha3) * c3 + self.level3(c2, None, adj)
#         return c0, c1, c2, c3
#     def _forward_reverse(self, *args):
#         local_funs = [self.level0, self.level1, self.level2, self.level3]
#         alpha = [self.alpha0, self.alpha1, self.alpha2, self.alpha3]
#         _, c0, c1, c2, c3 = ReverseFunction.apply(
#             local_funs, alpha, *args)
#         return c0, c1, c2, c3
#
#     def forward(self, *args):
#         self._clamp_abs(self.alpha0.data, 1e-3)
#         self._clamp_abs(self.alpha1.data, 1e-3)
#         self._clamp_abs(self.alpha2.data, 1e-3)
#         self._clamp_abs(self.alpha3.data, 1e-3)
#         if self.save_memory:
#             return self._forward_reverse(*args)
#         else:
#             return self._forward_nonreverse(*args)
#     def _clamp_abs(self, data, value):
#         with torch.no_grad():
#             sign = data.sign()
#             data.abs_().clamp_(value)
#             data *= sign

class FullNet(nn.Module):
    def __init__(self, channels, num_subnet, nfeat, save_memory) -> None:
        super().__init__()
        self.num_subnet = num_subnet
        self.channels = channels
        self.gcn_pca = NewGConv(nfeat, nfeat)
        for i in range(num_subnet):
            first_col = True if i == 0 else False
            self.add_module(f'subnet{str(i)}', SubNet(
                channels, first_col, save_memory=save_memory, nfeat = nfeat))
    #two levels
    def forward(self, x, adj, stack, final=False):
        c0, c1 = 0, 0
        x = self.gcn_pca(x, adj, 1)
        result = []
        if final:
            get_num = self.num_subnet - 1
        else:
            get_num = np.random.randint(self.num_subnet - 3, self.num_subnet)
        for i in range(self.num_subnet):
            c0, c1 = getattr(self, f'subnet{str(i)}')(x, c0, c1, adj, stack)
            if i == get_num:
                result.append(c0)
                result.append(c1)
        if final:
            x_permuted = torch.cat(result, dim=-1)
        else:
            x_permuted = torch.cat(result, dim=-1)
        return x_permuted

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            trunc_normal_(module.weight, std=.02)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=.02)
            nn.init.constant_(module.bias, 0)

    #four levels
    # def forward(self, x, adj, stack, final=False):
    #     c0, c1, c2, c3 = 0, 0, 0, 0
    #     x = self.gcn_pca(x, adj, 1)
    #     result = []
    #     if final:
    #         get_num = self.num_subnet - 1
    #     else:
    #         get_num = np.random.randint(self.num_subnet - 3, self.num_subnet)
    #     for i in range(self.num_subnet):
    #         c0, c1, c2, c3 = getattr(self, f'subnet{str(i)}')(x, c0, c1, c2, c3, adj, stack)
    #         if i == get_num:
    #             if final:
    #                 result.append(c0)
    #                 result.append(c1)
    #                 result.append(c2)
    #                 result.append(c3)
    #             else:
    #                 result.append(c0)
    #                 result.append(c1)
    #                 result.append(c2)
    #                 result.append(c3)
    #     if final:
    #         x_permuted = torch.cat(result, dim=-1)
    #     else:
    #         x_permuted = torch.cat(result, dim=-1)
    #     return x_permuted
    #
    # def _init_weights(self, module):
    #     if isinstance(module, nn.Conv2d):
    #         trunc_normal_(module.weight, std=.02)
    #         nn.init.constant_(module.bias, 0)
    #     elif isinstance(module, nn.Linear):
    #         trunc_normal_(module.weight, std=.02)
    #         nn.init.constant_(module.bias, 0)