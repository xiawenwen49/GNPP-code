import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.inits import glorot


class TimeEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super(TimeEncoder, self).__init__()
        pass
    def clip_weights(self):
        pass


class HarmonicEncoder(TimeEncoder):
    """ In paper 'Inductive representation learning on temporal graphs'
    """
    def __init__(self, dimension):
        super(HarmonicEncoder, self).__init__()
        assert dimension % 2 == 0, 'dimension should be an even'
        self.dimension = dimension
        self.basis_freq = torch.nn.Parameter((torch.from_numpy(1 / 5 ** np.linspace(1, 9, dimension//2))).float()) # omega_1, ..., omega_d
        self.phase_cos = torch.zeros(dimension//2, dtype=torch.float, requires_grad=False) # no gradient
        self.phase_sin = torch.zeros(dimension//2, dtype=torch.float, requires_grad=False)
    
    def forward(self, ts):
        """ harmonic encoding mapping """
        # ts shape: maybe [N]
        device = ts.device
        self.basis_freq = self.basis_freq.to(device)
        self.phase_cos = self.phase_cos.to(device)
        self.phase_sin = self.phase_sin.to(device)

        batch_size = ts.size(0)
        ts = ts.view(batch_size, 1)# [N, 1]
        map_ts = ts * self.basis_freq.view(1, -1) # [N, dimension]
        harmonic_cos = torch.cos(map_ts + self.phase_cos.view(1, -1))
        harmonic_sin = torch.sin(map_ts + self.phase_sin.view(1, -1))
        harmonic = math.sqrt(1/(self.dimension//2)) * torch.cat([harmonic_cos, harmonic_sin], axis=1)
        return harmonic


class PositionEncoder(TimeEncoder):
    """
    discrete + position encoding
    """
    def __init__(self, maxt, rows: int = 50000, dimension: int = 128, **kwargs):
        super(PositionEncoder, self).__init__()
        self.maxt = maxt
        self.rows = rows
        self.deltat = maxt / rows
        self.dimension = dimension
        self.time_embedding = torch.tensor(self.get_timing_encoding_matrix(rows, dimension), dtype=torch.float32)

    def forward(self, timestamps: Tensor):
        if self.time_embedding.device != timestamps.device:
            self.time_embedding = self.time_embedding.to(timestamps.device)
        indexes = self.timestamps_to_indexes(timestamps)
        indexes = indexes.reshape((-1))
        return self.time_embedding[indexes]

    def timestamps_to_indexes(self, timestamps: Tensor):
        """ convert float tensor timestamps to long tensor indexes """
        indexes = timestamps // self.deltat
        indexes = torch.clamp(indexes, 0, self.rows - 1)
        if indexes.is_cuda:
            indexes = indexes.type(torch.cuda.LongTensor)
        else:
            indexes = indexes.type(torch.LongTensor)
        return indexes

    def get_timing_encoding_matrix(self, length, dimension, min_timescale=1.0, max_timescale=1.0e4, start_index=0):
        """ https://kazemnejad.com/blog/transformer_architecture_positional_encoding/ """
        assert dimension % 2 == 0, 'TAT time encoding dimension must be even'
        T = np.arange(length).reshape((length, 1))
        W_inv_log = np.arange(dimension // 2) * 2 / (dimension - 2) * np.log(
            max_timescale)  # (dimension-2)就等价与tensorflow的实现。
        W = 1 / np.exp(W_inv_log)
        position_encodings = T @ W.reshape((1, dimension // 2))
        position_encodings = np.concatenate([np.sin(position_encodings), np.cos(position_encodings)], axis=1)
        return position_encodings


class GNPP(nn.Module):
    """
    `GNPP` model
    Devised for edge2node graph data.
    Each data sample should be a star graph, each node represents an edge on original graph. 
    Node timestamps are edge(node pair) timestamps on original graph
    """
    def __init__(self, G, time_encoder_args, num_heads, dropout, **kwargs):
        super(GNPP, self).__init__()
        self.G = G
        self.time_encoder_args = time_encoder_args
        self.kwargs = kwargs

        if time_encoder_args['type'] == 'pe':
            self.time_encoder = PositionEncoder(time_encoder_args['maxt']*1.1, time_encoder_args['rows'], time_encoder_args['dimension']) # output vector
        elif time_encoder_args['type'] == 'he':
            self.time_encoder = HarmonicEncoder(time_encoder_args['dimension'])
        
    
        self.Atten_self = nn.MultiheadAttention(embed_dim=time_encoder_args['dimension'], num_heads=num_heads, dropout=dropout)
        self.Atten_neig = nn.MultiheadAttention(embed_dim=time_encoder_args['dimension'], num_heads=num_heads, dropout=dropout)
        self.Linear_lambda = nn.Linear(time_encoder_args['dimension']*2, 1, bias=False)
        self.Linear_pred = nn.Linear(time_encoder_args['dimension']*2, 1)
        self.W_H = torch.nn.Parameter(torch.zeros(time_encoder_args['dimension'], 1))
        self.phi = 0.1
        self.initialize()
    
    def initialize(self,):
        glorot(self.W_H)

    def forward(self, batch, t):
        e_nodes_exp = batch.e_nodes_exp
        e_nodes_ts = batch.e_nodes_ts
        e_node_target = batch.e_node_target 

        # make sure neig_mask, and self_mask is not empty
        e_nodes_ts = torch.cat([torch.tensor([0, 0], dtype=e_nodes_ts.dtype, device=e_nodes_ts.device), e_nodes_ts], axis=0 ) 
        e_nodes_exp = torch.cat([torch.tensor([-1], dtype=e_nodes_exp.dtype, device=e_nodes_exp.device), e_node_target, e_nodes_exp], axis=0 )

        
        # build self_mask and neighbor_mask
        ts_mat = e_nodes_ts.reshape((1, -1)).repeat((t.numel(), 1))
        key_padding_mask = ts_mat > t.reshape((-1, 1)) # True will be ignored

        self_mask = (e_nodes_exp != e_node_target).reshape(1, -1).repeat((t.numel(), 1))
        neig_mask = (e_nodes_exp == e_node_target).reshape(1, -1).repeat((t.numel(), 1))
        key_padding_mask_self = key_padding_mask + self_mask
        key_padding_mask_neig = key_padding_mask + neig_mask

        assert isinstance(self.Atten_self, nn.MultiheadAttention)

        phi_t = self.time_encoder(t).reshape((1, t.numel(), -1)) # query [L, N, E], i.e., [1, t.numel(), encoding_dim)
        phi_ts_mat = self.time_encoder(e_nodes_ts).reshape((e_nodes_ts.numel(), 1, -1)).repeat((1, t.numel(), 1)) # key/value [S, N, E], i.e., [e_nodes_ts.numel(), t.numel(), encoding_dim]
        # self information
        self_atten_output, self_weights = self.Atten_self(phi_t, phi_ts_mat, phi_ts_mat, key_padding_mask=key_padding_mask_self)
        # neighbor information
        if self.kwargs['with_neig']:
            neig_atten_output, neig_weights = self.Atten_neig(phi_t, phi_ts_mat, phi_ts_mat, key_padding_mask=key_padding_mask_neig)
        else:
            neig_atten_output = torch.zeros_like(self_atten_output)
        
        atten_output = torch.cat([self_atten_output, neig_atten_output], axis=2).squeeze(0)
        lambdav = self.Linear_lambda(atten_output)

        # import ipdb; ipdb.set_trace()
        lambdav = soft_plus(self.phi, lambdav)
        return lambdav, atten_output        


def soft_plus(phi, x):
    x = x * phi
    x[x>20] = 20
    res = 1.0/phi * torch.log( 1 + torch.exp(x) )
    return res


def get_model(G, embedding_matrix, args, logger):
    if args.model == 'GNPP':
        model = GNPP(G, args.time_encoder_args, args.num_heads, args.dropout, with_neig=args.with_neig)
    elif args.model in ['GAT', 'GraphSAGE']:
        from xww.utils.models import GNNModel
        model = GNNModel(args.model, args.layers, args.in_channels, args.hidden_channels, out_features=1, set_indice_size=1, dropout=args.dropout)
    else:
        raise NotImplementedError("Not implemented now")
    return model



