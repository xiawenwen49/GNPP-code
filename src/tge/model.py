import numpy as np
import math
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import MessagePassing, GATConv
from torch_geometric.nn.inits import glorot
from torch_geometric.utils import softmax
from tge.utils import expand_edge_index_timestamp


class TimeEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super(TimeEncoder, self).__init__()
        pass
    def clip_weights(self):
        pass


class HarmonicEncoder(TimeEncoder):
    """ In paper 'Inductive representation learning on temporal graphs'
    """
    def __init__(self, dimension, nnodes):
        super(HarmonicEncoder, self).__init__()
        assert dimension % 2 == 0, 'dimension should be an even'
        self.dimension = dimension
        self.nnodes = nnodes
        self.basis_freq = torch.nn.Parameter((torch.from_numpy(1 / 5 ** np.linspace(1, 9, dimension//2))).float()) # omega_1, ..., omega_d
        self.phase_cos = torch.zeros(dimension//2, dtype=torch.float, requires_grad=False) # no gradient
        self.phase_sin = torch.zeros(dimension//2, dtype=torch.float, requires_grad=False)

        # self.alpha = torch.nn.Parameter( torch.FloatTensor([1]) )
        self.alpha = torch.nn.Parameter( torch.ones(nnodes, nnodes) )
        # self.alpha = torch.FloatTensor([1], requires_grad=False)
    
    def forward(self, ts):
        """ harmonic encoding mapping """
        # ts: [N]
        device = ts.device
        self.basis_freq = self.basis_freq.to(device)
        self.phase_cos = self.phase_cos.to(device)
        self.phase_sin = self.phase_sin.to(device)

        batch_size = ts.size(0)
        ts = ts.view(batch_size, 1)# [N, L, 1]
        # import ipdb; ipdb.set_trace()
        map_ts = ts * self.basis_freq.view(1, -1) # [N, dimension]
        # map_ts += self.phase.view(1, -1)
        harmonic_cos = torch.cos(map_ts + self.phase_cos.view(1, -1))
        harmonic_sin = torch.sin(map_ts + self.phase_sin.view(1, -1))
        harmonic = math.sqrt(1/self.dimension) * torch.cat([harmonic_cos, harmonic_sin], axis=1)
        return harmonic # self.dense(harmonic)
    
    def cos_encoding_mean(self, ts, u, v):
        """ ts should be t_1 - t_2 
        ts should be from the same node pair u-v
        """
        # import ipdb; ipdb.set_trace()
        device = ts.device
        self.phase_cos = self.phase_cos.to(device)

        batch_size = ts.size(0)
        ts = ts.view(batch_size, 1)

        map_ts = ts * self.basis_freq.view(1, -1)
        harmonic_cos = torch.cos(map_ts + self.phase_cos.view(1, -1))

        # mean = torch.mean(harmonic_cos, axis=1)
 
        mean = self.alpha[u][v] * (torch.mean(harmonic_cos, axis=1) + 1) # NOTE: new formula

        return mean

    def sin_divide_omega_mean(self, ts, u, v):
        """ ts should be t_{n^{u,v}} - t_j 
        ts should be from the same node pair u-v
        """
        # import ipdb; ipdb.set_trace()
        device = ts.device
        # self.basis_freq = self.basis_freq.to(device)
        self.phase_sin = self.phase_sin.to(device)

        batch_size = ts.size(0)
        ts = ts.view(batch_size, 1)

        map_ts = ts * self.basis_freq.view(1, -1)
        harmonic_sin = torch.sin(map_ts + self.phase_sin.view(1, -1))
        harmonic_sin_divide_omega = harmonic_sin / (self.basis_freq.view(1, -1) + 1e-6)

        # mean = torch.mean(harmonic_sin_divide_omega, axis=1)
        mean = self.alpha[u][v] * torch.mean(harmonic_sin_divide_omega, axis=1) # NOTE: new formula
        return mean

    def clip_weights(self):
        # self.basis_freq = self.basis_freq.clamp(0, 1)
        # import ipdb; ipdb.set_trace()
        self.alpha.data.clamp_(1e-6, 1e8) # here should use im-place clamp
        # self.alpha = torch.nn.Parameter(self.alpha.clamp(1e-6, 1e8))


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

        self.time_embedding = torch.nn.Embedding.from_pretrained( torch.tensor(self.get_timing_encoding_matrix(rows, dimension)), freeze=True )

    def forward(self, timestamps: Tensor):
        indexes = self.timestamps_to_indexes(timestamps)
        return self.time_embedding(indexes)

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


class TGN_e2n(nn.Module):
    """
    Devised for edge2node graph data.
    Each data sample should be a star graph, each node represents an edge on original graph. 
    Node timestamps are edge(node pair) timestamps on original graph
    """
    def __init__(self):
        self.Atten_self = nn.MultiheadAttention()
        self.Atten_neig = nn.MultiheadAttention()
        self.Linear = nn.Linear()
        pass

    def forward(self, batch, t):
        pass



class TGN(nn.Module):
    """ Temporal graph event model """
    def __init__(self, G, embedding_matrix, time_encoder_args, layers, in_channels, hidden_channels, out_channels, dropout):
        super(TGN, self).__init__()
        self.G = G
        self.edgelist = torch.LongTensor( np.concatenate( [np.array(G.edges(), dtype=np.int).T, np.array(G.edges(), dtype=np.int).T[[1, 0], :]], axis=1 ) )
        self.embedding = torch.nn.Embedding.from_pretrained( torch.FloatTensor(embedding_matrix), freeze=True )
        self.time_encoder_args = time_encoder_args
        # self.time_encoder = HarmonicEncoder(time_encoder_args['dimension'], G.number_of_nodes())
        self.time_encoder = PositionEncoder(time_encoder_args['maxt'], time_encoder_args['rows'], time_encoder_args['dimension'])
        # self.gru = torch.nn.GRU(input_size=time_encoder_args['dimension'], hidden_size=time_encoder_args['dimension'], num_layers=2)
        self.AttenModule = torch.nn.MultiheadAttention(time_encoder_args['dimension'], num_heads=1)
        self.alpha = torch.nn.Embedding(G.number_of_nodes(), 128)
        self.W_H = torch.nn.Parameter(torch.zeros(2*hidden_channels+time_encoder_args['dimension'], 1)) # merger

        self.layers = nn.ModuleList()
        # TODO: add layers
        for i in range(layers):
            # in_channels = self.layers[-1].heads*self.layers[-1].out_channels if i >= 1 else in_channels
            # heads = 8 if i < layers -1 else 1
            # self.layers.append(GATConv(in_channels=in_channels, out_channels=hidden_channels, heads=heads)) # use gat first
            self.layers.append( TGNLayer(in_channels=in_channels, out_channels=hidden_channels, time_encoder=self.time_encoder) )
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

        self.hidden_rep = None
        self.W_S = torch.nn.Parameter(torch.zeros((1, hidden_channels*2)))
        self.W_S_ = nn.Linear(hidden_channels*2, 1, bias=False)
        self.W_E_ = nn.Linear(hidden_channels*2, 1, bias=False)

        self.initialize()
        pass
    
    def clip_time_encoder_weight(self):
        self.time_encoder.clip_weights()
        # self.alpha.weight.clamp_(0, np.inf)


    def initialize(self):
        glorot(self.alpha.weight)
        glorot(self.W_H)
        glorot(self.W_S)
        glorot(self.W_S_.weight)
        glorot(self.W_E_.weight)
        self.clip_time_encoder_weight()

    def forward(self, batch, t):
        """ [u, v], and a subgraph in batch 
        """
        # [u, v], sub_edge_index = batch.nodepair, batch.edge_index
        device = t.device
        assert batch.num_graphs == 1, 'Only support batch_size=1 now'
        sub_nodes = batch.x
        sub_edge_index = batch.edge_index
        u, v = batch.nodepair # torch.LongTensor
        sub_edgearray = batch.edgearray
        
        # extend edge_index and timestamps
        exp_edge_index, exp_ts = expand_edge_index_timestamp(sub_nodes, sub_edgearray, t)

        # model forward
        out = self.embedding(sub_nodes) # sub graph all node embedding
        for layer in self.layers:
            out = layer(out, exp_edge_index, exp_ts, t)
            assert out.shape[0] == len(sub_nodes)
        
        nodepair_rep = out[batch.mapping]
        return nodepair_rep
    
    def reset_hidden_rep(self):
        self.hidden_rep = None
    
    def prepare_hidden_rep(self):
        device = self.embedding.weight.device
        nodes = torch.LongTensor(torch.arange(self.G.number_of_nodes())).to(device)
        self.edgelist = self.edgelist.to(device) # set device
        x = self.embedding(nodes)
        for layer in self.layers:
            x = layer(x, self.edgelist)
            x = self.act(x)
            x = self.dropout(x)
        self.hidden_rep = x

class TGNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, time_encoder, heads=None ):
        super(TGNLayer, self).__init__(aggr='add')
        self.time_encoder = time_encoder
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads # only one head now
        # self.atten = torch.nn.MultiheadAttention()
        self.Q = nn.Linear( in_channels+self.time_encoder.dimension, out_channels, bias=False)
        self.K = nn.Linear( in_channels+self.time_encoder.dimension, out_channels, bias=False)
        self.V = nn.Linear( in_channels+self.time_encoder.dimension, out_channels, bias=False)

        self.initialize()
        pass

    def initialize(self):
        pass
    
    def forward(self, x, edge_index, ts, t):
        """
        Args:
            x: here x should be all node's features, but where sub_nodes indicate is the last layer's features
            edge_index: expanded edge_index
            ts: expanded timestamps
            t: the `current` time
        """
        out = self.propagate(edge_index, x=x, ts=ts, t=t) # will invoke message function
        return out
    
    def message(self, edge_index, x_j: Tensor, x_i: Tensor, ts: Tensor, t: torch.float) -> Tensor:
        # COMPLETED: transform computation
        phi_ts = self.time_encoder(ts)
        phi_t = self.time_encoder( t.repeat(ts.shape[0]) )
        x_j = torch.cat([x_j, phi_ts], axis=1)
        x_i = torch.cat([x_i, phi_t], axis=1)

        x_j = x_j.to(dtype=torch.float32, device=t.device)
        x_i = x_i.to(dtype=torch.float32, device=t.device)

        Q = self.Q(x_i)
        K = self.K(x_j)
        V = self.V(x_j)

        alpha = (Q * K).sum(axis=1)
        alpha = softmax(alpha, edge_index[1]) # group according to x_i

        out = V * alpha.unsqueeze(-1) # TODO: check
        return out


def get_model(G, embedding_matrix, args, logger):
    if args.model == 'TGN':
        model = TGN(G, embedding_matrix, args.time_encoder_args, args.layers, args.in_channels, args.hidden_channels, args.out_channels, args.dropout)
    else:
        raise NotImplementedError("Not implemented now")
    return model