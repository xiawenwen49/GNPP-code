import numpy as np
import math
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, GATConv
from torch_geometric.nn.inits import glorot

class TimeEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super(TimeEncoder, self).__init__()
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

        self.alpha = torch.nn.Parameter( torch.FloatTensor([1]) )
        # self.alpha = torch.FloatTensor([1], requires_grad=False)
    
    def forward(self, ts):
        """ harmonic encoding mapping """
        # ts: [N]
        device = ts.device
        self.basis_freq = self.basis_freq.to(device)
        self.phase_cos = self.phase_cos.to(device)
        self.phase_sin = self.phase_sin.to(device)

        batch_size = ts.size(0)
        # seq_len = 1
                
        ts = ts.view(batch_size, 1)# [N, L, 1]
        # import ipdb; ipdb.set_trace()

        map_ts = ts * self.basis_freq.view(1, -1) # [N, dimension]
        # map_ts += self.phase.view(1, -1)
        
        harmonic_cos = torch.cos(map_ts + self.phase_cos.view(1, -1))
        harmonic_sin = torch.sin(map_ts + self.phase_sin.view(1, -1))
        harmonic = math.sqrt(1/self.dimension) * torch.cat([harmonic_cos, harmonic_sin], axis=1)
        return harmonic # self.dense(harmonic)
    
    def cos_encoding_mean(self, ts):
        """ ts should be t_1 - t_2 """
        # import ipdb; ipdb.set_trace()
        device = ts.device
        self.phase_cos = self.phase_cos.to(device)

        batch_size = ts.size(0)
        ts = ts.view(batch_size, 1)

        map_ts = ts * self.basis_freq.view(1, -1)
        harmonic_cos = torch.cos(map_ts + self.phase_cos.view(1, -1))

        # mean = torch.mean(harmonic_cos, axis=1)
 
        mean = self.alpha * (torch.mean(harmonic_cos, axis=1) + 1) # NOTE: new formula

        return mean

    def sin_divide_omega_mean(self, ts):
        """ ts should be t_{n^{u,v}} - t_j """
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
        mean = self.alpha * torch.mean(harmonic_sin_divide_omega, axis=1) # NOTE: new formula
        return mean

    def clip_weights(self):
        # self.basis_freq = self.basis_freq.clamp(0, 1)
        # import ipdb; ipdb.set_trace()
        self.alpha.data.clamp_(1e-6, 1e8) # here should use im-place clamp
        # self.alpha = torch.nn.Parameter(self.alpha.clamp(1e-6, 1e8))
    
    # def sin_encoding_mean(self, ts):
    #     """ ts should be t_1 - t_2 """
    #     device = ts.device
    #     self.basis_freq = self.basis_freq.to(device)
    #     self.phase_sin = self.phase_sin.to(device)

    #     batch_size = ts.size(0)
    #     ts = ts.view(batch_size, 1)

    #     map_ts = ts * self.basis_freq.view(1, -1)
    #     harmonic_sin = torch.sin(map_ts + self.phase_sin.view(1, -1))

    #     mean = torch.mean(harmonic_sin, axis=1)

    #     return mean

        


class TGN(nn.Module):
    """ Temporal graph event model """
    def __init__(self, G, embedding_matrix, time_encoder_dimension, layers, in_channels, hidden_channels, out_channels, dropout):
        super(TGN, self).__init__()
        self.G = G
        self.edgelist = torch.LongTensor( np.concatenate( [np.array(G.edges(), dtype=np.int).T, np.array(G.edges(), dtype=np.int).T[[1, 0], :]], axis=1 ) )
        self.embedding = torch.nn.Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1])
        self.embedding.weight = torch.nn.Parameter(torch.FloatTensor(embedding_matrix), requires_grad=True) # NOTE: test
        self.time_encoder_dimension = time_encoder_dimension
        self.time_encoder = HarmonicEncoder(time_encoder_dimension)

        
        self.layers = nn.ModuleList()
        for i in range(layers):
            in_channels = self.layers[-1].heads*self.layers[-1].out_channels if i >= 1 else in_channels
            heads = 8 if i < layers -1 else 1
            self.layers.append(GATConv(in_channels=in_channels, out_channels=hidden_channels, heads=heads)) # use gat first
            # self.layers.append( TGNLayer() )
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


    def initialize(self):
        glorot(self.W_S)
        glorot(self.W_S_.weight)
        glorot(self.W_E_.weight)

    def forward(self, batch):
        uv, t = batch
        u, v = uv[0] # assume default batch_size = 1
        # t = t[0]
        # import ipdb; ipdb.set_trace()
        device = u.device
        if self.hidden_rep is not None:
            out = self.hidden_rep[u], self.hidden_rep[v]
        else:
            nodes = torch.LongTensor(torch.arange(self.G.number_of_nodes())).to(device)
            self.edgelist = self.edgelist.to(device) # set device
            x = self.embedding(nodes)
            for layer in self.layers:
                x = layer(x, self.edgelist)
                x = self.act(x)
                x = self.dropout(x)
            self.hidden_rep = x
            out = self.hidden_rep[u], self.hidden_rep[v]

        return out
    
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
    def __init__(self):
        self.initialize()
        pass

    def initialize(self):
        pass


def get_model(G, embedding_matrix, args, logger):
    if args.model == 'TGN':
        model = TGN(G, embedding_matrix, args.time_encoder_dimension, args.layers, args.in_channels, args.hidden_channels, args.out_channels, args.dropout)
    else:
        raise NotImplementedError("Not implemented now")
    return model