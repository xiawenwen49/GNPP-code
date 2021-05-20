import numpy as np
import math
import networkx as nx
from numpy.lib.function_base import _insert_dispatcher
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MessagePassing
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
    
    def forward_batch(self, ts):
        """ ts: [N, L] -> [N, L, dimension]
        """
        device = ts.device
        self.basis_freq = self.basis_freq.to(device)
        self.phase_cos = self.phase_cos.to(device)
        self.phase_sin = self.phase_sin.to(device)

        ts = ts.unsqueeze(-1)
        # self.basis_freq = self.basis_freq.view((1, 1, self.basis_freq.numel()))
        # self.phase_cos = self.phase_cos.view((1, 1, self.phase_cos.numel()))
        # self.phase_sin = self.phase_sin.view((1, 1, self.phase_sin.numel()))


        map_ts = ts * self.basis_freq.view((1, 1, self.basis_freq.numel()) ) # [N, L, dimension]
        harmonic_cos = torch.cos(map_ts + self.phase_cos.view((1, 1, self.phase_cos.numel())) )
        harmonic_sin = torch.sin(map_ts + self.phase_sin.view((1, 1, self.phase_sin.numel())) )
        harmonic = math.sqrt(1/self.dimension) * torch.cat([harmonic_cos, harmonic_sin], axis=-1)
        return harmonic
    
    def fit(self, X: np.array, y: np.array):
        X = torch.tensor(X)
        y = torch.tensor(y)
        X0 = torch.zeros_like(X)

        optimizer = torch.optim.Adam( filter(lambda p: p.requires_grad, self.parameters()), lr=1e-2)
        for i in range(1000):
            # import ipdb; ipdb.set_trace()
            mse = ( (self(X) * self(X0)).sum(axis=1) - y).square().mean()
            optimizer.zero_grad()
            mse.backward()
            optimizer.step()
            if (i+1) % 2 == 0:
                print('epoch: {}, loss: {}, rmse: {}'.format(i, mse.cpu().item(), mse.cpu().sqrt().item()))

    def predict(self, X: np.array):
        X = torch.tensor(X)
        X0 = torch.zeros_like(X)
        with torch.no_grad():
            pred = (self(X) * self(X0)).sum(axis=1)
            # import ipdb; ipdb.set_trace()
            return pred.cpu().numpy()
    
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

class FourierEncoder(TimeEncoder):
    """ Not the `Fourier encoder` in `Deep Fourier Kernel for Self-attentive Point Processes`.
    But only output a scalar. 
    """
    def __init__(self, maxt=1000, dimension: int=128, **kwargs):
        super(FourierEncoder, self).__init__()
        self.maxt = maxt
        self.dimension = dimension
        self.alpha0 = torch.nn.Parameter(torch.tensor([0.1]))
        self.alpha = torch.nn.Parameter(torch.zeros((1, dimension)))
        self.beta = torch.nn.Parameter(torch.zeros((1, dimension)))
        self.n_omega0 = (2*np.pi/maxt * torch.arange(1, dimension+1) ).reshape((1, dimension))

        pass

    def forward(self, timestamps):
        """ Fourier approximation output (scalar) """
        device = timestamps[0].device
        self.alpha0 = self.alpha0.to(device)
        self.alpha = self.alpha.to(device)
        self.beta = self.beta.to(device)
        self.n_omega0 = self.n_omega0.to(device)

        timestamps = timestamps.reshape((-1, 1))
        phase = timestamps * self.n_omega0
        cos = torch.cos(phase) * self.alpha
        sin = torch.sin(phase) * self.beta
        res = (cos + sin).sum(axis=1) + 0.5 * self.alpha0
        # res = soft_plus(0.1, res)
        return res
    
    # def forward(self, timestamps):
    #     """ constant fourier embedding """
    #     device = timestamps[0].device
    #     self.alpha0 = self.alpha0.to(device)
    #     self.alpha = self.alpha.to(device)
    #     self.beta = self.beta.to(device)
    #     self.n_omega0 = self.n_omega0.to(device)

    #     timestamps = timestamps.reshape((-1, 1))
    #     phase = timestamps * self.n_omega0
    #     cos = torch.cos(phase)
    #     sin = torch.sin(phase)
    #     output = torch.cat([cos, sin], axis=1)
    #     return output





    def fit(self, X: np.array, y: np.array):
        X = torch.tensor(X)
        y = torch.tensor(y)
        optimizer = torch.optim.SGD( filter(lambda p: p.requires_grad, self.parameters()), lr=1e-2)
        for i in range(500):
            mse = (self(X) - y).square().mean()
            optimizer.zero_grad()
            mse.backward()
            optimizer.step()
            if (i+1) % 2 == 0:
                print('epoch: {}, loss: {}, rmse: {}'.format(i, mse.cpu().item(), mse.cpu().sqrt().item()))

    def predict(self, X: np.array):
        X = torch.tensor(X)
        with torch.no_grad():
            pred = self(X)
            return pred.cpu().numpy()


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

class EncoderAtten(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super(EncoderAtten, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.time_encoders = nn.ModuleList([ HarmonicEncoder(embed_dim) for i in range(num_heads) ])
        # self.attens = nn.ModuleList([nn.MultiheadAttention(embed_dim, num_heads=1, dropout=dropout) for i in range(num_heads)])
        self.sm = nn.Softmax(dim=1)
        # self.linear = nn.Linear((embed_dim)*num_heads, embed_dim)
        self.V = nn.Parameter(torch.rand(embed_dim, embed_dim))

        

        # self._reset_parameters()
    
    # def _reset_parameters(self):
    #     for i in range(self.num_heads): # actually it is invalid!
    #         self.attens[i].q_proj_weight = nn.Parameter(torch.eye(self.embed_dim), requires_grad=False)
    #         self.attens[i].k_proj_weight = nn.Parameter(torch.eye(self.embed_dim), requires_grad=False)


    # def forward(self, query, key, value, key_padding_mask=None, need_weights=False, attn_mask=None):
    def forward(self, t, e_nodes_ts, key_padding_mask=None, need_weights=False, attn_mask=None):
        """ Here the query, key, value should all be timestamps.
        N: batch size
        L: target sequence length
        S: source sequence length
        E: embedding dimision of one item
        Args:
            t: [N,],
            e_nodes_ts: [L,], all timestamps (self & neighbor)
            key_padding_mask: [N, L]. self_mask or neighbor_mask
        
        Output:
            attn_output: [N, E].
            attn_output_weights: [N, L].
        """
        atten_outputs = []
        atten_weights = []
        N = t.numel()
        L = e_nodes_ts.numel()
        E = self.embed_dim
        for i in range(self.num_heads):
            phi_t = self.time_encoders[i](t).reshape((N, -1))
            phi_ts = self.time_encoders[i](e_nodes_ts).reshape((L, -1))

            inner_prod = phi_t @ phi_ts.T

            # import ipdb; ipdb.set_trace()
            inner_prod[key_padding_mask] = float('-inf') # TODO: for check            

            atten_weight = self.sm(inner_prod)

            phi_ts_mat = phi_ts.reshape((1, L, -1)).repeat((N, 1, 1)) # (N, L, E) prepare candidate time embeddings
            phi_ts_mat = phi_ts_mat @ self.V # value embedding space
            t_rep = phi_ts_mat * atten_weight.reshape((N, L, 1)).repeat(1, 1, E) # (N, L, E), weight embeddings
            t_rep = t_rep.sum(axis=1) # (N, E), sum up embeddings

            atten_outputs.append(t_rep)
            atten_weights.append(atten_weight)
        
        atten_outputs = torch.cat(atten_outputs, axis=1)
        return atten_outputs, atten_weights


class EncoderInnerProduct(nn.Module):
    def __init__(self, maxt, embed_dim, num_heads, dropout):
        super(EncoderInnerProduct, self).__init__()
        self.maxt = maxt
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.time_encoders = nn.ModuleList([ FourierEncoder(maxt, embed_dim) for i in range(num_heads) ])
        # self.coeffs = nn.Parameter(0.1*torch.ones((1, num_heads)))
        # self.mlps = nn.ModuleList([ nn.Sequential(nn.Linear(1, 512), nn.LeakyReLU(), nn.Linear(512, 1) ) for i in range(num_heads) ])



    def forward(self, t, e_nodes_ts, key_padding_mask=None, need_weights=False, attn_mask=None):
        # key_padding_mask: True will be ignored
        atten_outputs = []
        atten_weights = []
        for i in range(self.num_heads):
            # Fourier encoder
            t_exp = t.reshape((-1, 1)).repeat( (1, e_nodes_ts.numel())).reshape((1, -1))
            ts_exp = e_nodes_ts.reshape((1, -1)).repeat((1, t.numel())).reshape((1, -1))
            interval = t_exp - ts_exp
            output = self.time_encoders[i](interval)
            output = output.reshape((t.numel(), e_nodes_ts.numel()))
            

            # inner product
            # phi_t = self.time_encoders[i](t).reshape((t.numel(), -1))
            # phi_ts_mat = self.time_encoders[i](e_nodes_ts).reshape((e_nodes_ts.numel(), -1))
            # output = phi_t @ phi_ts_mat.T


            output = output * (~key_padding_mask).type(output.dtype)
            output = output.sum(axis=1, keepdim=True)
            
            atten_outputs.append(output)
            atten_weights.append(output)
        
        atten_outputs = torch.cat(atten_outputs, axis=1)
        # atten_outputs = atten_outputs * self.coeffs
        atten_outputs = atten_outputs.sum(axis=1, keepdim=True)

        return atten_outputs, atten_weights
    
    def forward_mlp(self, t, e_nodes_ts, key_padding_mask=None, need_weights=False, attn_mask=None):
        atten_outputs = []
        atten_weights = []
        for i in range(self.num_heads):
            t_exp = t.reshape((-1, 1)).repeat((1, e_nodes_ts.numel())).reshape((-1, 1))
            e_nodes_ts_exp = e_nodes_ts.reshape((-1, 1)).repeat((t.numel(), 1))
            t_interval = t_exp - e_nodes_ts_exp
            
            output = self.mlps[i](t_interval)
            mask = (~key_padding_mask).reshape((-1, 1)).type(output.dtype)
            output = (output * mask).reshape((t.numel(), e_nodes_ts.numel())).sum(axis=1, keepdim=True)

            atten_outputs.append(output)
            atten_weights.append(output)
        
        atten_outputs = torch.cat(atten_outputs, axis=1)
        atten_outputs = atten_outputs * self.coeffs
        atten_outputs = atten_outputs.sum(axis=1, keepdim=True)
        
        return atten_outputs, atten_weights


    

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
        # elif time_encoder_args['type'] == 'fe':
        #     self.time_encoder = FourierEncoder(time_encoder_args['maxt']*1.5, time_encoder_args['dimension']//2) # output scalar
        
    
        self.Atten_self = nn.MultiheadAttention(embed_dim=time_encoder_args['dimension'], num_heads=num_heads, dropout=dropout)
        self.Atten_neig = nn.MultiheadAttention(embed_dim=time_encoder_args['dimension'], num_heads=num_heads, dropout=dropout)
        
        # self.Atten_self = EncoderAtten(embed_dim=time_encoder_args['dimension'], num_heads=num_heads, dropout=dropout)
        # self.Atten_neig = EncoderAtten(embed_dim=time_encoder_args['dimension'], num_heads=num_heads, dropout=dropout)

        # self.Atten_self = EncoderInnerProduct(maxt=50, embed_dim=time_encoder_args['dimension'], num_heads=num_heads, dropout=dropout)
        # self.Atten_neig = EncoderInnerProduct(maxt=50, embed_dim=time_encoder_args['dimension'], num_heads=num_heads, dropout=dropout)
        
        self.Linear_lambda = nn.Linear(time_encoder_args['dimension']*2, 1, bias=False)

        if isinstance(self.Atten_self, EncoderInnerProduct):
            self.Linear_pred = nn.Linear(1, 1)
        else:
            self.Linear_pred = nn.Linear(time_encoder_args['dimension']*2, 1)

        self.W_H = torch.nn.Parameter(torch.zeros(time_encoder_args['dimension'], 1))
        # self.phi = torch.nn.Parameter(torch.tensor([0.1]))
        # self.phi = 10 # 10 synthetic_neg
        self.phi = 0.1 # synthetic_pos

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

        
        # build mask
        ts_mat = e_nodes_ts.reshape((1, -1)).repeat((t.numel(), 1))
        key_padding_mask = ts_mat > t.reshape((-1, 1)) # True will be ignored

        self_mask = (e_nodes_exp != e_node_target).reshape(1, -1).repeat((t.numel(), 1))
        neig_mask = (e_nodes_exp == e_node_target).reshape(1, -1).repeat((t.numel(), 1))
        key_padding_mask_self = key_padding_mask + self_mask
        key_padding_mask_neig = key_padding_mask + neig_mask

        # MultiHeadAtten setting
        if isinstance(self.Atten_self, nn.MultiheadAttention):
            phi_t = self.time_encoder(t).reshape((1, t.numel(), -1)) # query [L, N, E], i.e., [1, t.numel(), encoding_dim)
            phi_ts_mat = self.time_encoder(e_nodes_ts).reshape((e_nodes_ts.numel(), 1, -1)).repeat((1, t.numel(), 1)) # key/value [S, N, E], i.e., [e_nodes_ts.numel(), t.numel(), encoding_dim]
            self_atten_output, self_weights = self.Atten_self(phi_t, phi_ts_mat, phi_ts_mat, key_padding_mask=key_padding_mask_self)
            
            if self.kwargs['with_neig']:
                neig_atten_output, neig_weights = self.Atten_neig(phi_t, phi_ts_mat, phi_ts_mat, key_padding_mask=key_padding_mask_neig)
            else:
                neig_atten_output = torch.zeros_like(self_atten_output)
            
            atten_output = torch.cat([self_atten_output, neig_atten_output], axis=2).squeeze(0)
            lambdav = self.Linear_lambda(atten_output)

        elif isinstance(self.Atten_self, EncoderAtten): # EncoderAtten setting
            self_atten_output, self_weights = self.Atten_self(t, e_nodes_ts, key_padding_mask_self)
            neig_atten_output, neig_weights = self.Atten_neig(t, e_nodes_ts, key_padding_mask_neig)
            atten_output = torch.cat([self_atten_output, neig_atten_output], axis=1)
            lambdav = self.Linear_lambda(atten_output)

        elif isinstance(self.Atten_self, EncoderInnerProduct): # EncoderInnerProduct setting
            self_atten_output, self_weights = self.Atten_self(t, e_nodes_ts, key_padding_mask_self)
            neig_atten_output, neig_weights = self.Atten_neig(t, e_nodes_ts, key_padding_mask_neig)
            atten_output = self_atten_output + neig_atten_output
            lambdav = atten_output
        



        # import ipdb; ipdb.set_trace()
        lambdav = soft_plus(self.phi, lambdav)
        if torch.isnan(lambdav).any() or torch.isinf(lambdav).any():
            import ipdb; ipdb.set_trace()
        return lambdav, atten_output        




def soft_plus(phi, x):
    x = x * phi
    x[x>20] = 20
    res = 1.0/phi * torch.log( 1 + torch.exp(x) )
    # res = torch.where(x/phi < 20, 1e-6 + phi * torch.log1p( torch.exp(x/phi) ), x ) # 1e-6 is important, to make sure lambda > 0
    return res

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
    elif args.model == 'GNPP':
        model = GNPP(G, args.time_encoder_args, args.num_heads, args.dropout, with_neig=args.with_neig)
    elif args.model in ['GAT', 'GraphSAGE']:
        from xww.utils.models import GNNModel
        model = GNNModel(args.model, args.layers, args.in_channels, args.hidden_channels, out_features=1, set_indice_size=1, dropout=args.dropout)
    else:
        raise NotImplementedError("Not implemented now")
    return model



