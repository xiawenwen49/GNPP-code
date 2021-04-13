import numpy as np
import math
import networkx as nx
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
    def __init__(self, dimension, nnodes=None):
        super(HarmonicEncoder, self).__init__()
        assert dimension % 2 == 0, 'dimension should be an even'
        self.dimension = dimension
        # self.nnodes = nnodes
        self.basis_freq = torch.nn.Parameter((torch.from_numpy(1 / 5 ** np.linspace(1, 9, dimension//2))).float()) # omega_1, ..., omega_d
        self.phase_cos = torch.zeros(dimension//2, dtype=torch.float, requires_grad=False) # no gradient
        self.phase_sin = torch.zeros(dimension//2, dtype=torch.float, requires_grad=False)

        # self.alpha = torch.nn.Parameter( torch.FloatTensor([1]) )
        # self.alpha = torch.nn.Parameter( torch.ones(nnodes, nnodes) )
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
        return harmonic
    
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
    def __init__(self, maxt=1000, dimension: int=128, **kwargs):
        super(FourierEncoder, self).__init__()
        self.maxt = maxt
        self.dimension = dimension
        self.alpha0 = torch.nn.Parameter(torch.tensor([0.1]))
        self.alpha = torch.nn.Parameter(torch.zeros((1, dimension)))
        self.beta = torch.nn.Parameter(torch.zeros((1, dimension)))
        self.n_omega0 = (2*np.pi/maxt * torch.arange(1, dimension+1) ).reshape((1, dimension))

        pass

    # def forward(self, timestamps):
    #     """ Fourier approximation output (scalar) """
    #     device = timestamps[0].device
    #     self.alpha0 = self.alpha0.to(device)
    #     self.alpha = self.alpha.to(device)
    #     self.beta = self.beta.to(device)
    #     self.n_omega0 = self.n_omega0.to(device)

    #     timestamps = timestamps.reshape((-1, 1))
    #     phase = timestamps * self.n_omega0
    #     cos = torch.cos(phase) * self.alpha
    #     sin = torch.sin(phase) * self.beta
    #     res = (cos + sin).sum(axis=1) + 0.5 * self.alpha0
    #     res = soft_plus(0.1, res)
    #     return res
    
    def forward(self, timestamps):
        """ embedding """
        device = timestamps[0].device
        self.alpha0 = self.alpha0.to(device)
        self.alpha = self.alpha.to(device)
        self.beta = self.beta.to(device)
        self.n_omega0 = self.n_omega0.to(device)

        timestamps = timestamps.reshape((-1, 1))
        phase = timestamps * self.n_omega0
        cos = torch.cos(phase)
        sin = torch.sin(phase)
        output = torch.cat([cos, sin], axis=1)
        return output





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

        # self.time_embedding = torch.nn.Embedding.from_pretrained( torch.tensor(self.get_timing_encoding_matrix(rows, dimension), dtype=torch.float32), freeze=True )
        self.time_embedding = torch.tensor(self.get_timing_encoding_matrix(rows, dimension), dtype=torch.float32)

    def forward(self, timestamps: Tensor):
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


class TGN_e2n(nn.Module):
    """
    `GNPP`
    Devised for edge2node graph data.
    Each data sample should be a star graph, each node represents an edge on original graph. 
    Node timestamps are edge(node pair) timestamps on original graph
    """
    def __init__(self, G, time_encoder_args, hidden_channels):
        super(TGN_e2n, self).__init__()
        self.G = G
        self.G_e2n = nx.line_graph(G)
        self.mapping = dict(zip(self.G_e2n.nodes, np.arange(self.G_e2n.number_of_nodes())))
        # self.time_encoder = PositionEncoder(time_encoder_args['maxt']*1.5, time_encoder_args['rows'], time_encoder_args['dimension'])
        # self.time_encoder = FourierEncoder(time_encoder_args['maxt']*1.5, time_encoder_args['dimension']//2)
        self.time_encoder = HarmonicEncoder(time_encoder_args['dimension'])
        
        

        self.Atten_self = nn.MultiheadAttention(embed_dim=time_encoder_args['dimension'], num_heads=1)
        self.Atten_neig = nn.MultiheadAttention(embed_dim=time_encoder_args['dimension'], num_heads=1)
        
        
        self.Linear = nn.Linear(time_encoder_args['dimension']*2, 1)
        self.Linear_pred = nn.Linear(time_encoder_args['dimension']*2, 1)
        # self.Linear_pred = nn.Sequential(nn.Linear(time_encoder_args['dimension']*2, 128), nn.LeakyReLU(), nn.Linear(128, 1))

        self.W_H = torch.nn.Parameter(torch.zeros(time_encoder_args['dimension'], 1))
        self.phi = torch.nn.Parameter(torch.tensor([1.0]))


        self.miu = torch.nn.Parameter(torch.zeros((1, 1)))
        self.linear_weights = torch.nn.Parameter(torch.zeros((1, time_encoder_args['dimension'])))

        self.initialize()
    
    def initialize(self,):
        torch.nn.init.uniform_(self.miu)
        torch.nn.init.normal_(self.linear_weights)
        glorot(self.W_H)

# TODO: multi-head ?
    def forward(self, batch, t):
        e_nodes_exp = batch.e_nodes_exp
        e_nodes_ts = batch.e_nodes_ts
        e_node_target = batch.e_node_target # because of the SPECIFIC star graph, this e_node_target is not so important


        # phi_t = self.time_encoder(t).reshape((1, 1, -1))
        # observe_mask = e_nodes_ts < t
        # observed_ts = e_nodes_ts[observe_mask]
        # phi_ts = self.time_encoder(observed_ts).reshape((observe_mask.sum().cpu().item(), 1, -1))
        # phi_tall = torch.cat([phi_ts, phi_t], axis=0)
        # atten_output, atten_weight = self.Atten_neig(phi_t, phi_tall, phi_tall)
        # out = self.Linear(atten_output)
        # out = soft_plus(1, out)
        # return out


        # self_atten concat neighbor_atten.      Assume: for query, (L=1, N=t.numel(), E). for key, (S=len(e_nodes_ts), N=t.numel, E)
        phi_t = self.time_encoder(t).reshape((1, 1, -1))
        phi_t = phi_t.reshape((1, t.numel(), -1))
        
        # make sure neig_mask, and self_mask is not empty
        e_nodes_ts = torch.cat([torch.tensor([0, 0], dtype=e_nodes_ts.dtype, device=e_nodes_ts.device), e_nodes_ts], axis=0 ) 
        e_nodes_exp = torch.cat([torch.tensor([-1], dtype=e_nodes_exp.dtype, device=e_nodes_exp.device), e_node_target, e_nodes_exp], axis=0 )

        ts_mat = e_nodes_ts.reshape((1, -1)).repeat((t.numel(), 1))
        key_padding_mask = ts_mat > t.reshape((-1, 1)) # True will be ignored

        self_mask = (e_nodes_exp != e_node_target).reshape(1, -1).repeat((t.numel(), 1))
        neig_mask = (e_nodes_exp == e_node_target).reshape(1, -1).repeat((t.numel(), 1))
        key_padding_mask_self = key_padding_mask + self_mask # # True positions will be ignored
        key_padding_mask_neig = key_padding_mask + neig_mask

        phi_ts_mat = self.time_encoder(e_nodes_ts).reshape((e_nodes_ts.numel(), 1, -1)).repeat((1, t.numel(), 1)) # [S, N, E]
        self_atten_output, self_weights = self.Atten_self(phi_t, phi_ts_mat, phi_ts_mat, key_padding_mask=key_padding_mask_self)
        neig_atten_output, neig_weights = self.Atten_neig(phi_t, phi_ts_mat, phi_ts_mat, key_padding_mask=key_padding_mask_neig)

        atten_output = torch.cat([self_atten_output, neig_atten_output], axis=2).squeeze(0)
        # atten_output = self_atten_output.squeeze(0)

        lambdav = self.Linear(atten_output)
        lambdav = soft_plus(self.phi, lambdav)

        # import ipdb; ipdb.set_trace()

        if torch.isnan(lambdav).any() or torch.isinf(lambdav).any():
            import ipdb; ipdb.set_trace()


        return lambdav, atten_output


        
        # observe_mask = e_nodes_ts < t
        # observed_ts = e_nodes_ts[observe_mask]
        # observed_nodes = e_nodes_exp[observe_mask]
        # dt = t - observed_ts
        # phi_dt = self.time_encoder(dt) # time encoding of delta t
        # linear_weights = self.linear_weights(observed_nodes)
        # out = phi_dt * linear_weights
        # out = out.sum(axis=1) 
        # out = out.sum(dim=0, keepdim=True) + self.miu(e_node_target)
        # out = soft_plus(100, out)
        # return out

        # observe_mask = e_nodes_ts < t
        # observed_ts = e_nodes_ts[observe_mask]
        # observed_nodes = e_nodes_exp[observe_mask]
        # dt = t - observed_ts
        # # import ipdb; ipdb.set_trace()
        # beta = self.beta(observed_nodes).reshape(dt.shape)
        # miu = self.miu(observed_nodes).reshape(dt.shape)
        # out = beta * torch.exp( -1 * torch.nn.functional.relu(beta) * dt ) + miu
        # if torch.isnan(out).any() or torch.isinf(out).any():
        #     import ipdb; ipdb.set_trace()
        # out = out.sum(dim=0, keepdim=True)
        # return out

        # observe_mask = e_nodes_ts < t
        # observed_ts = e_nodes_ts[observe_mask]
        # observed_nodes = e_nodes_exp[observe_mask]
        # dt = t - observed_ts
        # out = 5*torch.exp(-5*dt)*0.1
        # out = out.sum(dim=0, keepdim=True) + 0.1
        # return out

        # observe_mask = e_nodes_ts < t
        # observed_ts = e_nodes_ts[observe_mask]
        # observed_nodes = e_nodes_exp[observe_mask]
        # dt = t - observed_ts
        # phi_dt = self.time_encoder(dt) # time encoding of delta t
        # linear_weights = self.linear_weights
        # out = phi_dt * linear_weights
        # out = out.sum(axis=1) 
        # out = out.sum(dim=0, keepdim=True) + self.miu
        # out = soft_plus(100, out)
        # return out


        # only fourier encoder and recent neighbors
        # observe_mask = e_nodes_ts < t
        # observed_ts = e_nodes_ts[observe_mask]
        # observed_nodes = e_nodes_exp[observe_mask]
        # dt = t - observed_ts
        # dt, _ = torch.sort(dt)
        # dt = dt[:10]
        # phi_dt = self.time_encoder(dt) # time encoding of delta t
        # out = phi_dt.sum(axis=0, keepdim=True)
        # return out




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
    elif args.model == 'TGN_e2n':
        model = TGN_e2n(G, args.time_encoder_args, args.hidden_channels)
    else:
        raise NotImplementedError("Not implemented now")
    return model