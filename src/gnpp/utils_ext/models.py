import torch
import torch.nn as nn


from gnpp.model import HarmonicEncoder
class GNNModel(nn.Module):
    """
    GNN models.
    e.g., 'GCN', 'GAT', 'GraphSAGE',

    The get_mini_batch_embeddings will extract embeddings of nodes in set_indice for each graph sample in a batch
    """
    def __init__(self, model_name, layers, in_features, hidden_features, out_features, set_indice_size, dropout, **kwargs):
        super(GNNModel, self).__init__()
        self.model_name = model_name
        self.layers = layers
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.set_indice_size = set_indice_size # how many nodes in set_indice for a graph sample
        
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

        self.layers = nn.ModuleList()
        Layer = self.get_layer_class()
        for i in range(1, layers+1):
            in_channels = in_features if i == 1 else hidden_features
            out_channels = hidden_features
            if self.model_name in ['GCN', 'GraphSAGE']:
                self.layers.append(Layer(in_channels=in_channels, out_channels=out_channels))
            elif self.model_name == 'GAT':
                heads = 1 if i == layers else 4
                in_channels = in_features if i == 1 else self.layers[-1].heads*self.layers[-1].out_channels
                self.layers.append(Layer(in_channels=in_channels, out_channels=out_channels, heads=heads))
            elif self.model_name == 'TAGCN':
                self.layers.append(Layer(in_channels=in_channels, out_channels=out_channels, K=kwargs['prop_depth']))
            else:
                raise NotImplementedError

        # self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_features) for i in range(layers)])
        self.linear = nn.Linear(hidden_features*set_indice_size, out_features)
        self.time_encoder = HarmonicEncoder(self.in_features)
        self.time_dim = self.time_encoder.dimension
        self.lstm = nn.GRU(input_size=self.time_dim,
                            hidden_size=self.time_dim,
                            num_layers=1,
                            batch_first=True,)

    def get_layer_class(self, ):
        from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv
        layer_dict = {'GCN': GCNConv, 'GAT': GATConv, 'GraphSAGE': SAGEConv, 'GIN': GINConv}
        Layer = layer_dict.get(self.model_name, None)
        return Layer

    def forward(self, batch):
        x, edge_index = batch.x, batch.edge_index

        # add lstm
        # add time encoder
        x_t_emb = self.time_encoder.forward_batch(x)
        # import ipdb; ipdb.set_trace()
        h0 = torch.zeros_like(x)[:, [0]]
        h0 = self.time_encoder.forward_batch(h0) # [N, 1, embed_dim]
        h0 = h0.permute(1, 0, 2) # [1, N, embed_dim]

        # import ipdb; ipdb.set_trace()
        output, h_n = self.lstm(x_t_emb, h0)
        x = output[:, -1, :].squeeze(1) # restore shape of x

        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            x = self.act(x)
            x = self.dropout(x)
        
        mini_batch_x = self.get_mini_batch_embeddings(x, batch) # different application may be diverse for this function
        mini_batch_x = self.linear(mini_batch_x)
        return mini_batch_x

    def get_mini_batch_embeddings(self, x, batch): 
        """
        Extract embeddings of nodes in set_indice for each graph sample in a batch
        """  
        device = x.device
        set_indice, batch_idx, num_graphs = batch.set_indice, batch.batch, batch.num_graphs
        num_nodes = torch.eye(num_graphs)[batch_idx].to(device).sum(dim=0) # 每一纬表示那个subgraph，其有多少node
        zero = torch.tensor([0], dtype=torch.long).to(device)
        index_bases = torch.cat([zero, torch.cumsum(num_nodes, dim=0, dtype=torch.long)[:-1] ]) # 每一个graph的第一个node在x中的的索引号
        index_bases = index_bases.unsqueeze(1).expand(-1, set_indice.size(-1)) # 复制为若干列
        set_indice_batch = index_bases + set_indice # 这一步是计算每一个subgraph中的那个target node set，其在x中的index是多少

        x = x[set_indice_batch] # [batch_size, set_indice.size(-1), hidden_features]
        x = self.pool(x)
        return x

    def pool(self, x):
        """
        For each graph sample, pool embeddings of nodes in set_indice extracted from this graph sample 
        """
        x = x.reshape((x.shape[0], -1)) # simple flatten here
        return x