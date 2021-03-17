from time import time
import numpy as np
import networkx as nx
import torch
from tqdm import tqdm
from pathlib import Path
# from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch_geometric.data import DataLoader
from torch_geometric.data import Data, InMemoryDataset, NeighborSampler
from torch_geometric.utils import k_hop_subgraph


# COMPLETED: clarify the dataset format
# COMPLETED: implement temporal graph event model.
# COMPLETED: add deterministic time encoder/position encoding

class EventDataset(Dataset):
    def __init__(self, G, name):
        self.G = G
        self.name = name
        self.all_edges = []
        for edge in G.edges():
            if len(G[edge[0]][edge[1]]['timestamp']) >= 5: # >= 5
                self.all_edges.append(edge)
        self.all_edges = self.all_edges[:40] # for debug
    
    def __len__(self):
        return len(self.all_edges)

    def __getitem__(self, index):
        if self.name == 'train':
            u, v = self.all_edges[index]
            return torch.LongTensor([u, v]), torch.FloatTensor(self.G[u][v]['timestamp'][:-1] ) # return all timestamps except the last one, as train
        else:
            u, v = self.all_edges[index]
            # return torch.LongTensor([u, v]), torch.FloatTensor([self.G[u][v]['timestamp'][-1] ] ) # only return the last one, as label
            return torch.LongTensor([u, v]), torch.FloatTensor(self.G[u][v]['timestamp']) # return all,  the last one as label



# COMPLETED: 1, the usage of InMemoryDataset? Why InMemoryDataset? -> just to use its collate() function, to combine many samples to one
# COMPLETED: 2, the process() and extract_subgraph()? -> only static subgraph, regardless of timestamps
class TGNDataset(InMemoryDataset):
    def __init__(self, G, root, name=None):
        """
        Args:
            G: nx obj, with all timestamps
            root: str, root dir of a specific dataset, e.g., ./data/dataset1/
            # dataset: str, the dataset name, e.g., dataset1
            name: 'train', 'val', 'test'
        """
        self.G = G
        self.root=root
        # self.dataset = dataset
        self.name=name
        self.edge_index = torch.LongTensor( np.concatenate([ np.array(self.G.edges(), dtype=np.int), np.array(self.G.edges(), dtype=np.int)[:, [1,0]] ], axis=0 ).T )
        self.length_thres = 5
        super(TGNDataset, self).__init__(root=root) # will run self.process()
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def processed_file_names(self):
        # return ["{}.pt".format(self.dataset), ]
        return ["train.pt", ]

    def process(self):
        """ Here nodepairs only contains those with len(timestamps)>length_thres """
        nodepairs = []
        for u, v in self.G.edges():
            if len(self.G[u][v]['timestamp']) >= self.length_thres:
                nodepairs.append([u, v])
        
        data_list = self.extract_enclosing_subgraphs(nodepairs, self.edge_index) # extract subgraph
        data_list_storage = self.collate(data_list)
        file_name = self.processed_paths[0] # e.g., ./data/dataset1/`processed`/train.pt
        torch.save(data_list_storage, file_name)


    def extract_enclosing_subgraphs(self, nodepairs, edge_index):
        """
        Extract subgraphs for each node pair in nodepairs.
        Here the extracted subgraph is based on all edge_index, regardless of timestamps.
        Masking timestamps<=t with a specific `t` is done in model.forward().
        """
        data_list = []
        for u, v in tqdm(nodepairs, total=len(nodepairs)):
            sub_nodes, sub_edge_index, mapping, edge_mask = k_hop_subgraph([u, v], 1, edge_index, relabel_nodes=True)
            # confirmed: if relabel_nodes=True, new label of sub_nodes[i] is i, for new edge_index. 
            if self.name == 'train':
                T = torch.tensor(self.G[u][v]['timestamp'][:-1])
            else:
                T = torch.tensor(self.G[u][v]['timestamp'])
            
            edgearray = []
            new_old_map = dict(zip(range(len(sub_nodes)), sub_nodes.cpu().numpy()))
            for i, (u, v) in enumerate(zip(sub_edge_index[0].cpu().numpy(), sub_edge_index[1].cpu().numpy())):
                if self.name == 'train':
                    timestamp = torch.tensor( self.G[ new_old_map[u] ][ new_old_map[v] ]['timestamp'][:-1] ).reshape(1, -1)
                else:
                    timestamp = torch.tensor( self.G[ new_old_map[u] ][ new_old_map[v] ]['timestamp'] ).reshape(1, -1)
                e_ext = sub_edge_index[:, i].reshape(-1, 1).repeat(1, timestamp.numel())
                e_ext = torch.cat([e_ext, timestamp], axis=0)
                edgearray.append(e_ext)
            edgearray = torch.cat(edgearray, axis=1).T # save all expanded edge-t into a Data() sample, to accelerate edge-t selection in model forward.

            data = Data(x=sub_nodes, edge_index=sub_edge_index, nodepair=torch.LongTensor([u, v]), mapping=mapping, T=T, edgearray=edgearray)
            data_list.append(data)
        return data_list

def compute_max_interval(edgearray):
    edges = edgearray[:, :2].astype(np.int)
    timestamps = edgearray[:, -1]
    G = nx.Graph(edges.tolist())
    for i, (u, v) in enumerate(edges):
        if G[u][v].get('timestamp', None) is None:
            G[u][v]['timestamp'] = [timestamps[i]]
        else: 
            G[u][v]['timestamp'].append(timestamps[i])
    max_intervals = []
    for i, (u, v) in enumerate(edges):
        intervals = np.array( G[u][v]['timestamp'] )
        if len(intervals) >= 5:
            intervals = np.sort(intervals)
            intervals = intervals[1:] - intervals[:-1]
            max_intervals.append(intervals.max())
    max_interval = np.max(max_intervals)
    # import ipdb; ipdb.set_trace()
    return max_interval


def read_file(datadir, dataset, directed=False, preprocess=True, logger=None, return_edgearray=False):
    directory = Path(datadir) / dataset / (dataset + '.txt')
    edgearray = np.loadtxt(directory)
    edgearray = edgearray[ edgearray[:, -1].argsort() ] # sort timestamped edges in ascending order
    edgearray[:, -1] = edgearray[:, -1] - min(edgearray[:, -1]) # set earliest timestamp as 0

    edges = edgearray[:, :2].astype(np.int) 
    mask = edges[:, 0] != edges[:, 1] # 自己->自己的边不允许
    edges = edges[mask]
    edgearray = edgearray[mask]
    edges = edges.tolist() # mush be a list, or an ajcanency np array

    if directed:
        G = nx.DiGraph(edges)
    else:
        G = nx.Graph(edges)
    
    # change scale
    if preprocess:
        max_interval = compute_max_interval(edgearray)
        # scale = max_interval / (np.pi - 1e-6)
        scale = 360
        edgearray[:, -1] = edgearray[:, -1] / scale

    # add timestamps to G    
    for i, edge in enumerate(edges):
        if G[edge[0]][edge[1]].get('timestamp', None) is None:
            G[edge[0]][edge[1]]['timestamp'] = [edgearray[i][-1]]
        else:
            G[edge[0]][edge[1]]['timestamp'].append(edgearray[i][-1])
    
    # relabel all nodes using integers from 0 to N-1
    old_new_dict = dict( zip(list(G.nodes), range(G.number_of_nodes()) ) )
    G = nx.convert_node_labels_to_integers(G, first_label=0, ordering='default')
    G.maxt = edgearray[:, -1].max()
    
    # read embeddings
    emb_file = Path(datadir) / dataset / (dataset + '.emb')
    embeddings = np.loadtxt(emb_file, skiprows=1)
    nodeids = embeddings[:, 0].astype(np.int)
    embeddings = embeddings[:, 1:]
    embedding_matrix = np.zeros_like(embeddings)
    for i, nid in enumerate(nodeids):
        embedding_matrix[ old_new_dict[nid] ] = embeddings[i]

    if logger is not None:
        logger.info('Read in {}, # nodes: {}, # edges: {}, # inters: {}'.format(
            dataset, G.number_of_nodes(), G.number_of_edges(), len(edgearray) ))
    
    if return_edgearray:
        return G, embedding_matrix, edgearray
    else: return G, embedding_matrix


def expand_edge_index_timestamp(sub_nodes: torch.LongTensor, sub_edgearray: torch.LongTensor, t: torch.float):
    index = sub_edgearray[:, 2] <= t
    exp_edge_index = sub_edgearray[index, :2].to(dtype=torch.int64).T
    exp_t = sub_edgearray[index, 2]

    loop_index = torch.arange(0, len(sub_nodes), dtype=torch.long, device=t.device).reshape((1, -1)).repeat(2, 1)
    loop_t = torch.zeros(len(sub_nodes), dtype=torch.float, device=t.device)
    exp_edge_index = torch.cat([exp_edge_index, loop_index], axis=1)
    exp_t = torch.cat([exp_t, loop_t], axis=0)

    return exp_edge_index, exp_t


# def expand_edge_index_timestamp(G, mapping, sub_nodes, sub_edge_index: torch.LongTensor, t: torch.float):

#     """ expand edge_index, according to legal timestamps, i.e., <=t """
#     device = t.device
#     sub_edge_index = sub_edge_index.cpu().numpy()
#     t = t.cpu().item()
#     exp_edge_index = []
#     exp_t = []
#     for (u, v) in sub_edge_index.T:
#         u_ = mapping[u] # original node number
#         v_ = mapping[v]
#         timestamps = np.array(G[u_][v_]['timestamp'])
#         if timestamps.min() > t:
#             continue
#         if timestamps.max() <= t:
#             pos = len(timestamps)
#         else:
#             pos = np.searchsorted(timestamps, t, side='right')

#         exp_edge_index.extend([[u, v]]*pos)
#         exp_t.extend(timestamps[:pos])
    

#     exp_edge_index = torch.LongTensor(exp_edge_index).T.to(device)
#     exp_t = torch.FloatTensor(exp_t).to(device)

#     # add self loops
#     loop_index = torch.arange(0, len(sub_nodes), dtype=torch.long, device=device).reshape((1, -1)).repeat(2, 1)
#     loop_t = torch.zeros(len(sub_nodes), dtype=torch.long, device=device)
#     exp_edge_index = torch.cat([exp_edge_index, loop_index], axis=1)
#     exp_t = torch.cat([exp_t, loop_t], axis=0)

#     assert exp_t.shape[0] == exp_edge_index.shape[1], 'The length should be equal'
#     return exp_edge_index, exp_t

def get_dataset(G, args):
    # train_set = EventDataset(G, 'train')
    # val_set = EventDataset(G, 'val')
    # test_set = EventDataset(G, 'test')

    train_set = TGNDataset(G, args.datadir/args.dataset, 'train')
    val_set = TGNDataset(G, args.datadir/args.dataset, 'val')
    test_set = TGNDataset(G, args.datadir/args.dataset, 'test')
    return train_set, val_set, test_set

def get_dataloader(train_set, val_set, test_set, args):
    """ the data format for model input
    customize collect_fn for combining multiple data instances
    maybe default collect_fn is torch.cat ...
    """
    batch_size = args.batch_size
    pin_memory = False
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
    return train_loader, val_loader, test_loader
    
