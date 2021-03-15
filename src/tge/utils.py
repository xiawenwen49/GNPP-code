import numpy as np
import networkx as nx
import copy
from pandas.core.indexes import interval
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch_geometric.data import InMemoryDataset, NeighborSampler


# COMPLETED: clarify the dataset format
# COMPLETED: implement temporal graph event model.
# COMPLETED: add deterministic time encoder/position encoding


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
    edges = edgearray[:, :2].astype(np.int) 
    edgearray[:, -1] = edgearray[:, -1] - min(edgearray[:, -1]) # the earliest as 0
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
        # import ipdb; ipdb.set_trace()
        scale = max_interval / (np.pi - 1e-6)
        scale = 360
        edgearray[:, -1] = edgearray[:, -1] / scale
    # edgearray[:, -1] = edgearray[:, -1] / 360 # change scale
    
    for i, edge in enumerate(edges): # a list
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



class TGNDataset(InMemoryDataset):
    def __init__(self, ):
        super(TGNDataset, self).__init__()
        pass

    def extract_enclosing_subgraphs(self):
        pass



def get_dataset(G, args=None):
    """ (u, v), and timestamps between u v """
    train_set = EventDataset(G, 'train')
    val_set = EventDataset(G, 'val')
    test_set = EventDataset(G, 'test')
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
    
