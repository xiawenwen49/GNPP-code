# from time import time

import numpy as np
import networkx as nx
import torch
import os.path as osp
from torch.random import seed
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



class SYNDataset(InMemoryDataset):
    """ edge to node dataset """
    def __init__(self, G, root, name):
        self.G = G
        self.G_e2n = nx.line_graph(G)
        self.e2n_mapping = dict(zip(self.G_e2n.nodes, np.arange(self.G_e2n.number_of_nodes())))
        self.root= Path(root) if not isinstance(root, Path) else root
        self.name=name
        self.length_thres = 5
        super(SYNDataset, self).__init__(root=root) # will check if root/processed/processed_file_names exist. if not, run self.process()
        
        if name == 'train':
            self.data, self.slices = torch.load(osp.join(self.processed_dir, self.processed_file_names[0]))
        else:
            self.data, self.slices = torch.load(osp.join(self.processed_dir, self.processed_file_names[1]))

    @property
    def processed_file_names(self):
        return ["train.pt", "test.pt"]
    
    def process(self):

        # files = list(self.root.glob('*.txt'))
        # data_list_multiple = []
        # for graph_file in tqdm(files, total=len(files)):
        #     G, _ = read_file(graph_file, relable_nodes=False) # relable_nodes=False is important, otherwise e_node index will be different in different G
        #     self.G = G
        #     self.G_e2n = nx.line_graph(G)

        #     nodepairs = []
        #     for u, v in self.G.edges():
        #         if len(self.G[u][v]['timestamp']) >= self.length_thres:
        #             nodepairs.append([u, v])

        #     data_list = self.extract_edge_subgraphs(nodepairs) # extract subgraph
        #     data_list_multiple. extend(data_list)
        
        nodepairs = []
        for u, v in self.G.edges():
            if len(self.G[u][v]['timestamp']) >= self.length_thres:
                nodepairs.append([u, v])
        
        if 'Reddit' in str(self.root):
            nodepairs = nodepairs[:400]



        data_list = self.extract_edge_subgraphs(nodepairs) # extract subgraph
        
        train_idx = int(len(data_list)*0.8)
        train_list = data_list[:train_idx]
        test_list = data_list[train_idx:]

        # import ipdb; ipdb.set_trace()

        train_storage = self.collate(train_list)
        test_storage = self.collate(test_list)
        torch.save(train_storage, osp.join(self.processed_dir, self.processed_file_names[0])) # e.g., ./data/dataset1/`processed`/train.pt
        torch.save(test_storage, osp.join(self.processed_dir, self.processed_file_names[1]))


    def extract_edge_subgraphs(self, nodepairs):
        """ 
        1. extract a subgraph enclosing an edge. 
        2, regard edges as nodes, it should be a star graph (for 1-hop subgraph).
        3. rename all nodes (edges in original graph)

        Each Data() sample:
            x: nodes of the star edge subgraph, Integer
            edge_index: None
            e_nodes_target: a integer represents the original nodepair, Integer
            e_nodes_exp: expanded node labels, Integer
            e_nodes_ts: expanded timestamps with nodes, Float
        """
        data_list = []
        for edge in tqdm(nodepairs, total=len(nodepairs), leave=True):
            e_node_label = (edge[0], edge[1]) if edge[0] < edge[1] else (edge[1], edge[0]) # the label of target edge in self.G_e2n
            edge_subgraph = nx.ego_graph(self.G_e2n, e_node_label, radius=1)
            # new_old_map = dict(zip(range(edge_subgraph.number_of_nodes()), edge_subgraph.nodes))
            # old_new_map = dict(zip(edge_subgraph.nodes, range(edge_subgraph.number_of_nodes())))
            # edge_subgraph = nx.convert_node_labels_to_integers(edge_subgraph, first_label=0) # relabel the edge subgraph

            e_nodes_exp, e_nodes_ts = self.expand_node_timestamps(edge_subgraph) # integer, float

            e_nodes = torch.LongTensor( [self.e2n_mapping[e_label] for e_label in edge_subgraph.nodes ] )
            e_node_target = torch.LongTensor([self.e2n_mapping[e_node_label]]) # integer
            e_edge_index = None # actually TGN_e2n model do not need edge_index

            e_nodes_exp = torch.LongTensor(e_nodes_exp) 
            e_nodes_ts = torch.FloatTensor(e_nodes_ts)
            nodepair = torch.LongTensor(edge)
            T = torch.FloatTensor(self.G[edge[0]][edge[1]]['timestamp'])

            min_t = e_nodes_ts.min() - 1
            scale = 1 if 'poisson' in str(self.root) else 100
            e_nodes_ts = (e_nodes_ts - min_t)/scale
            T = (T - min_t)/scale
            
            data = Data(x=e_nodes, edge_index=e_edge_index, 
                        e_node_target=e_node_target, e_nodes_exp=e_nodes_exp, 
                        e_nodes_ts=e_nodes_ts, nodepair=nodepair, T=T) # for likelihood and lambda function in train.py
            data_list.append(data)
        return data_list
    
    def expand_node_timestamps(self, edge_subgraph):
        """ expand nodes, each corresponding with a t in e_nodes_ts """
        e_nodes_exp = []
        e_nodes_ts = []
        for e_node in edge_subgraph.nodes:
            u, v = e_node
            # if self.name == 'train':
                # timestamps = self.G[u][v]['timestamp'][:-1]
            # else:
                # timestamps = self.G[u][v]['timestamp']
            timestamps = self.G[u][v]['timestamp']
            e_node_index = self.e2n_mapping[e_node]
            e_nodes_exp.extend([e_node_index]*len(timestamps))
            e_nodes_ts.extend(timestamps)
        
        return e_nodes_exp, e_nodes_ts


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
        super(TGNDataset, self).__init__(root=root) # will check if root/processed/processed_file_names exist. if not, run self.process()
        # index = ["train", "test"].index(name)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def processed_file_names(self):
        return ["{}.pt".format(self.name)]

    def process(self):
        """ Here nodepairs only contains those with len(timestamps)>length_thres """
        nodepairs = []
        for u, v in self.G.edges():
            if len(self.G[u][v]['timestamp']) >= self.length_thres:
                nodepairs.append([u, v])
        
        data_list = self.extract_enclosing_subgraphs(nodepairs, self.edge_index) # extract subgraph
        data_list_storage = self.collate(data_list)
        # index = ["train", "test"].index(self.name)
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


def read_file(graph_file, directed=False, rescale=False, return_edgearray=False, relable_nodes=True, logger=None, **kwargs):
    # directory = Path(datadir) / dataset / (dataset + '.txt')
    edgearray = np.loadtxt(graph_file)
    edgearray = edgearray[ edgearray[:, -1].argsort() ] # sort timestamped edges in ascending order
    edgearray[:, -1] = edgearray[:, -1] - min(edgearray[:, -1]) # set earliest timestamp as 0

    edges = edgearray[:, :2].astype(np.int) 
    mask = edges[:, 0] != edges[:, 1] # 自己->自己的边不允许
    edges = edges[mask]
    edgearray = edgearray[mask]
    edges = edges.tolist() # mush be a list, or an ajcanency np array
    timestamps = edgearray[:, -1].astype(np.float)

    if directed:
        G = nx.DiGraph(edges)
    else:
        G = nx.Graph(edges)
    
    # change scale
    if rescale:
        max_interval = compute_max_interval(edgearray)
        # scale = max_interval / (np.pi - 1e-6)
        scale = 100.0 # 360
        timestamps = timestamps / scale

    # add timestamps to G    
    for i, edge in enumerate(edges):
        if G[edge[0]][edge[1]].get('timestamp', None) is None:
            G[edge[0]][edge[1]]['timestamp'] = [timestamps[i]]
        else:
            G[edge[0]][edge[1]]['timestamp'].append(timestamps[i])
    
    # relabel all nodes using integers from 0 to N-1
    old_new_dict = dict( zip(list(G.nodes), range(G.number_of_nodes()) ) )
    if relable_nodes:
        G = nx.convert_node_labels_to_integers(G, first_label=0, ordering='default')
    G.maxt = timestamps.max()

    if logger is not None:
        logger.info('# nodes: {}, # edges: {}, # inters: {}'.format(
                G.number_of_nodes(), G.number_of_edges(), len(edgearray) ))
    
    # read embeddings
    if kwargs.get('emb_file', None) is not None:
        emb_file = kwargs['emb_file']
        embeddings = np.loadtxt(emb_file, skiprows=1)
        nodeids = embeddings[:, 0].astype(np.int)
        embeddings = embeddings[:, 1:]
        embedding_matrix = np.zeros_like(embeddings)
        for i, nid in enumerate(nodeids):
            embedding_matrix[ old_new_dict[nid] ] = embeddings[i]
    else:
        embedding_matrix = None
    
    if return_edgearray:
        return G, embedding_matrix, edgearray
    else: 
        return G, embedding_matrix
    


def expand_edge_index_timestamp(sub_nodes: torch.LongTensor, sub_edgearray: torch.LongTensor, t: torch.float):
    index = sub_edgearray[:, 2] <= t
    exp_edge_index = sub_edgearray[index, :2].to(dtype=torch.int64).T
    exp_t = sub_edgearray[index, 2]

    loop_index = torch.arange(0, len(sub_nodes), dtype=torch.long, device=t.device).reshape((1, -1)).repeat(2, 1)
    loop_t = torch.zeros(len(sub_nodes), dtype=torch.float, device=t.device)
    exp_edge_index = torch.cat([exp_edge_index, loop_index], axis=1)
    exp_t = torch.cat([exp_t, loop_t], axis=0)

    return exp_edge_index, exp_t



def get_dataset(G, args):
    # train_set = EventDataset(G, 'train')

    # train_set = TGNDataset(G, args.datadir/args.dataset, 'train')
    # test_set = TGNDataset(G, args.datadir/args.dataset, 'test')
    
    train_set = SYNDataset(G, args.datadir/args.dataset, 'train')
    test_set = SYNDataset(G, args.datadir/args.dataset, 'test')
    
    return train_set, None, test_set

def get_dataloader(train_set, val_set, test_set, args):
    """ the data format for model input
    customize collect_fn for combining multiple data instances
    maybe default collect_fn is torch.cat ...
    """
    batch_size = args.batch_size
    pin_memory = False
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
    # val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
    val_loader = None
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
    return train_loader, val_loader, test_loader
    


class SyntheticSimulate(object):
    """
    Generate a synthetic dataset:
    10 nodes, 20 edges, hawkes process.
    """
    def __init__(self, root, N=100, deg=2, p=0.3, seed=123):
        """
        Args:
            # G: network topology
            root: dir for save G and edge_timestamps
            N: original graph, with N nodes
            deg: original graph, with average degree=deg.
                The original graph should have N nodes, N*deg edges.
                For timestamp simulation, the number of 'nodes' should be N*deg. Because we regard each edge of the original graph as one event 'type'.
                But we share a same model for all edges.
                And the 'adjacency' for tick should be [N*deg, N*deg]
            p: rewire probability
            seed: random seed

            N, deg, p, seed: args for generate G

        """
        self.root = root
        self.N = N
        self.deg = deg
        self.p = p
        self.seed = seed
        # self.num_instance = num_instance
        self.G = self.generate_G()
        self.G_e2n = nx.line_graph(self.G)
        self.hawkes = []
    
    def generate_G(self):
        sw = nx.watts_strogatz_graph(self.N, self.deg, self.p, seed=self.seed)
        return sw

    def generate_simulations(self, model):
        if model == 'hawkes':
            self.simulate_hawkes_timestamps()
        elif model == 'poisson':
            self.simulate_poisson_timestamps()
        else:
            raise NotImplementedError

    def simulate_self_correction_timestamps(self):
        pass

    def simulate_poisson_timestamps(self):
        " inhomogeneous poisson process "
        from tick.base import TimeFunction
        from tick.hawkes import SimuInhomogeneousPoisson
        run_time = 30

        T = np.arange((run_time * 0.9) * 5, dtype=float) / 5
        Y = np.maximum(
            15 * np.sin(T) * (np.divide(np.ones_like(T),
                                        np.sqrt(T + 1) + 0.1 * T)), 0.001)

        tf = TimeFunction((T, Y), dt=0.01)

        # We define a 1 dimensional inhomogeneous Poisson process with the
        # intensity function seen above

        edgearray = []
        for i, edge in enumerate(self.G_e2n.nodes):
            in_poi = SimuInhomogeneousPoisson([tf], end_time=run_time, verbose=False, seed=self.seed+i)

            # We activate intensity tracking and launch simulation
            # in_poi.track_intensity(0.1)
            in_poi.simulate()
            for t in in_poi.timestamps[0]:
                edgearray.append([edge[0], edge[1], t])
        
        file_name = self.root/'Synthetic_poisson.txt'
        edgearray = np.array(edgearray)
        sort_idx = np.argsort(edgearray[:, -1])
        edgearray = edgearray[sort_idx]

        np.savetxt(file_name, edgearray, fmt='%d %d %.6f')
        print('instance {} saved.'.format(file_name))
        
        
        return edgearray


    def simulate_hawkes_timestamps(self):
        from tick.hawkes import SimuHawkesExpKernels
        # edgelabel2index_mapping = dict(zip(self.G_e2n.nodes(), range(self.G_e2n.number_of_nodes()) ) )
        index2edgelabel_mapping = dict(zip(range(self.G_e2n.number_of_nodes()), self.G_e2n.nodes() ) )

        n_nodes = self.G_e2n.number_of_nodes() # each 'nodes' represents one 'type' for tick0
        adjacency = nx.adjacency_matrix(self.G_e2n, nodelist=self.G_e2n.nodes).toarray().astype(np.float) * 0.1
        baseline = 0.1 * np.ones(n_nodes)
        decays = 5 * np.ones((n_nodes, n_nodes))
        runtime = 200
        dt = 0.01


        hawkes = SimuHawkesExpKernels(adjacency=adjacency, decays=decays, baseline=baseline, verbose=False, seed=self.seed)
        hawkes.end_time = runtime
        hawkes.track_intensity(dt)
        hawkes.simulate()
        self.hawkes = hawkes

        file_name = self.root/'Synthetic_hawkes.txt'
        self.save(hawkes.timestamps, file_name)

        return hawkes.timestamps, (index2edgelabel_mapping, adjacency, baseline, decays)


        # timestamps_list = []
        # for i in range(self.num_instance):
        #     self.seed = self.seed + i
        #     hawkes = SimuHawkesExpKernels(adjacency=adjacency, decays=decays, baseline=baseline, verbose=False, seed=self.seed)
        #     hawkes.end_time = runtime
        #     # hawkes.track_intensity(dt)
        #     hawkes.simulate()
        #     timestamps_list.append(hawkes.timestamps)
        #     self.hawkes.append(hawkes)
        # return timestamps_list, (index2edgelabel_mapping, adjacency, baseline, decays)



    def save(self, timestamps, file_name):
        """ save one simulation result 
        """
        edgearray = []
        for edge, edge_ts in zip(self.G_e2n.nodes(), timestamps):
            for t in edge_ts:
                edgearray.append( [edge[0], edge[1], t] )
        edgearray = np.array(edgearray)
        sort_idx = np.argsort(edgearray[:, -1])
        edgearray = edgearray[sort_idx]

        np.savetxt(file_name, edgearray, fmt='%d %d %.6f')
        print('instance {} saved.'.format(file_name))



    # def save(self, timestamps, index=0):
    #     """ save one simulation result 
    #     """
    #     import os.path as osp
    #     edgearray = []
    #     for edge, edge_ts in zip(self.G_e2n.nodes(), timestamps):
    #         for t in edge_ts:
    #             edgearray.append( [edge[0], edge[1], t] )
    #     edgearray = np.array(edgearray)
    #     sort_idx = np.argsort(edgearray[:, -1])
    #     edgearray = edgearray[sort_idx]

    #     if index == 0:
    #         file_name = osp.join(self.root, 'Synthetic.txt')
    #     else:
    #         file_name = osp.join(self.root, 'Synthetic_{}.txt'.format(index))
    #     np.savetxt(file_name, edgearray, fmt='%d %d %.6f')
    #     print('instance {} saved.'.format(index))

    
    
    # def save_multiple(self):
    #     """ save multiple (self.num_instances) simulation results
    #     """
    #     timestamps_list, (index2edgelabel_mapping, adjacency, baseline, decays) = self.simulate_hawkes_timestamps()
    #     for i, timestamps in enumerate(timestamps_list):
    #         self.save(timestamps, index=i)
        


