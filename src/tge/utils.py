
import numpy as np
import networkx as nx
import torch
import os
import os.path as osp
from tqdm import tqdm
from pathlib import Path
from torch_geometric.data import DataLoader
from torch_geometric.data import Data, InMemoryDataset, NeighborSampler
from tick.base import TimeFunction
from tick.hawkes import SimuInhomogeneousPoisson, SimuHawkes, SimuHawkesExpKernels, HawkesKernelTimeFunc, HawkesKernel0, HawkesKernelExp



class GNPPDataset(InMemoryDataset):
    """ edge to node dataset """
    def __init__(self, G, root, name, rescale=10000):
        self.G = G
        self.G_e2n = nx.line_graph(G)
        self.e2n_mapping = dict(zip(self.G_e2n.nodes, np.arange(self.G_e2n.number_of_nodes())))
        self.root= Path(root) if not isinstance(root, Path) else root
        self.name=name
        self.rescale=rescale
        self.length_thres = 5
        super(GNPPDataset, self).__init__(root=root) # will check if root/processed/processed_file_names exist. if not, run self.process()
        
        if name == 'train':
            self.data, self.slices = torch.load(osp.join(self.processed_dir, self.processed_file_names[0]))
        else:
            self.data, self.slices = torch.load(osp.join(self.processed_dir, self.processed_file_names[1]))

    @property
    def processed_file_names(self):
        return ["train.pt", "test.pt"]
    
    def process(self):
        nodepairs = []
        for u, v in self.G.edges():
            if len(self.G[u][v]['timestamp']) >= self.length_thres:
                nodepairs.append([u, v])
        
        # if 'Reddit' in str(self.root):
        #     nodepairs = nodepairs[:500] # Reddit has a dense subgraph
    
        data_list = self.extract_edge_subgraphs(nodepairs, rescale=self.rescale) # extract subgraph
        
        train_idx = int(len(data_list)*0.75)
        train_list = data_list[:train_idx]
        test_list = data_list[train_idx:]

        train_storage = self.collate(train_list)
        test_storage = self.collate(test_list)
        torch.save(train_storage, osp.join(self.processed_dir, self.processed_file_names[0])) # e.g., ./data/dataset1/`processed`/train.pt
        torch.save(test_storage, osp.join(self.processed_dir, self.processed_file_names[1]))


    def extract_edge_subgraphs(self, nodepairs, rescale=1):
        """ 
        1. extract a subgraph enclosing an edge. 
        2, regard edges as nodes, it should be a star graph (for 1-hop subgraph).
        3. rename all nodes as there indexs (each subgraph nodes index is not from 0 to |subgraph.nodes|)

        Each Data() sample:
            x: nodes of the edge subgraph, Integer
            edge_index: None
            e_nodes_target: a integer represents the original nodepair, Integer
            e_nodes_exp: expanded edge-node labels, Integer
            e_nodes_ts: expanded timestamps on edge-nodes, Float
        """
        data_list = []
        for edge in tqdm(nodepairs, total=len(nodepairs), leave=True):
            e_node_label = tuple( sorted(edge) ) # edge label in line graph
            edge_subgraph = nx.ego_graph(self.G_e2n, e_node_label, radius=1)

            e_nodes_exp, e_nodes_ts = self.expand_node_timestamps(edge_subgraph) # integer, float

            e_nodes = torch.LongTensor( [self.e2n_mapping[e_label] for e_label in edge_subgraph.nodes ] )
            e_node_target = torch.LongTensor([self.e2n_mapping[e_node_label]]) # integer
            e_edge_index = None # actually TGN_e2n model do not need edge_index

            e_nodes_exp = torch.LongTensor(e_nodes_exp) 
            e_nodes_ts = torch.FloatTensor(e_nodes_ts)
            nodepair = torch.LongTensor(edge)
            T = torch.FloatTensor(self.G[edge[0]][edge[1]]['timestamp'])

            # rescale and start from 0, for each subgraph sample
            min_t = e_nodes_ts.min()
            e_nodes_ts = e_nodes_ts - min_t
            T = T - min_t
            e_nodes_ts = e_nodes_ts/rescale
            T = T/rescale
            
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
            timestamps = self.G[u][v]['timestamp']
            e_node_index = self.e2n_mapping[e_node]
            e_nodes_exp.extend([e_node_index]*len(timestamps))
            e_nodes_ts.extend(timestamps)
        
        return e_nodes_exp, e_nodes_ts


class GNNDatatset(GNPPDataset):
    """ For GNN adaptation

    1. extract a subgraph enclosing an edge. 
    2, regard edges as nodes, it should be a star graph (for 1-hop subgraph).
    3. rename all nodes in a subgraph from 0 to |subgraph.nodes|.
    4, each data sampele must have edgeindex, for GNN model message-passing.
    5, each node feature is timestamps, padded with 0 to a fixed length. (timestamps may be scaled)
    6, each datasample has a y label, indicats the last event time of central node, for prediction and evaluation.
    """
    def process(self):
        super(GNNDatatset, self).process() # must include this, otherwise self.__class__.__dict__ has no 'process' string in keys
        
    def extract_edge_subgraphs(self, nodepairs, rescale=1):
        data_list = []
        feature_dim = 20 # pre-defined feature length
        for edge in tqdm(nodepairs, total=len(nodepairs), leave=True):
            e_node_label = tuple( sorted(edge) ) # edge label in line graph
            edge_subgraph = nx.ego_graph(self.G_e2n, e_node_label, radius=1) # sub line graph
            if edge_subgraph.number_of_edges() < 2:
                continue
            new_old_mapping = dict(zip(range(edge_subgraph.number_of_nodes()), edge_subgraph.nodes()))
            old_new_mapping = dict(zip(edge_subgraph.nodes(), range(edge_subgraph.number_of_nodes())))
            edge_subgraph = nx.convert_node_labels_to_integers(edge_subgraph, first_label=0, ordering="default")
            e_node_index = old_new_mapping[e_node_label] # new index of the central edge-node
            
            edge_list = np.array(edge_subgraph.edges(), dtype=int) # edge list
            edge_list = np.concatenate([edge_list, edge_list[:, [1, 0]]])
            edge_list = edge_list.T
            
            x = []
            for i in range(edge_subgraph.number_of_nodes()): # all node timestamps
                u, v = new_old_mapping[i]
                timestamp = np.array(self.G[u][v]['timestamp']) / rescale
                timestamp = timestamp[:feature_dim]
                timestamp = np.pad(timestamp, (0, max(feature_dim-len(timestamp), 0)), 'constant') # default constant=0
                x.append(timestamp)
            x = np.array(x)

            y = self.G[e_node_label[0]][e_node_label[1]]['timestamp'][-1] / rescale # target timestamp: the last timestamp of central node
            thres = self.G[e_node_label[0]][e_node_label[1]]['timestamp'][-2] / rescale
            x[x>thres] = 0 # mask all should-not-observed timestamps

            edge_list = torch.LongTensor(edge_list)
            x = torch.FloatTensor(x)
            y = torch.FloatTensor([y])
            c_index = torch.LongTensor([[e_node_index,]])
            T = torch.FloatTensor(self.G[edge[0]][edge[1]]['timestamp']) / rescale
            data = Data(x=x, edge_index=edge_list, y=y, set_indice=c_index, T=T)
            # import ipdb; ipdb.set_trace()
            data_list.append(data)
        return data_list




def read_file(graph_file, directed=False, rescale=False, return_edgearray=False, relable_nodes=True, logger=None, **kwargs):
    edgearray = np.loadtxt(graph_file)

    edgearray = edgearray[ edgearray[:, -1].argsort() ] # IMPORTANT: sort timestampes
    edgearray[:, -1] = edgearray[:, -1] - min(edgearray[:, -1]) # set earliest timestamp as 0

    edges = edgearray[:, :2].astype(np.int) 
    mask = edges[:, 0] != edges[:, 1] # self->self is not allowed
    edges = edges[mask]
    edgearray = edgearray[mask]
    edges = edges.tolist() # mush be a list, or an ajcanency np array
    timestamps = edgearray[:, -1].astype(np.float)

    if directed:
        G = nx.DiGraph(edges)
    else:
        G = nx.Graph(edges)
    
    if rescale:
        assert kwargs.get('scale', None) is not None
        scale = kwargs['scale']
        timestamps = timestamps / scale

    # add timestamps
    for i, edge in enumerate(edges):
        if G[edge[0]][edge[1]].get('timestamp', None) is None:
            G[edge[0]][edge[1]]['timestamp'] = [timestamps[i]]
        else:
            if timestamps[i] > G[edge[0]][edge[1]]['timestamp'][-1]:
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
    

def get_dataset(G, args):
    if args.model == 'GNPP':
        DatasetClass = GNPPDataset
    elif args.model in ['GAT', 'GraphSAGE']:
        DatasetClass = GNNDatatset
    
    if args.dataset in ['Synthetic_hawkes_neg', 'Synthetic_hawkes_pos', 'Synthetic_poisson']:
        rescale = 1
    elif args.dataset in ['Wikipedia', 'Reddit', 'CollegeMsg']:
        rescale = 10000
    else:
        raise NotImplementedError()


    train_set = DatasetClass(G, args.datadir/args.dataset, 'train', rescale=rescale)
    test_set = DatasetClass(G, args.datadir/args.dataset, 'test', rescale=rescale)
    
    return train_set, None, test_set

def get_dataloader(train_set, val_set, test_set, args):
    """ the data format for model input
    customize collect_fn for combining multiple data instances
    maybe default collect_fn is torch.cat ...
    """
    batch_size = args.batch_size
    pin_memory = False
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
    val_loader = None
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
    return train_loader, val_loader, test_loader
    


class SyntheticGenerator(object):
    """
    Generate a synthetic dataset
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
        """
        self.root = root
        self.N = N
        self.deg = deg
        self.p = p
        self.seed = seed
        self.G = self.generate_G(self.N, self.deg, self.p, seed=self.seed)
        self.G_e2n = nx.line_graph(self.G)
        self.elabel_idx_map = dict(zip(self.G_e2n.nodes(), range(self.G_e2n.number_of_nodes()) ) )
        self.idx_elabel_map = dict(zip(range(self.G_e2n.number_of_nodes()), self.G_e2n.nodes() ) )
        self.hawkes = []
    
    def generate_G(self, n, k, p, seed):
        sw = nx.watts_strogatz_graph(n, k, p, seed)
        return sw

    def generate_simulations(self, model_name):
        if model_name in ['hawkes_pos', 'hawkes_neg']:
            self.simulate_hawkes(model_name)
        elif model_name == 'poisson':
            self.simulate_poisson()
        else:
            raise NotImplementedError


    def simulate_hawkes(self, model_name):
        self.model_name = model_name
        def y_func_pos(t_values):
            y_values = 0.02*np.exp(-t_values)
            return y_values
        
        def y_func_neg(t_values):
            y_values = -0.1*np.exp(-t_values)
            return y_values
        
        if model_name == 'hawkes_neg':
            y_func = y_func_neg
        elif model_name == 'hawkes_pos':
            y_func = y_func_pos


        t_values = np.linspace(0, 101, 100)
        y_values = y_func(t_values)
        tf = TimeFunction([t_values, y_values], inter_mode=TimeFunction.InterLinear, dt=0.1)

        tf_kernel = HawkesKernelTimeFunc(tf)
        
        N_enodes = self.G_e2n.number_of_nodes() # regarded as 'N_enodes' types

        base_int = 0.2
        baselines = [base_int for i in range(N_enodes)]
        kernels = [[] for i in range(N_enodes)]
        for i in range(N_enodes):
            for j in range(N_enodes):
                if i == j:
                    # kernels[i].append(HawkesKernel0())
                    kernels[i].append(HawkesKernelExp(.1, 4)) # self influence
                else:
                    if self.G_e2n.has_edge( self.idx_elabel_map[i], self.idx_elabel_map[j] ):
                        kernels[i].append(tf_kernel)
                    else:
                        kernels[i].append(HawkesKernel0())

        hawkes = SimuHawkes(kernels=kernels,
                            baseline=baselines,
                            verbose=False, seed=self.seed)
        hawkes.threshold_negative_intensity(allow=True)

        run_time = 100
        hawkes.end_time = run_time
        hawkes.simulate()
        timestamps = hawkes.timestamps

        self.save(timestamps, self.model_name)


    def simulate_poisson(self):
        " inhomogeneous poisson process "
        self.model_name = 'poisson'
    
        run_time = 100
        T = np.arange((run_time * 0.9) * 5, dtype=float) / 5
        Y = np.maximum(
            15 * np.sin(T) * (np.divide(np.ones_like(T),
                                        np.sqrt(T + 1) + 0.1 * T)), 0.001)

        tf = TimeFunction((T, Y), dt=0.01)

        # We define a 1 dimensional inhomogeneous Poisson process with the intensity function seen above
        timestamps = []
        for i, edge in enumerate(self.G_e2n.nodes):
            in_poi = SimuInhomogeneousPoisson([tf], end_time=run_time, verbose=False, seed=self.seed+i)

            # We activate intensity tracking and launch simulation
            # in_poi.track_intensity(0.1)
            in_poi.simulate()
            ts = in_poi.timestamps[0]
            timestamps.append(ts)
        
        self.save(timestamps, self.model_name)


    def save(self, timestamps, model_name):
        """ save to Temporal Graph's edgearray format.
        """
        edgearray = []
        
        for idx, ts in enumerate(timestamps):
            edge = self.idx_elabel_map[idx]
            for t in ts:
                edgearray.append( [edge[0], edge[1], t] )
            

        edgearray = np.array(edgearray)
        sort_idx = np.argsort(edgearray[:, -1])
        edgearray = edgearray[sort_idx]

        # import ipdb; ipdb.set_trace()
        if not self.root.exists():
            os.makedirs(self.root)
        file_name = self.root/f'Synthetic_{model_name}.txt'
        np.savetxt(file_name, edgearray, fmt='%d %d %.6f')
        print('Model {}, graph file {} saved.'.format(model_name, file_name))
    
