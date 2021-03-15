import numpy as np
import torch
import tge.utils
from pathlib import Path
from tge.model import HarmonicEncoder, PositionEncoder, get_model
from tge.main import parse_args
from tge.train import criterion, ConditionalIntensityFunction, ConditionalDensityFunction, AttenIntensity


class TestCase():

    def test_time_encoder(self):
        timestamps = torch.arange(1, 9)
        harmonic_encoder = HarmonicEncoder(dimension=16)
        time_encodings = harmonic_encoder(timestamps)
        print(time_encodings)
        pass

    def test_dataloader(self):
        datadir = '/Users/xiawenwen/workspace/tgnn/data/CollegeMsg/'
        dataset = 'CollegeMsg'
        G, _ = tge.utils.read_file(datadir, dataset)
        train_set, val_set, test_set = tge.utils.get_dataset(G)
        train_loader, val_loader, test_loader = tge.utils.get_dataloader(train_set, val_set, test_set)

        # import ipdb; ipdb.set_trace()
        for batch in train_loader:
            assert len(batch[0]) > 0, 'Should be at least one timestamp'
            # print(batch) # e.g. [tensor([[1028, 1339]]), tensor([[4823086., 4823270.]])]

        pass

    def test_G(self):
        datadir = '/Users/xiawenwen/workspace/tgnn/data/'
        dataset = 'CollegeMsg'
        G, _ = tge.utils.read_file(datadir, dataset)

        for (u, v) in G.edges():
            if len(G[u][v]['timestamp']) == 0:
                print(u, v)
                assert False, 'Unexpected'
    
    def test_edgelist_node2vec(self):
        datadir = '/Users/xiawenwen/workspace/tgnn/data/'
        dataset = 'CollegeMsg'
        G, _ = tge.utils.read_file(datadir, dataset)

        edgelist_filename = Path(datadir)/ dataset / (dataset + '.edgelist')
        edgelist = np.array(G.edges(), dtype=np.int)
        np.savetxt(edgelist_filename, edgelist, fmt='%d')

    def test_load_emb(self):
        emb_file = '/Users/xiawenwen/workspace/tgnn/data/CollegeMsg/CollegeMsg.emb'
        embeddings = np.loadtxt(emb_file, skiprows=1)
        nodes = embeddings[:, 0].astype(np.int)
        embeddings = embeddings[:, 1:]

    def test_read_file(self):
        datadir = '/Users/xiawenwen/workspace/tgnn/data/'
        dataset = 'CollegeMsg'
        G, embedding_matrix = tge.utils.read_file(datadir, dataset)

        pass

    def test_model(self):
        datadir = '/Users/xiawenwen/workspace/tgnn/data/'
        dataset = 'CollegeMsg'
        G, embedding_matrix = tge.utils.read_file(datadir, dataset)
        args = parse_args(['--layers','2', '--in_channels', '128', '--hidden_channels', '128', '--out_channels', '128', '--dropout', '0.0'])
        # import ipdb; ipdb.set_trace()
        logger = None
        model = get_model(G, embedding_matrix, args, logger)
        train_set, val_set, test_set = tge.utils.get_dataset(G)
        train_loader, val_loader, test_loader = tge.utils.get_dataloader(train_set, val_set, test_set)
        import ipdb; ipdb.set_trace()
        for batch in train_loader:
            out = model(batch)
            # import ipdb; ipdb.set_trace()
            assert out[0].shape == (128,)
        pass

    def test_criterion(self):
        datadir = '/Users/xiawenwen/workspace/tgnn/data/'
        dataset = 'CollegeMsg'
        G, embedding_matrix = tge.utils.read_file(datadir, dataset)
        args = parse_args(['--layers','2', '--in_channels', '128', '--hidden_channels', '128', '--out_channels', '128', '--dropout', '0.0'])
        # import ipdb; ipdb.set_trace()
        logger = None
        model = get_model(G, embedding_matrix, args, logger)
        train_set, val_set, test_set = tge.utils.get_dataset(G)
        train_loader, val_loader, test_loader = tge.utils.get_dataloader(train_set, val_set, test_set)

        time_encoder = HarmonicEncoder(args.time_encoder_dimension)
        for batch in train_loader:
            uv, t = batch
            u, v = uv[0]
            t = t[0]
            hidden_reps = model(batch)
            embeddings = model.embedding(u), model.embedding(v)
            loss = criterion(hidden_reps, embeddings, batch, model, time_encoder)
            print(loss)
            break
            # import ipdb; ipdb.set_trace()
            pass

    def test_lambda(self):
        """ test conditional intensity function 
    
        """
        datadir = '/Users/xiawenwen/workspace/tgnn/data/'
        dataset = 'CollegeMsg'
        G, embedding_matrix = tge.utils.read_file(datadir, dataset)
        args = parse_args(['--layers','2', '--in_channels', '128', '--hidden_channels', '128', '--out_channels', '128', '--dropout', '0.0'])
        logger = None
        model = get_model(G, embedding_matrix, args, logger)
        train_set, val_set, test_set = tge.utils.get_dataset(G)
        train_loader, val_loader, test_loader = tge.utils.get_dataloader(train_set, val_set, test_set)

        model.prepare_hidden_rep()
        time_encoder = HarmonicEncoder(args.time_encoder_dimension)
        batch = next(iter(train_loader))
        uv, t = batch
        u, v = uv[0]
        T = t[0]

        lambdaf = ConditionalIntensityFunction(model, time_encoder)
        ft = ConditionalDensityFunction(model, time_encoder)
        lambda_t = lambdaf(u, v, T[-1] + T[-1]-T[-2], T)
        print(lambda_t)

        f_t = ft(u, v, T[-1] + T[-1]-T[-2], T)
        print(f_t)

        pass

    
    def test_ft(self):
        """ conditional density function """
        pass

    
    def test_predict(self):
        """ for expectation """
        pass
    
    def test_AttenIntensity(self):
        datadir = '/Users/xiawenwen/workspace/tgnn/data/'
        dataset = 'CollegeMsg'
        G, embedding_matrix = tge.utils.read_file(datadir, dataset)
        args = parse_args(['--layers','2', '--in_channels', '128', '--hidden_channels', '128', '--out_channels', '128', '--dropout', '0.0'])
        args.time_encoder_maxt = G.maxt # from the dataset
        args.time_encoder_args = {'maxt': args.time_encoder_maxt, 'rows': args.time_encoder_rows, 'dimension': args.time_encoder_dimension}
        logger = None
        model = get_model(G, embedding_matrix, args, logger)
        train_set, val_set, test_set = tge.utils.get_dataset(G)
        train_loader, val_loader, test_loader = tge.utils.get_dataloader(train_set, val_set, test_set, args)

        model.prepare_hidden_rep()
        batch = next(iter(train_loader))
        uv, t = batch
        u, v = uv[0]
        T_all = t[0]
        t, T = T_all[-1], T_all[:-1]

        lambdaf = AttenIntensity(model)
        lambda_t = lambdaf(u, v, T, t)
        print(lambda_t)

        ff = ConditionalDensityFunction(lambdaf)
        f_t = ff(u, v, T, T[-1] + T[-1]-T[-2])
        print(f_t)

        pred = ff.predict(u, v, T)
        print("T: {}".format(T.cpu().numpy()))
        print("t: {:.4f}, pred: {:.4f}".format(t.cpu().item(), pred.cpu().item()))
    
    def test_TGNDataset(self):
        
        pass


if __name__ == "__main__":
    test = TestCase()
    # test.test_AttenIntensity()
    test.test_TGNDataset()


