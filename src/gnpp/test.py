import argparse

class TestCase():
    def test_synthetic_generate(self, model_name):
        """ Generate synthetic datasets """
        from gnpp.main import ROOT_DIR
        from gnpp.utils import SyntheticGenerator
        
        # model_name = 'poisson' # hawkes_pos, hawkes_neg, poisson,
        root = ROOT_DIR/'data'/f'Synthetic_{model_name}'
        syn = SyntheticGenerator(root, N=1000, deg=2)
        syn.generate_simulations(model_name=model_name)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser('Preprocessing')
    parser.add_argument('--dataset', type=str, choices=['hawkes_pos', 'hawkes_neg', 'poisson'], default='poisson', help='Dataset' )
    args = parser.parse_args()
    test = TestCase()
    # import ipdb; ipdb.set_trace()
    test.test_synthetic_generate(args.dataset)



