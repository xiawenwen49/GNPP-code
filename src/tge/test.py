
class TestCase():
    def test_synthetic_generate(self):
        """ Generate synthetic datasets """
        from tge.main import ROOT_DIR
        from tge.utils import SyntheticGenerator
        
        model_name = 'poisson' # hawkes_pos, hawkes_neg, poisson,
        root = ROOT_DIR/'data'/f'Synthetic_{model_name}'
        syn = SyntheticGenerator(root, N=1000, deg=2)
        syn.generate_simulations(model_name=model_name)

    
if __name__ == "__main__":
    test = TestCase()
    test.test_synthetic_generate()



