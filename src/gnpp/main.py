import argparse
import sys
import os
import time
import warnings
from pathlib import Path

from gnpp.utils_ext.log import set_up_log
from gnpp.utils import read_file, get_dataset, get_dataloader
from gnpp.model import get_model
from gnpp.train import train_model

warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)) ).parent.parent

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args(argstring=None):
    parser = argparse.ArgumentParser('Temporal GNNs.')
    
    # general settings
    parser.add_argument('--root_dir', type=str, default=ROOT_DIR, help='Root directory' )
    parser.add_argument('--checkpoint_dir', type=str, default=ROOT_DIR/'checkpoint/', help='Root directory' )
    parser.add_argument('--datadir', type=str, default= ROOT_DIR/'data/', help='Dataset edge file name')
    parser.add_argument('--dataset', type=str, default='Wikipedia', choices=['CollegeMsg', 'Wikipedia', 'Reddit', 'Synthetic_hawkes_neg', 'Synthetic_hawkes_pos', 'Synthetic_poisson', 'emailEuCoreTemporal', 'SMS-A', 'facebook-wall', 'Synthetic'], help='Dataset edge file name')
    parser.add_argument('--directed', type=bool, default=False, help='(Currently unavailable) whether to treat the graph as directed')
    parser.add_argument('--gpu', type=int, default=1, help='-1: cpu, others: gpu index')
    parser.add_argument('--eval', type=str, default='', help='a time_str. evaluate model using checpoint_dir/dataset/time_str/state_dict_filename.state_dict')
    parser.add_argument('--state_dict', type=str, help='state_dict filename (without suffix)')
    parser.add_argument('--parallel_eval', default=None, action='store_true', help='parallel evaluate mode')

    # dataset 
    parser.add_argument('--seed', type=int, default=0, help='seed to initialize all the random modules')
    parser.add_argument('--data_usage', type=float, default=1.0, help='ratio of used data for all data samples')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='test ratio in the used data samples')

    # model parameter
    parser.add_argument('--model', type=str, default='GNPP', choices=['GNPP', 'GAT', 'GraphSAGE'], help='model name')
    parser.add_argument('--layers', type=int, default=1, help='largest number of layers')
    parser.add_argument('--in_channels', type=int, default=128, help='input dim')
    parser.add_argument('--hidden_channels', type=int, default=128, help='hidden dim')
    parser.add_argument('--out_channels', type=int, default=128, help='output dim')
    parser.add_argument('--num_heads', type=int, default=1, help='number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--negative_slope', type=float, default=0.2, help='for leakey relu function')
    parser.add_argument('--with_neig', type=int, default=1, help='1: with neighbor, 0: without neighbor')
    
    parser.add_argument('--epochs', type=int, default=30, help='training epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='mini batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--l2', type=float, default=1e-3, help='l2 regularization') # 1e-4
    parser.add_argument('--optim', type=str, default='adam', help='optimizer (string)')

    # time encoder params
    parser.add_argument('--time_encoder_type', type=str, choices=['pe', 'fe', 'he'], default='pe', help='time encoder type')
    parser.add_argument('--time_encoder_maxt', type=float, default=300, help='time encoder maxt') # 3e6
    parser.add_argument('--time_encoder_rows', type=int, default=int(30000), help='time encoder rows') # 1e6
    parser.add_argument('--time_encoder_dimension', type=int, default=512, help='time encoding dimension') # 128, 512

    # logging and debug
    parser.add_argument('--log_dir', type=str, default=ROOT_DIR/'log/', help='log directory')
    parser.add_argument('--save_log', default=True, action='store_true', help='save console log into log file')
    parser.add_argument('--debug', default=False, action='store_true', help='debug mode')
    parser.add_argument('--desc', type=str, default='description_string', help='a string description for an experiment')
    parser.add_argument('--time_str', type=str, default=time.strftime('%Y_%m_%d_%H_%M_%S'), help='execution time')
    

    try:
        if argstring is not None:
            args = parser.parse_args(argstring)
        else:
            args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    return args

def main():
    # arg and log settings
    args = parse_args()
    sys_argv = sys.argv
    logger = set_up_log(args, sys_argv)

    # read in dataset
    graph_file = Path(args.datadir) / args.dataset / (args.dataset + '.txt')
    relable_nodes = False if args.dataset == 'Synthetic' else True
    G, embedding_matrix = read_file(graph_file, relable_nodes=relable_nodes) # read graph

    # dataloaders
    train_set, val_set, test_set = get_dataset(G, args)
    dataloaders = get_dataloader(train_set, val_set, test_set, args)
    

    # build model
    args.time_encoder_maxt = G.maxt # from the dataset
    args.time_encoder_args = {'maxt': args.time_encoder_maxt, 'rows': args.time_encoder_rows, 'dimension': args.time_encoder_dimension, 'type': args.time_encoder_type}
    model = get_model(G, embedding_matrix, args, logger)

    # train model
    train_model(model, dataloaders, args, logger)

if __name__ == "__main__":
    main()
