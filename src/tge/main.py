import argparse
import numpy as np
import sys
import torch
import os
import time
import json
import warnings
from pathlib import Path
from . import log
from . import utils
from .model import get_model, HarmonicEncoder
from .train import train_model, evaluate_state_dict


warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
torch.autograd.set_detect_anomaly(True)


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
    parser.add_argument('--root_dir', type=str, default=os.path.dirname(os.path.abspath(__file__)), help='Root directory' )
    parser.add_argument('--checkpoint_dir', type=str, default=os.path.join(ROOT_DIR, 'checkpoint/'), help='Root directory' )
    parser.add_argument('--datadir', type=str, default= os.path.join(ROOT_DIR, 'data/'), help='Dataset edge file name')
    parser.add_argument('--dataset', type=str, default='CollegeMsg', choices=['CollegeMsg', 'emailEuCoreTemporal', 'SMS-A', 'facebook-wall'], help='Dataset edge file name')
    parser.add_argument('--directed', type=bool, default=False, help='(Currently unavailable) whether to treat the graph as directed')
    parser.add_argument('--gpu', type=int, default=0, help='-1: cpu, others: gpu index')
    parser.add_argument('--eval', type=str, default='', help='a time_str. evaluate model using checpoint_dir/dataset/time_str/state_dict_filename.state_dict')
    parser.add_argument('--state_dict', type=str, help='state_dict filename (without suffix)')
    parser.add_argument('--parallel_eval', default=None, action='store_true', help='parallel evaluate mode')

    # dataset 
    parser.add_argument('--seed', type=int, default=0, help='seed to initialize all the random modules')
    parser.add_argument('--data_usage', type=float, default=1.0, help='ratio of used data for all data samples')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='test ratio in the used data samples')

    # model training
    parser.add_argument('--model', type=str, default='TGN', choices=['TGN'], help='model name')
    parser.add_argument('--layers', type=int, default=2, help='largest number of layers')
    parser.add_argument('--in_channels', type=int, default=128, help='input dim')
    parser.add_argument('--hidden_channels', type=int, default=128, help='hidden dim')
    parser.add_argument('--out_channels', type=int, default=128, help='output dim')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--negative_slope', type=float, default=0.2, help='for leakey relu function')
    
    parser.add_argument('--epochs', type=int, default=30, help='training epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='mini batch size')
    parser.add_argument('--lr', type=float, default=5*1e-4, help='learning rate')
    parser.add_argument('--l2', type=float, default=1e-3, help='l2 regularization')
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer (string)')
    # parser.add_argument('--metric', type=str, default='acc', help='evaluation metric')

    # important features
    # parser.add_argument('--in_features', type=int, default=9, help='initial input features of nodes')
    # parser.add_argument('--out_features', type=int, default=6, help='number of target classes')

    # parser.add_argument('--time_encoder_type', type=str, default='tat', choices=['tat', 'harmonic', 'empty'], help='time encoder type')
    parser.add_argument('--time_encoder_maxt', type=float, default=3e6, help='time encoder maxt')
    parser.add_argument('--time_encoder_rows', type=int, default=int(1e6), help='time encoder rows')
    parser.add_argument('--time_encoder_dimension', type=int, default=128, help='time encoding dimension')
    # parser.add_argument('--time_encoder_deltas', type=float, default=0.5, help='scale of mean time interval for discretization')

    # logging and debug
    parser.add_argument('--log_dir', type=str, default=os.path.join(ROOT_DIR, 'log/'), help='log directory')
    parser.add_argument('--save_log', default=True, action='store_true', help='save console log into log file')
    # parser.add_argument('--debug', default=False, action='store_true', help='debug mode')
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
    logger = log.set_up_log(args, sys_argv)

    # read in dataset
    G, embedding_matrix = utils.read_file(args.datadir, args.dataset) # read graph and embeddings

    # dataloaders
    train_set, val_set, test_set = utils.get_dataset(G)
    dataloaders = utils.get_dataloader(train_set, val_set, test_set, args)
    
    # build model
    args.time_encoder_maxt = G.maxt # from the dataset
    args.time_encoder_args = {'maxt': args.time_encoder_maxt, 'rows': args.time_encoder_rows, 'dimension': args.time_encoder_dimension}
    model = get_model(G, embedding_matrix, args, logger)

    if args.eval != '':
        assert len(args.eval) == 19, 'Must be a time_str with format YEAR_MO_DA_HO_MI_SE' # e.g., 2021_02_26_15_14_38
        assert args.state_dict is not None, "Must indict state_dict filename"
        evaluate_state_dict(model, dataloaders, args, logger, time_str=args.eval, parallel=args.parallel_eval)
    else:
        train_model(model, dataloaders, args, logger)

if __name__ == "__main__":
    main()
