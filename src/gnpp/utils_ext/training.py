from turtle import pd
import torch
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
from collections import OrderedDict, deque

bcolors = {
    'header': '\033[95m',
    'blue': '\033[94m',
    'cyan': '\033[96m',
    'green': '\033[92m',
    'yellow': '\033[93m',
    'red': '\033[91m',
    'ENDC': '\033[0m',
    'bold': '\033[1m',
    'underline': '\033[4m' # not supported
}
    
def color_str(string, color='red'):
    color_suffix = bcolors.get(color, '')
    end = bcolors.get('ENDC')
    return color_suffix + string + end
    

def get_device(gpu_index):
    if gpu_index >= 0:
        try:
            assert torch.cuda.is_available(), 'cuda not available'
            assert gpu_index >= 0 and gpu_index < torch.cuda.device_count(), 'gpu index out of range'
            return torch.device('cuda:{}'.format(gpu_index))
        except AssertionError:
            return torch.device('cpu')
    else:
        return torch.device('cpu')

def get_optimizer(model, optim, lr, l2):
    if optim == 'adam':
        return torch.optim.Adam( filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=l2)
        # return torch.optim.AdamW( filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    elif optim == 'sgd':
        return torch.optim.SGD( filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=l2)
    else:
        raise NotImplementedError

def get_model_device(model):
    return next(model.parameters()).device



class Recorder(object):
    """
    """
    def __init__(self, minmax: Dict, checkpoint_dir: str, dataset: str, time_str: str, max_ckpts: int=5, args=None):
        """ 
        1. save metric results as csv files, at checkpoint_dir/dataset/time_str/record.csv
        2. save model state dict as .state_dict files, at checkpoint_dir/dataset/time_str/state.state_dict

        Args:
            minmax: for each metric, indicating the lower the better or the higher the better. 0: lower, 1: better.
                    e.g. {'loss': 0, 'acc': 1}
        

        Notes:
        self.full_metrics: {
            'train': [ {'loss': 0.1, 'mse': 0.2}, {...} ],
            'val': [ {'loss': 0.1, 'mse': 0.2}, {...} ],
            'test': [ {'loss': 0.1, 'mse': 0.2}, {...} ]

        }

        """
        self.minmax = minmax
        self.full_metrics = OrderedDict( {'train': [], 'val': [], 'test': []} )
        self.dataset = dataset
        self.time_str = time_str
        self.model_state = []
        self.checkpoint_dir = Path(checkpoint_dir) if not isinstance(checkpoint_dir, Path) else checkpoint_dir
        self.save_dir = self.checkpoint_dir/self.dataset/self.time_str
        self.saved_ckpts = deque() # only save a few recent checpoint dirs
        self.max_ckpts = max_ckpts
        self.dict_list = {}
        self.args = args
        self.args_saved = False

    def __getitem__(self, key):
        """ allow recorder['key'].append(value) """
        if self.dict_list.get(key, None) is None:
            self.dict_list[key] = []
        return self.dict_list[key]
    
    def save_args_to_json(self):
        import json
        # import ipdb; ipdb.set_trace()
        args = vars(self.args)
        fn = self.save_dir/'args.json'
        
        for k, v in args.items():
            if isinstance(v, Path):
                args[k] = str(v)
        try:
            with open(fn, 'w') as f:
                json.dump(args, f, indent=4)
        except:
            # print(color_str('Warning: dump args in Recorder() failed.', color='yellow'))
            pass

    def ensure_dir_exists(self, dir):
        dir = Path(dir)
        if not dir.exists():
            os.makedirs(dir)

    def append_metrics(self, metrics_results, name):
        """ 增加一个step的数据 
        Args:
            metric_results: dict of value of each metric, e.g., {'loss': 0.1, 'acc': 0.8}
            name: 'train'/'val'/'test'
        """
        assert name in ['train', 'val', 'test']
        self.full_metrics[name].append(metrics_results)
    
    def get_best_metric(self, name):
        """
        Args:
            name: 'train'/'val'/'test'. Which dataloader to return for.
        """
        if len(self.full_metrics[name]) == 0:
            return None, None
        df = pd.DataFrame( self.full_metrics[name] )
        best_metric = {}
        best_epoch = {}
        for key in df.keys():
            if key not in set(self.minmax.keys()):
                print( color_str(f"Warning: metric {key} does not designate minmax, default 0.", 'yellow') )
            data = np.array(df[key])
            best_metric[key] = np.max(data) if self.minmax.get(key, 0) else np.min(data)
            best_epoch[key] = np.argmax(data) if self.minmax.get(key, 0) else np.argmin(data)
        return best_metric, best_epoch
    
    def get_latest_metric(self, name):
        """
        Args:
            name: 'train'/'val'/'test'
        """
        latest_metric = self.full_metrics[name][-1]
        return latest_metric
    
    def append_model_state(self, state_dict):
        self.model_state.append(state_dict)

    def save_record(self):
        """ save train, val, test metric results 
        """
        self.ensure_dir_exists(self.save_dir)
        if not self.args_saved:
            self.save_args_to_json()
            self.args_saved = True
        

        filename = self.save_dir/'record.csv'
        concates = []
        for key, metric_list in self.full_metrics.items():
            if len(metric_list) > 0:
                df = pd.DataFrame(metric_list)
                concates.append( df.rename(columns=lambda x: key+'_'+x) ) # 加前缀, 'train', ['val'], 'test'
        if len(concates) > 0:
            df = pd.concat(concates, axis=1 ) # combine train, val, test
            df.to_csv(filename, float_format='%.10f', index=True, index_label='epoch' )

        # save other data in dict_list, if exist
        if len(self.dict_list) > 0:
            for key, item in self.dict_list.items():
                filename = self.save_dir/"{}.csv".format(key)
                df = pd.DataFrame(item)
                df.to_csv(filename)

    @staticmethod
    def load_record(checkpoint_dir, dataset, time_str):
        """ load a record.csv file """
        filename = Path(checkpoint_dir)/dataset/time_str/'record.csv'
        assert filename.exists(), 'No such a file: {}'.format(filename)
        df = pd.read_csv(filename)
        return df
    
    def save_model(self, model, i, interval=2):
        """
        Args:
            latest: save the latest state_dict,
                    Otherwise, 对于每一个metric指标，保存test set上这个指标的最好的epoch的model
        """
        self.ensure_dir_exists(self.save_dir/'state_dicts')
        if i % interval == 0:
            dir = self.save_dir/'state_dicts'/f'epoch{i}.state_dict'
            torch.save( model.state_dict(), dir)
            self.saved_ckpts.append(dir)
            # if len(self.saved_ckpts) > self.max_ckpts:
            #     oldest = self.saved_ckpts.popleft()
            #     os.remove(oldest)

    @staticmethod
    def load_model(state_dict_fn, map_location=None):
        """ readin one state_dict """
        if map_location is None:
            state_dict = torch.load(state_dict_fn)
        else:
            state_dict = torch.load(state_dict_fn, map_location=map_location)
        return state_dict
