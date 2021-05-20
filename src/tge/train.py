import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from torch_geometric.data import Data
from xww.utils.training import get_device, get_model_device, get_optimizer, Recorder
from xww.utils.multiprocessing import MultiProcessor


def soft_plus(phi, x):
    res = torch.where(x/phi < 20, 1e-6 + phi * torch.log1p( torch.exp(x/phi) ), x ) # 1e-6 is important, to make sure lambda > 0
    return res

def relu_plus(phi, x):
    return phi + torch.nn.functional.relu(x)

def compute_integral(model, batch, t_start, t_end, N=200):
    assert t_end >= t_start
    device = t_start.device
    points = torch.rand(N+2, device=device)[1:-1]*(t_end-t_start)+t_start
    points = points.to(device)

    values, atten_output = model(batch, points)
    values = values.reshape((-1, 1))

    intervals = (t_end - t_start)/N
    intervals = intervals.reshape(-1, 1).repeat(1, N).reshape(-1, 1)
    assert intervals.shape == values.shape, "intervals.shape should == values.shape"

    values = values * intervals
    return_values = values.sum()
    return return_values


def constant_1(model, hid_u, hid_v, emb_u, emb_v):
    return model.W_S_( torch.cat([hid_u, hid_v]).view(1, -1) ) + model.W_E_( torch.cat([emb_u, emb_v]).view(1, -1) )

def model_device(model):
    return next(model.parameters()).device


def criterion(model, batch, **kwargs):
    # import ipdb; ipdb.set_trace()
    T_all = batch.T

    intervals = T_all[2:] - T_all[1:-1]
    lambdav, atten_output = model(batch, T_all[1:])
    pred = model.Linear_pred(atten_output)

    # neg likelihood
    ll0 = torch.log( 1e-9 + lambdav ).sum()
    integral = compute_integral(model, batch, T_all[0], T_all[-1], N=100)
    nll = -1 * (ll0 - integral)

    # time mse
    time_error = torch.mean(torch.square(pred[:-1] - intervals))
    
    loss = nll + 0.001 * time_error
    return loss, -nll, time_error

def criterion_gnn(model, batch, **kwargs):
    y = batch.y
    pred = model(batch)
    loss = torch.mean(torch.square(pred - y))
    time_error = torch.mean(torch.square(pred - y)) # mse
    ll = torch.tensor(1.0)

    return loss, ll, time_error



# @profiler(Path(os.path.dirname(os.path.abspath(__file__)) ).parent.parent/'log'/'profile')
def optimize_epoch(model, optimizer, train_loader, args, logger, **kwargs):
    model.train() # train mode
    device = get_device(args.gpu)
    model = model.to(device)
    batch_counter = 0
    recorder = {'loss': [], 'll':[], 'rmse': []}

    if args.model in ['GAT', 'GraphSAGE']:
        criter = criterion_gnn # for gnn model run
    else:
        criter = criterion

    for i, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc='- [training]', leave=False):
        if isinstance(batch, list):
            batch = list(map(lambda x: x.to(device), batch)) # set device
        elif isinstance(batch, Data):
            batch = batch.to(device)

        try:
            loss, ll, time_mse = criter(model, batch)
            loss.backward()

            batch_counter += 1
            if batch_counter % 8 == 0: # update model parameters for one step
                optimizer.step()
                optimizer.zero_grad()
                batch_counter = 0
          
            recorder['loss'].append(loss.cpu().item())
            recorder['ll'].append(ll.cpu().item())
            recorder['rmse'].append(time_mse.sqrt().cpu().item())
        except Exception as e:
            if 'CUDA out of memory' in e.args[0]:
                continue
            else: raise
        
        if args.debug and i == 0:
            break
    recorder['loss'] = np.mean(recorder['loss'])
    recorder['ll'] = np.mean(recorder['ll'])
    recorder['rmse'] = np.mean(recorder['rmse'])
    return recorder


def evaluate_batch(model, batch, **kwargs):
    device = get_model_device(model)
    if isinstance(batch, list):
        batch = list(map(lambda x: x.to(device), batch)) # set device
    elif isinstance(batch, Data):
        batch = batch.to(device)

    if kwargs['args'].model in ['GAT', 'GraphSAGE']:
        criter = criterion_gnn # for gnn model run
    else:
        criter = criterion
        
    # evaluate metrics
    loss, ll, time_error = criter(model, batch) # mse
    loss = loss.cpu().item()
    ll = ll.cpu().item()

    rmse = time_error.sqrt().cpu().item() # rmse
    return {'loss': loss, 'rmse': rmse, 'll': ll}



# @profiler(Path(os.path.dirname(os.path.abspath(__file__)) ).parent.parent/'log'/'profile')
def evaluate_epoch(model, test_loader, args, logger, **kwargs):
    model.eval() # eval mode
    device = get_device(args.gpu)
    model = model.to(device)

    batch_results = {'loss': [], 'rmse': [], 'll': [], 'abs_ratio': []}
    if kwargs.get('parallel', None) is not None:
        mp = MultiProcessor(40)
        result = mp.run_queue(evaluate_batch, test_loader, model=model, **kwargs)
        result = pd.DataFrame(result)
    else:
        for i, batch_test in tqdm(enumerate(test_loader), total=len(test_loader), desc='- [testing]', leave=False):
            try:
                with torch.no_grad():
                    batch_result = evaluate_batch(model, batch_test, debug=args.debug, args=args)
                for key in batch_result.keys():
                    batch_results[key].append(batch_result[key])
            except Exception as e:
                if 'CUDA out of memory' in e.args[0]:
                    continue
                else: raise
            
            if args.debug and i == 100:
                break
    for key in batch_results.keys():
        batch_results[key] = np.mean(batch_results[key])
    
    return batch_results


def train_model(model, dataloaders, args, logger):
    train_loader, val_loader, test_loader = dataloaders
    optimizer = get_optimizer(model, args.optim, args.lr, args.l2)
    recorder = Recorder({'loss': 0}, args.checkpoint_dir, args.dataset, args.time_str)
    for i in range(args.epochs):
        train_results = optimize_epoch(model, optimizer, train_loader, args, logger, epoch=i)
        recorder.save_model(model, i=i)
        eval_results = evaluate_epoch(model, test_loader, args, logger, epoch=i)
        
        recorder.append_full_metrics(train_results, 'train')
        recorder.append_full_metrics(eval_results, 'test')
        recorder.save_record()

        logger.info(f" [ Epoch {i} ] ")
        logger.info(f"   - (training)    loss: {train_results['loss']:.5f}    rmse: {train_results['rmse']:.5f}    ll: {train_results['ll']:.5f}")
        logger.info(f"   - (testing.)    loss: {eval_results['loss']:.5f}    rmse: {eval_results['rmse']:.5f}    ll: {eval_results['ll']:.5f}")
    # logger.info(f"   - (best res)    loss: {eval_results['loss']:.6f}    rmse: {eval_results['rmse']:.5f}    abs_ratio: {eval_results['abs_ratio']:.5f}")
        
        
    logger.info(f"Training finished, best test loss: , best test rmse: , btest abs_ratio: ")










