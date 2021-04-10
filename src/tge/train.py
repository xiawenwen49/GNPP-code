import torch
import numpy as np
import pandas as pd
import warnings
import os
from pathlib import Path
from tqdm import tqdm
from torch_geometric.data import Data
from tge.model import HarmonicEncoder, PositionEncoder
from xww.utils.training import get_device, get_optimizer, Recorder
from xww.utils.multiprocessing import MultiProcessor
from xww.utils.profile import profiler


class ConditionalIntensityFunction():
    """ the conditional intensity function \lambda, if given observed history """
    def __init__(self, model):
        self.model = model

    def __call__(self, u, v, T, t):
        """ \lambda^{u, v}(t|T_{u, v}) 
        Args: 
            u, v: node pair
            t: target time for value calculation
            T: observed timestamps before t.
        """
        pass

    def integral(self, u, v, T, t_start, t_end):
        """ integral from t_start to t_end. By MC or by closed-form.
        Args: 
        Args:
            u, v: the node pair
            t_start: integral start
            t_end: integral end
            T: observed timestamps `before` t_start. max(T) <= t_start
        """
        pass

# COMPLETED: f function
# COMPLETED: predict function
class ConditionalDensityFunction():
    """ the conditional density function f(t) """
    def __init__(self, lambdaf): # can not use self.lambda as variable name. `lambda` cannot appear as a variabel
        """ lambdaf is a conditional intensity function instance """
        # self.model = model
        self.lambdaf = lambdaf
    
    def __call__(self, u, v, T, t, **kwargs):
        """ f^{u, v}(t|T_{u, v}) = \lambda^{u, v}(t|T_{u, v}) * \int_{0}^{max(T_{u, v})} \lambda(s) ds
        COMPLETED: to support batch computation of f(t), where t is a batch.
        make sure t.min() >= T.max().

        Args: 
            T: timestamps history, a batch
            t: to support t as a batch
        """
        # assert t.min() >= T.max(), 't should >= T.max(), i.e., t_n'
        lambda_t = self.lambdaf(u, v, T, t, **kwargs)
        integral = self.lambdaf.integral(u, v, T, T[-1], t, **kwargs) # NOTE: integral() batch mode is the KEY.
        # integral = torch.clip(integral, -60, 60)

        lambda_t = lambda_t.reshape((-1, 1))
        integral = integral.reshape((-1, 1))
        f_t = lambda_t * torch.exp(-1 * integral)
        return f_t
    
    def predict(self, u, v, T, **kwargs):
        """ next event time expectation by MC
        """
        device = u.device
        intervals = T[1:] - T[:-1]
        max_interval = intervals.mean() + intervals.var()
        # T_interval = 2 * max_interval
        t_end = T[-1] + max_interval

        counter = 0
        test_value = self(u, v, T, t_end, **kwargs)
        while (test_value < 1e-3 or test_value > 0.1) and t_end > T[-1]: # >1e6 for overflow
            if test_value < 1e-3 and t_end > T[-1]+1e-2: # shrink
                t_end = (T[-1] + t_end)/2
            elif test_value > 0.1: # expand
                t_end = t_end + (t_end - T[-1])/2
            counter = counter + 1
            if counter > 20:
                break
            test_value = self(u, v, T, t_end, **kwargs)

        T_interval = torch.abs(t_end - T[-1])
        size = 15
        t_samples = torch.linspace(T[-1]+T_interval/size , T[-1] + T_interval, size, device=device) # NOTE: in order
        # NOTE: here it supports batch mode of t_samples
        values = self(u, v, T, t_samples, **kwargs)
        values = (T_interval/size) * values # it should be probability now.

        values = values / (values.sum() + 1e-6) # normalilze, the result of this step should be similar with that of the former step.
        t_samples = t_samples.reshape(-1, 1)
        assert t_samples.shape == values.shape, "t_samples.shape must == values.shape"
        estimated_expectation = (values * t_samples).sum()
        return estimated_expectation


def predict(model, batch):
    T_all = batch.T
    t = T_all[-2]
    lambdav, pred = model(batch)
    return lambdav, pred

class HarmonicIntensity(ConditionalIntensityFunction):
    def __init__(self, model):
        super(HarmonicIntensity, self).__init__(model)
        pass
        
    def __call__(self, u, v, t, T):
        self.model.eval()
        hid_u, hid_v, emb_u, emb_v = self.model.hidden_rep[u], self.model.hidden_rep[v], self.model.embedding(u), self.model.embedding(v)
        C1 = constant_1(self.model, hid_u, hid_v, emb_u, emb_v)
        lambda_t = torch.exp(C1) + self.model.time_encoder.cos_encoding_mean(t - T, u, v).sum()
        # lambda_t = lambda_t.item()
        return lambda_t
    
    def integral(self, u, v, T, t_start, t_end):
        """ by closed-form """
        # self.model.eval()
        # import ipdb; ipdb.set_trace()
        # lambda_t = self.lambdaf(u, v, t_end, T)
        hid_u, hid_v, emb_u, emb_v = self.model.hidden_rep[u], self.model.hidden_rep[v], self.model.embedding(u), self.model.embedding(v)
        C1 = constant_1(self.model, hid_u, hid_v, emb_u, emb_v)
        integral = (t_end - T[-1]) * torch.exp(C1) + torch.sum(self.model.time_encoder.sin_divide_omega_mean(t_end - T, u, v) - self.model.time_encoder.sin_divide_omega_mean(T[-1] - T, u, v))
        integral = integral + len(T) * self.model.time_encoder.alpha[u][v] * (t_end - T[-1]) # NOTE: new formula
        # integral = integral.item()
        # integral = torch.clamp(integral, -100, 100)
        return integral

class AttenIntensity(ConditionalIntensityFunction):
    def __init__(self, model):
        super(AttenIntensity, self).__init__(model)
    
    def __call__(self, u, v, T, t, **kwargs):
        # COMPLETED: to support t as a batch
        # COMPLETED: add GNN in the lambda^{u, v}(t|H^{u,v}) computation
        assert isinstance(self.model.time_encoder, PositionEncoder), 'here the time encoder should be PositionEncoder'
        time_encoding_dim = self.model.time_encoder_args['dimension']
        emb_t = self.model.time_encoder(t)
        emb_T = self.model.time_encoder(T)
        emb_t = emb_t.view(t.numel(), 1, time_encoding_dim)
        emb_T = emb_T.view(T.numel(), 1, time_encoding_dim)

        emb_t = emb_t.to(dtype=torch.float32, device=t.device) # https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
        emb_T = emb_T.to(dtype=torch.float32, device=t.device)
 
        atten_output, atten_output_weight = self.model.AttenModule(emb_t, emb_T, emb_T) # query, key, value.
        atten_output = atten_output.squeeze(1)

        batch = kwargs['batch']
        # uv_agg = self.model(batch, t.mean()) # t.mean() is an appro.
        uv_agg = self.model(batch, t.min()) # t.min() supports t as a batch, and is somewhat reasonable
        uv_agg = uv_agg.reshape((1, -1)).repeat((t.numel(), 1))
        atten_output = torch.cat([atten_output, uv_agg], axis=1) # COMPLETED: to add gnn representation

        # alpha = relu_plus(1e-1, torch.dot(self.model.alpha(u), self.model.alpha(v)) ) # ensure alpha > 0
        alpha = torch.nn.functional.sigmoid( torch.dot(self.model.alpha(u), self.model.alpha(v)) )
        value = soft_plus(alpha, atten_output @ self.model.W_H ) # ensure >= 0
        return value
    
    def integral(self, u, v, T, t_start, t_end, N=10, **kwargs):
        """ approximate the integral by MC 
        Args:
            t_start: integral start, not a batch
            t_end: integral end, -> to support t_end as a batch. NOTE: this integral batch mode is the KEY of batch mode for predict() of f().
        """
        if t_start is None:
            t_start = 0
        t_end = t_end.reshape(-1) # if t = tensor(1.0), to change to ==> tensor([1.0])
        assert t_end.min() >= t_start

        device = u.device
        # points = [torch.linspace(t_start+1e-6, t_end[i], N, device=device) if i==0 else torch.linspace(t_end[i-1], t_end[i], N, device=device) for i in range(len(t_end)) ] # all interpoloted points[t_start, ..., t_end[0],  ]
        points = [torch.rand(N, device=device)*(t_end[i]-t_start)+t_start if i==0 else torch.rand(N, device=device)*(t_end[i]-t_end[i-1])+t_end[i] for i in range(len(t_end)) ]
        points = torch.cat(points).to(device)
        # import ipdb; ipdb.set_trace()
        values = self(u, v, T, points, batch=kwargs['batch']) # NOTE: batch mode for interploted points is the KEY.
        values = values.reshape((-1, 1))

        intervals = torch.cat([t_start.reshape(-1), t_end])
        intervals = (intervals[1:] - intervals[:-1])/N
        intervals = intervals.reshape(-1, 1).repeat(1, N).reshape(-1, 1)
        assert intervals.shape == values.shape, "intervals.shape should == values.shape"

        values = values * intervals # FIXED: shape problem
        values = values.cumsum(dim=0)
        index = torch.arange(1, t_end.numel()+1, device=device) * N - 1
        return_values = values[index]
        return return_values

# class AttenIntensityE2N(AttenIntensity):
#     def __init__(self, model):
#         super(AttenIntensityE2N, self).__init__(model)
    
#     def __call__(self, u, v, T, t, **kwargs):
#         batch = kwargs['batch']
#         output, _ = self.model(batch, t)
#         value = output

#         return value


def soft_plus(phi, x):
    res = torch.where(x/phi < 20, 1e-6 + phi * torch.log1p( torch.exp(x/phi) ), x ) # 1e-6 is important, to make sure lambda > 0
    return res


def relu_plus(phi, x):
    return phi + torch.nn.functional.relu(x)


def compute_integral(model, batch, t_start, t_ends, N=200):
    t_ends = t_ends.reshape(-1)
    assert t_ends.min() >= t_start
    device = t_start.device
    points = [torch.rand(N, device=device)*(t_ends[i]-t_start)+t_start if i==0 else torch.rand(N, device=device)*(t_ends[i]-t_ends[i-1])+t_ends[i] for i in range(len(t_ends)) ]
    points = torch.cat(points).to(device)
    values, atten_output = model(batch, points)
    values = values.reshape((-1, 1))

    intervals = torch.cat([t_start.reshape(-1), t_ends])
    intervals = (intervals[1:] - intervals[:-1])/N
    intervals = intervals.reshape(-1, 1).repeat(1, N).reshape(-1, 1)
    assert intervals.shape == values.shape, "intervals.shape should == values.shape"

    values = values * intervals
    values = values.cumsum(dim=0)
    index = torch.arange(1, t_ends.numel()+1, device=device) * N - 1
    return_values = values[index]
    return return_values


def constant_1(model, hid_u, hid_v, emb_u, emb_v):
    return model.W_S_( torch.cat([hid_u, hid_v]).view(1, -1) ) + model.W_E_( torch.cat([emb_u, emb_v]).view(1, -1) )

def model_device(model):
    return next(model.parameters()).device


def criterion(model, batch, **kwargs):
    u, v = batch.nodepair
    T_all = batch.T

    intervals = T_all[2:] - T_all[1:-1]
    lambdav, atten_output = model(batch, T_all[1:])
    pred = model.Linear_pred(atten_output)

    # neg likelihood
    ll0 = torch.log( 1e-6 + lambdav ).sum()
    integral = compute_integral(model, batch, T_all[1], T_all[-1], N=200)
    nll = -1 * (ll0 - integral)

    # time pred mse
    time_error = torch.mean(torch.square(pred[:-1] - intervals))

    # loss = nll + 0.0001 * time_error

    loss = nll 

    return loss, -nll, time_error



# COMPLETED: how to accelerate?/parallelization? -> train is reasonable, then evaluate? ==> make lambda.integral (so f.__call__) supporting batch mode
@profiler(Path(os.path.dirname(os.path.abspath(__file__)) ).parent.parent/'log'/'profile')
def optimize_epoch(model, optimizer, train_loader, args, logger, **kwargs):
    model.train() # train mode
    device = get_device(args.gpu)
    model = model.to(device)
    batch_counter = 0
    recorder = {'loss': [], 'll':[], 'rmse': []}

    for i, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc='- [training]', leave=False):
        if isinstance(batch, list):
            batch = list(map(lambda x: x.to(device), batch)) # set device
        elif isinstance(batch, Data):
            batch = batch.to(device)

        try:
            loss, ll, time_mse = criterion(model, batch)
            loss.backward()

            # loss.backward(retain_graph=True)
            # for p in model.parameters():




            batch_counter += 1
            if batch_counter % 8 == 0: # update model parameters for one step
                optimizer.step()
                optimizer.zero_grad()
                batch_counter = 0
            # else:
                # loss.backward(retain_graph=True) # NOTE: only necessary when cached hidden rep of gnn model is used
          
            recorder['loss'].append(loss.cpu().item())
            recorder['ll'].append(ll.cpu().item())
            recorder['rmse'].append(time_mse.sqrt().cpu().item())
        except Exception as e:
            if 'CUDA out of memory' in e.args[0]:
                continue
            else:
                raise
        
        if args.debug and i == 0:
            break
    recorder['loss'] = np.mean(recorder['loss'])
    recorder['ll'] = np.mean(recorder['ll'])
    recorder['rmse'] = np.mean(recorder['rmse']) # 原来是np.sum()？？？？？？？
    return recorder


def evaluate_batch(model, batch, **kwargs):
    device = model.W_H.device
    if isinstance(batch, list):
        batch = list(map(lambda x: x.to(device), batch)) # set device
    elif isinstance(batch, Data):
        batch = batch.to(device)

    # u, v = batch.nodepair
    T_all = batch.T
    mean_interval = (T_all[1:] - T_all[:-1]).mean().cpu().item()

    # lambdaf = kwargs.get('lambdaf')
    # ff = ConditionalDensityFunction(lambdaf)

    # evaluate metrics
    loss, ll, time_error = criterion(model, batch) # mse
    loss = loss.cpu().item()
    ll = ll.cpu().item()

    rmse = time_error.sqrt().cpu().item() # rmse
    abs_ratio = rmse / (mean_interval + 1e-6)
    return {'loss': loss, 'rmse': rmse, 'll': ll, 'abs_ratio': abs_ratio}



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
            with torch.no_grad():
                batch_result = evaluate_batch(model, batch_test, debug=args.debug)
            for key in batch_result.keys():
                batch_results[key].append(batch_result[key])
            
            if args.debug and i == 100:
                break
    for key in batch_results.keys():
        batch_results[key] = np.mean(batch_results[key])
    
    return batch_results


def train_model(model, dataloaders, args, logger):
    train_loader, val_loader, test_loader = dataloaders
    optimizer = get_optimizer(model, args.optim, args.lr, args.l2)
    # lambdaf = AttenIntensityE2N(model) # lambdaf = AttenIntensity(model)
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
        logger.info(f"   - (testing.)    loss: {eval_results['loss']:.5f}    rmse: {eval_results['rmse']:.5f}    ll: {eval_results['ll']:.5f}    abs_ratio: {eval_results['abs_ratio']:.5f}")
        # logger.info(f"   - (best res)    loss: {eval_results['loss']:.6f}    rmse: {eval_results['rmse']:.5f}    abs_ratio: {eval_results['abs_ratio']:.5f}")
        
        
    logger.info(f"Training finished, best test loss: , best test rmse: , btest abs_ratio: ")



def evaluate_state_dict(model, dataloaders, args, logger, **kwargs):
    """ evaluate using an existing time_str checkpoint """
    time_str = args.eval
    state_dict_filename = args.state_dict
    device = get_device(args.gpu)
    recorder = Recorder(minmax={}, checkpoint_dir=args.checkpoint_dir, dataset=args.dataset, time_str=time_str)
    state_dict_dir = args.checkpoint_dir/args.dataset/time_str/f'{args.state_dict}.state_dict'
    state_dict = recorder.load_model(state_dict_dir)
    assert state_dict is not None, f"No {state_dict_filename}.state_dict in {args.dataset}/{time_str}/ dir"
    model.load_state_dict(state_dict)
    model = model.to(device)
    # lambdaf = AttenIntensityE2N(model) # AttenIntensity
    train_loader, val_loader, test_loader = dataloaders
    results = evaluate_epoch(model, test_loader, args, logger, recorder=recorder)
    recorder.save_record()
    
    logger.info(f" [ Eval using {time_str} ] ")
    logger.info(f"   loss: {results['loss']:.4f},   rmse: {results['rmse']:.4f},    abs_ratio: {results['abs_ratio']:.4f}")
    return lambdaf, model









