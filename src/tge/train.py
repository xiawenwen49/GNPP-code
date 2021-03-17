import torch
import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm
from torch_geometric.data import Data
from tge.model import HarmonicEncoder, PositionEncoder
from xww.utils.training import get_device, get_optimizer, Recorder
from xww.utils.multiprocessing import MultiProcessor


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
        """
        assert t >= T.max(), 't should >= T.max(), i.e., t_n'
        lambda_t = self.lambdaf(u, v, T, t, **kwargs)
        integral = self.lambdaf.integral(u, v, T, T[-1], t, **kwargs)
        f_t = lambda_t * torch.exp(-1 * integral)
        return f_t
    
    def predict(self, u, v, T, **kwargs):
        """ next event time expectation by MC
        """
        device = u.device
        intervals = T[1:] - T[:-1]
        max_interval = torch.max(intervals)
        T_interval = 2 * max_interval
        t_end = T[-1] + T_interval

        # import ipdb; ipdb.set_trace()
        counter = 0
        test_value = self(u, v, T, t_end, **kwargs)
        while (test_value < 1e-3 or test_value > 0.1) and t_end > T[-1]: # >1e6 for overflow
            if test_value < 1e-3: # shrink
                t_end = (T[-1] + t_end)/2
            elif test_value > 0.1: # expand
                t_end = t_end + (t_end - T[-1])/2
            counter = counter + 1
            if counter > 20:
                break
            test_value = self(u, v, T, t_end, **kwargs)

        T_interval = t_end - T[-1]
        size = 100
        # import ipdb; ipdb.set_trace()
        t_samples = torch.linspace(T[-1]+T_interval/size, T[-1] + T_interval, size).to(device)
        values = torch.zeros_like(t_samples)
        for i, t in enumerate(t_samples):
            f_t = self(u, v, T, t, **kwargs)
            values[i] = f_t.data.squeeze() # used for MC integral of t*f(t) dt
        
        values = (T_interval/size) * values # it should be probability now.
        values = values / (values.sum() + 1e-6) # normalilze, the result of this step should be similar with that of the former step.
        estimated_expectation = (values * t_samples).sum()
        return estimated_expectation



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
        integral = torch.clamp(integral, -100, 100)
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
        emb_t = emb_t.view(t.reshape(1, -1).shape[1], 1, time_encoding_dim)
        emb_T = emb_T.view(T.reshape(1, -1).shape[1], 1, time_encoding_dim)

        emb_t = emb_t.to(dtype=torch.float32, device=t.device) # https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
        emb_T = emb_T.to(dtype=torch.float32, device=t.device)
 
        atten_output, atten_output_weight = self.model.AttenModule(emb_t, emb_T, emb_T) # query, key, value. 
        atten_output = atten_output.squeeze(1)

        batch = kwargs['batch']
        uv_agg = self.model(batch, t.mean()) # t.mean() is an appro.
        uv_agg = uv_agg.flatten().reshape((1, -1)).repeat((t.numel(), 1))
        atten_output = torch.cat([atten_output, uv_agg], axis=1)

        alpha = soft_plus(1, (self.model.alpha(u) * self.model.alpha(v)).sum() ) # alpha should > 0
        value = soft_plus(alpha, atten_output @ self.model.W_H ) # TODO: to add gnn representation
        return value
    
    def integral(self, u, v, T, t_start, t_end, N=10, **kwargs):
        """ by MC """
        if t_start is None:
            t_start = 0
        assert t_end >= t_start
        assert t_start >= T.max(), 't_start should >= T.max(), i.e., t_n'
        device = model_device(self.model)
        points = torch.rand(N).to(device) * (t_end - t_start) + t_start
        # points = points.to(device)
        values = self(u, v, T, points, batch=kwargs['batch']) # COMPLETED: to debug
        return (t_end - t_start)/N * values.sum()

def soft_plus(phi, x):
    return phi * torch.log(1 + torch.exp( x / phi ))

def log_likelihood(u, v, lambdaf, T, **kwargs):
    """
    Args:
        u, v: node pair
        lambdaf: conditional intensity function
        T: observed hsitory
    """
    # ll = 0
    ll1 = 0
    integral = 0
    for i, t in enumerate(T[1:]):
        ll1 = ll1 + torch.log( lambdaf(u, v, T[:i+1], t, batch=kwargs['batch']) )
        integral = integral + lambdaf.integral(u, v, T[:i+1], T[i], T[i+1], batch=kwargs['batch'])
        pass
    ll = ll1 - integral # log likelihood
    return ll

def constant_1(model, hid_u, hid_v, emb_u, emb_v):
    return model.W_S_( torch.cat([hid_u, hid_v]).view(1, -1) ) + model.W_E_( torch.cat([emb_u, emb_v]).view(1, -1) )

def model_device(model):
    return next(model.parameters()).device

def criterion(model, batch, **kwargs):
    # COMPLETED: batch now has no uv, t
    u, v = batch.nodepair
    T = batch.T
    # uv, t = batch
    # u, v = uv[0]
    # T = t[0]

    lambdaf = kwargs['lambdaf']
    ll = log_likelihood(u, v, lambdaf, T, batch=batch)
    loss = -1 * ll # negative log likelihood
    return loss

# COMPLETED: no gnn, only self-attention for lambda function
# TODO: how to accelerate?/parallelization?
def optimize_epoch(model, optimizer, train_loader, args, logger, **kwargs):
    model.train() # train mode
    device = get_device(args.gpu)
    model = model.to(device)
    batch_counter = 0
    recorder = []

    for i, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
        if isinstance(batch, list):
            batch = list(map(lambda x: x.to(device), batch)) # set device
        elif isinstance(batch, Data):
            batch = batch.to(device)

        lambdaf = kwargs.get('lambdaf')
        try:
            loss = criterion(model, batch, lambdaf=lambdaf)
            batch_counter += 1
            if batch_counter % 8 == 0: # update model parameters for one step
                loss.backward()
                # import ipdb; ipdb.set_trace()
                optimizer.step()
                optimizer.zero_grad()
                # model.reset_hidden_rep() # reset hidden representation matrix
                # model.clip_time_encoder_weight()
                batch_counter = 0
                torch.cuda.empty_cache()
            else:
                # loss.backward(retain_graph=True)
                loss.backward() # NOTE: only necessary when cached hidden rep of gnn model is used
            recorder.append(loss.item())
        except Exception as e:
            if 'CUDA out of memory' in e.args[0]:
                logger.info(f'CUDA out of memory for batch {i}, skipped.')
            else: 
                raise

    # model.prepare_hidden_rep() # update hidden_rep, for next optimize epoch or for evaluation
    return np.mean(recorder)

def batch_evaluate(model, batch, **kwargs):
    device = model.W_S_.weight.device
    if isinstance(batch, list):
        batch = list(map(lambda x: x.to(device), batch)) # set device
    elif isinstance(batch, Data):
        batch = batch.to(device)

    # uv, T_all = batch
    # u, v = uv[0] # only one sample
    # T_all = T_all[0]
    u, v = batch.nodepair
    T_all = batch.T
    T, t = T_all[:-1], T_all[-1]
   

    lambdaf = kwargs.get('lambdaf')
    ff = ConditionalDensityFunction(lambdaf)
    # hidden_reps = model(batch)
    # embeddings = model.embedding(u), model.embedding(v)
    # loss = criterion(hidden_reps, embeddings, batch, model) # tensor
    # pred = predict(model, u, v, T)
    loss = criterion(model, batch, lambdaf=lambdaf) # tensor
    pred = ff.predict(u, v, T, batch=batch)
    se = (pred - t)**2 # item
    se = se.cpu().item()
    abs = np.abs((pred - t).cpu().item())
    abs_ratio = abs / ((T_all[1:] - T_all[:-1]).mean().cpu().item() + 1e-6) # now the denominator is average interval
    # import ipdb; ipdb.set_trace()
    return {'loss': loss.item(), 'se': se, 'abs_ratio': abs_ratio}

def evaluate(model, train_loader, test_loader, args, logger, **kwargs):
    model.eval() # eval mode
    device = get_device(args.gpu)
    model = model.to(device)

    loss_recorder = []
    se_recorder = [] # square error
    abs_ratio_recorder = [] # abs(pred-target) / target
    
    if kwargs.get('parallel', None) is True:
        # batch_evaluater = batch_evaluate_wraper(model, time_encoder, batch_evaluate)
        mp = MultiProcessor(40)
        # result = mp.run_imap(batch_evaluater, test_loader)
        result = mp.run_queue(batch_evaluate, test_loader, model=model, **kwargs)
        result = pd.DataFrame(result)
        loss_recorder = result['loss'].tolist()
        se_recorder = result['se'].tolist()
        abs_ratio_recorder = result['abs_ratio'].tolist()
    else:
        for batch_test in tqdm(test_loader, total=len(test_loader)):
            batch_result = batch_evaluate(model, batch_test, lambdaf=kwargs['lambdaf'])
            loss_recorder.append(batch_result['loss'])
            se_recorder.append(batch_result['se'])
            abs_ratio_recorder.append(batch_result['abs_ratio'])
    
    return {'loss': np.mean(loss_recorder), 'rmse': np.sqrt(np.mean(se_recorder)), 'abs_ratio': np.mean(abs_ratio_recorder)}
        
# COMPLETED: add recorder to save model state, then check the predic function.
def train_model(model, dataloaders, args, logger):
    train_loader, val_loader, test_loader = dataloaders
    # time_encoder = HarmonicEncoder(args.time_encoder_dimension)
    optimizer = get_optimizer(model, args.optimizer, args.lr, args.l2)
    lambdaf = AttenIntensity(model)
    recorder = Recorder({'loss': 0}, args.checkpoint_dir, args.dataset, args.time_str)
    # results = evaluate(model, train_loader, test_loader, args, logger) # evaluate with random model
    # logger.info(f"Without training, test_loss: {results['loss']:.4f}, test_rmse: {results['rmse']:.4f}, test_abs_ratio: {results['abs_ratio']:.4f}")
    for i in range(args.epochs):
        train_loss = optimize_epoch(model, optimizer, train_loader, args, logger, epoch=i, lambdaf=lambdaf)
        recorder.append_model_state(model.state_dict())
        recorder.save_model(latest=True)
        results = evaluate(model, train_loader, test_loader, args, logger, epoch=i, lambdaf=lambdaf)
        logger.info(f"Epoch {i}, train_loss: {train_loss:.4f}, test_loss: {results['loss']:.4f}, test_rmse: {results['rmse']:.4f}, test_abs_ratio: {results['abs_ratio']:.6f}")
        recorder.append_full_metrics({'loss': train_loss}, 'train')
        recorder.append_full_metrics({'loss': results['loss'], 'rmse': results['rmse'], 'abs_ratio': results['abs_ratio']}, 'test')
        recorder.save_record()
    logger.info(f"Training finished, best test loss: , best test rmse: , btest abs_ratio: ")

# COMPLETED: debug evaluate_state_dict()
# COMPLETED: optimize/check loss and predict
def evaluate_state_dict(model, dataloaders, args, logger, **kwargs):
    """ evaluate using an existing time_str checkpoint """
    time_str = args.eval
    state_dict_filename = args.state_dict
    device = get_device(args.gpu)
    recorder = Recorder(minmax={}, checkpoint_dir=args.checkpoint_dir, dataset=args.dataset, time_str=time_str)
    # model_states = recorder.load_model(time_str, state_dict_filename)
    # state_dict = model_states.get(state_dict_filename, None)
    state_dict = recorder.load_model(time_str, state_dict_filename)
    assert state_dict is not None, f"No {state_dict_filename}.state_dict in {args.dataset}/{time_str}/ dir"
    # import ipdb; ipdb.set_trace()
    model.load_state_dict(state_dict)
    model = model.to(device)
    lambdaf = AttenIntensity(model)
    logger.info(f'Evaluate using {time_str} state_dict.')
    train_loader, val_loader, test_loader = dataloaders
    results = evaluate(model, train_loader, test_loader, args, logger, lambdaf=lambdaf)
    logger.info(f"Eval, test_loss: {results['loss']:.4f}, test_rmse: {results['rmse']:.4f}, test_abs_ratio: {results['abs_ratio']:.4f}")









