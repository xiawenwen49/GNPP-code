import torch
import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm
from tge.model import HarmonicEncoder
from xww.utils.training import get_device, get_optimizer, Recorder
from xww.utils.multiprocessing import MultiProcessor

# np.seterr(all='raise')

# class Metric():
#     def __init__(self):
#         pass
#     def loss_metric(self):
#         pass

class ConditionalIntensityFunction():
    """ the conditional intensity function \lambda, if given observed history """
    def __init__(self, model):
        self.model = model
        # self.time_encoder = time_encoder

    def __call__(self, u, v, t, T):
        """ T is the observed times """
        self.model.eval()
        # self.time_encoder.eval()
        hid_u, hid_v, emb_u, emb_v = self.model.hidden_rep[u], self.model.hidden_rep[v], self.model.embedding(u), self.model.embedding(v)
        # C1 = self.model.W_S.view(1, -1) @ torch.cat([hid_u, hid_v]).view(-1, 1) + emb_u.view(1, -1) @ emb_v.view(-1, 1)
        C1 = constant_1(self.model, hid_u, hid_v, emb_u, emb_v)
        lambda_t = torch.exp(C1) + torch.sum(self.model.time_encoder.cos_encoding_mean(t - T))
        lambda_t = lambda_t.item()
        return lambda_t

class ConditionalDensityFunction():
    """ the conditional density function f(t) """
    def __init__(self, model):
        self.model = model
        self.lambdaf = ConditionalIntensityFunction(model) # can not use self.lambda as variable name. `lambda` cannot appear as a variabel
        # self.time_encoder = time_encoder # require \omega in time_encoder

    def __call__(self, u, v, t, T):
        """ can t be a batch? """
        self.model.eval()
        # self.time_encoder.eval()
        # import ipdb; ipdb.set_trace()
        lambda_t = self.lambdaf(u, v, t, T)
        hid_u, hid_v, emb_u, emb_v = self.model.hidden_rep[u], self.model.hidden_rep[v], self.model.embedding(u), self.model.embedding(v)
        C1 = constant_1(self.model, hid_u, hid_v, emb_u, emb_v)
        integral = (t - T[-1]) * torch.exp(C1) + torch.sum(self.model.time_encoder.sin_divide_omega_mean(t - T) - self.model.time_encoder.sin_divide_omega_mean(T[-1] - T))
        
        integral = integral + len(T) * self.model.time_encoder.alpha * (t - T[-1]) # NOTE: new formula
        
        integral = integral.item()
        
        # import ipdb; ipdb.set_trace()
        try:
            integral = np.clip(integral, -100, 100) # overflow
            f_t = lambda_t * np.exp(-1 * integral.item())
        except:
            # import ipdb; ipdb.set_trace()
            f_t = 0
        return f_t

def predict(model, u, v, T):
    """ predict the next event time for (u, v), given historical times T between (u, v) """
    f = ConditionalDensityFunction(model)

    intervals = T[1:] - T[:-1]
    max_interval = torch.max(intervals)
    # T_interval = 2*(T[-1] - T[0])
    T_interval = 200 * max_interval
    t_end = T[-1] + T_interval

    # import ipdb; ipdb.set_trace()
    counter = 0
    while (f(u, v, t_end, T) < 1e-4 or f(u, v, t_end, T) > 1e3) and t_end > T[-1]: # >1e6 for overflow
        t_end = (T[-1] + t_end)/2
        counter = counter + 1
        if counter > 100:
            break

    # t_start = T[-1]
    # while True:
    #     if f(u, v, t_end, T) < 1e-5:
    #         t_end = (t_start + t_end) / 2
    #     if f(u, v, t_end, T) > 1e-3:
    #         t_end = t_end + t_end - t_start
    #     if f(u, v, t_start, T) > 1e-3:
    #         t_start = (t_start + t_end) / 2
    #     if f(u, v, t_start, T) < 1e-5:
    #         t_start = min( t_start - (t_end - t_start), T[-1] )
    #
    #     counter = counter + 1
    #     if counter > 100:
    #         break
    # t_end = (t_start + t_end) / 2

    T_interval = (t_end - T[-1]).cpu().item()
    size = 10
    # t_samples = np.random.uniform(T[-1].cpu().item(), (T[-1]+T_interval).cpu().item(), size=size) # sampleing from a future interval -> too large variance
    t_samples = np.linspace(T[-1].cpu().item(), (T[-1]+T_interval).cpu().item(), num=size) + T_interval/size
    transformed_samples = []
    for t in t_samples:
        f_t = f(u, v, t, T) # item
        # transformed_samples.append( T_interval * f_t * t ) # importance sampling
        transformed_samples.append( f_t ) # used for area under f(t) for expectation
    
    transformed_samples = np.array(transformed_samples)
    transformed_samples = transformed_samples * (T_interval/size) # it should be probability now.
    transformed_samples = transformed_samples / (np.sum(transformed_samples) + 1e-6) # normalilze, the result of this step should be similar with that of the former step.
    estimated_expection = np.sum(transformed_samples * t_samples)

    # estimated_expection = np.mean(transformed_samples)
    return estimated_expection

def constant_1(model, hid_u, hid_v, emb_u, emb_v):
    # return model.W_S.view(1, -1) @ torch.cat([hid_u, hid_v]).view(-1, 1) + emb_u.view(1, -1) @ emb_v.view(-1, 1)
    return model.W_S_( torch.cat([hid_u, hid_v]).view(1, -1) ) + model.W_E_( torch.cat([emb_u, emb_v]).view(1, -1) )


def criterion(hidden_reps, embeddings, batch, model):
    """ compute negative log-likelihood """
    uv, t = batch
    u, v = uv[0] # assume batch_size=1
    T = t[0]
    # t = t - torch.min(t) # ?

    hid_u, hid_v = hidden_reps
    emb_u, emb_v = embeddings
    device = u.device
    # import ipdb; ipdb.set_trace()
    # event calculation
    C1 = constant_1(model, hid_u, hid_v, emb_u, emb_v)
    l1 = 0
    for i, t_i in enumerate(T):
        lambda_ti = torch.exp( C1 )
        if i >= 1:
            lambda_ti = lambda_ti + torch.sum(model.time_encoder.cos_encoding_mean(t_i - T[:i]) )
        
        l1 = l1 + (-torch.log(lambda_ti))
    
    # import ipdb; ipdb.set_trace()
    # integral calculation
    l2 = T[-1]*torch.exp( C1 ) 
    l2 = l2 + torch.sum( model.time_encoder.sin_divide_omega_mean( T[-1] - T[:-1] ) )
    # l2 = l2 + model.time_encoder.alpha * torch.sum( torch.arange(1, len(T)).to(device) * (T[1:]-T[:-1]) ) # NOTE: new formula, NOTE: comment amendment
    l2 = l2 + model.time_encoder.alpha * ( T[-1] - T[0] ) # NOTE: amendment
    l = l1 + l2
    return l

def optimize_epoch(model, optimizer, train_loader, args, logger, **kwargs):
    model.train() # train mode
    device = get_device(args.gpu)
    model = model.to(device)
    batch_counter = 0
    recorder = []

    for i, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
        batch = list(map(lambda x: x.to(device), batch)) # set device
        uv, t = batch
        u, v = uv[0]
        t = t[0]
        hidden_reps = model(batch)
        embeddings = model.embedding(u), model.embedding(v)
        loss = criterion(hidden_reps, embeddings, batch, model)

        batch_counter += 1
        if batch_counter % 16 == 0: # update model parameters for one step
            loss.backward()
            # import ipdb; ipdb.set_trace()
            optimizer.step()
            optimizer.zero_grad()
            model.reset_hidden_rep() # reset hidden representation matrix
            model.clip_time_encoder_weight()
            batch_counter = 0
        else:
            loss.backward(retain_graph=True)
        recorder.append(loss.item())

    model.prepare_hidden_rep() # update hidden_rep, for next optimize epoch or for evaluation
    return np.mean(recorder)

def batch_evaluate(batch, model):
    device = model.W_S_.weight.device
    batch = list(map(lambda x: x.to(device), batch)) # set device
    uv, T_all = batch
    u, v = uv[0] # only one sample
    T_all = T_all[0]
    T, t = T_all[:-1], T_all[-1]
    hidden_reps = model(batch)
    embeddings = model.embedding(u), model.embedding(v)
    loss = criterion(hidden_reps, embeddings, batch, model) # tensor
    pred = predict(model, u, v, T)
    se = (pred - t.cpu().item())**2 # item
    abs = np.abs(pred - t.cpu().item())
    abs_ratio = abs / (t - T[-1] + 1e-6).cpu().item()
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
        result = mp.run_queue(batch_evaluate, test_loader, model=model)
        result = pd.DataFrame(result)
        loss_recorder = result['loss'].tolist()
        se_recorder = result['se'].tolist()
        abs_ratio_recorder = result['abs_ratio'].tolist()
    else:
        for batch_test in tqdm(test_loader, total=len(test_loader)):
            batch_result = batch_evaluate(batch_test, model)
            loss_recorder.append(batch_result['loss'])
            se_recorder.append(batch_result['se'])
            abs_ratio_recorder.append(batch_result['abs_ratio'])
    
    return {'loss': np.mean(loss_recorder), 'rmse': np.sqrt(np.mean(se_recorder)), 'abs_ratio': np.mean(abs_ratio_recorder)}
        
# COMPLETED: add recorder to save model state, then check the predic function.
def train_model(model, dataloaders, args, logger):
    train_loader, val_loader, test_loader = dataloaders
    # time_encoder = HarmonicEncoder(args.time_encoder_dimension)
    optimizer = get_optimizer(model, args.optimizer, args.lr, args.l2)
    recorder = Recorder({'loss': 0}, args.checkpoint_dir, args.dataset, args.time_str)
    # results = evaluate(model, train_loader, test_loader, args, logger) # evaluate with random model
    # logger.info(f"Without training, test_loss: {results['loss']:.4f}, test_rmse: {results['rmse']:.4f}, test_abs_ratio: {results['abs_ratio']:.4f}")
    for i in range(args.epochs):
        train_loss = optimize_epoch(model, optimizer, train_loader, args, logger, epoch=i)
        recorder.append_model_state(model.state_dict())
        recorder.save_model(latest=True)
        results = evaluate(model, train_loader, test_loader, args, logger, epoch=i)
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
    model_states = recorder.load_model(time_str)
    state_dict = model_states.get(state_dict_filename, None)
    assert state_dict is not None, f"No {state_dict_filename}.state_dict in {args.dataset}/{time_str}/ dir"
    # import ipdb; ipdb.set_trace()
    model.load_state_dict(state_dict)
    model = model.to(device)
    logger.info(f'Evaluate using {time_str} state_dict.')
    train_loader, val_loader, test_loader = dataloaders
    results = evaluate(model, train_loader, test_loader, args, logger, **kwargs)
    logger.info(f"Eval, test_loss: {results['loss']:.4f}, test_rmse: {results['rmse']:.4f}, test_abs_ratio: {results['abs_ratio']:.4f}")