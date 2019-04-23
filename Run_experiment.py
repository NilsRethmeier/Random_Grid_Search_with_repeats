import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="6"
print(torch.cuda.device_count())
from collections import OrderedDict
import os
import sys
if "../models" not in sys.path: # only add this once
    sys.path.append("../models") # add the Visualization package to python_path so we can use it here
print(sys.path)
from RandomCVRecorder import ParamConfigGen, RecomputationGuardedCSVDictWriter
from TensorboardX_plots import parameter_search_vis
from VisdomLogger import VisdomLogger
import pickle
import collections
from pathlib import Path  
import pandas as pd
from collections import defaultdict
import copy
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
import time


def shorten_param(key, val, short=2, params_to_incudle={'batch_size', 'activation', 'loss', 'lr', 'embs'}):
    if key in params_to_incudle:
        if callable(val):
            val = val.__name__
        if key == 'embs':
            return val.split('/')[-1]
        return key[:short] + ":" + str(val)[:5]
    else:
        return '' # remove unwanted clutter

def clean_dict(d):
    # shorthen function intance names to just function names
    d2 = dict()
    for k, val in d.items():
        if callable(val):
            val = val.__name__
        d2[k] = val
    return d2
    
def make_exp_path_from_params(param_conf_dict, params_to_incudle={'l1', 'batch_size', 'activation', 'variable_bs', 'loss', 'rep_dim', 'lr', 'embs', 'reduce', 'CV_run'}):
    return '\n'.join([shorten_param(k, v, params_to_incudle=params_to_incudle) for k, v in param_conf_dict.items()])

def train_dev_split(embs, vocab, train_size_fraq=.9):
    df = pd.DataFrame()
    df['embs'] = embs
    df['vocab'] = vocab
    msk = np.random.rand(len(df)) < train_size_fraq
    train = df[msk]
    dev = df[~msk]
    return train['embs'].values.tolist(), train['vocab'].values.tolist(), dev['embs'].values.tolist(), dev['vocab'].values.tolist()

def get_optimizer(hyper_conf, net):
    return hyper_conf['optim']['obj'](filter(lambda x: x.requires_grad, net.parameters()), # filter params that are non-tuneable (e.g. Embedding layer)
                                      lr=hyper_conf['lr'],
                                      **hyper_conf['optim']['params'])

def run_experiments(hyper_params, environment_name=None):
    """ Main method to run experiments """
    start = time.time()
    visl = VisdomLogger(environment_name=environment_name)
    config_gen = ParamConfigGen()
    exp_result_dir = '../models/results/' + environment_name + '_paramsearch.csv'
    *measures = ["F1", "ACC"]
    # task measures
    *tasks = ["C", "R", "A"]

    *losses = ['BCE']
    guarded_csv = RecomputationGuardedCSVDictWriter(path=exp_result_dir, params=hyper_params, fields_to_ignore_in_hashing=["CV_run", 'fold'], finished_fields=measures)
    # main param search loop
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ptdtype = torch.float
    if ptdtype == torch.float:
        npdtype = np.float32

    exp_num = 1
    for pc in config_gen.parameterConfigGenerator(hyper_params, random_search=True, CV=6, max_experiments=2000, verbose=True):
        print("Experiment_num:", exp_num, end =" ")
        exp_num += 1
#         print(pc)
        deque = collections.deque(maxlen=10) # save the batchsize last losses so that you can average them

        train_embs, train_vocab, dev_embs, dev_vocab = train_dev_split(embs, vocab, train_size_fraq=pc['train_frac'])
        dev_embs = np.array(dev_embs, dtype=npdtype) # string array to float array
        pc['vocab_size'] = len(vocab)
        pc['emb_d'] = len(train_embs[0])

        model = pc['model'](pc)
        model = model.cuda() if torch.cuda.is_available() else model
        if not torch.cuda.is_available():
            print("WARNING: running on CPUs")
            
        optimizer = get_optimizer(pc, model) # torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        optimizer.zero_grad()
        epoch_opts=dict(xlabel='b='+str(pc['batch_size']), ylabel=str(pc['loss'].__name__), title='epoch_loss_train', showlegend=True)
        batch_opts=dict(xlabel='b='+str(pc['batch_size']), ylabel=str(pc['loss'].__name__), title='batch_loss_train', showlegend=True)
        epoch_opts_dev=dict(xlabel='b='+str(pc['batch_size']), ylabel=str(pc['loss'].__name__), title='epoch_loss_dev', showlegend=True)
        b_i = 0
        param_conf = clean_dict(pc)
        if guarded_csv.check_if_already_run(run_params=param_conf):
            """ Dont run an experiment twice """
            continue

        for epoch in range(1, pc['epochs'] + 1):
            e_i = 0
            dev = torch.tensor(dev_embs, requires_grad=False, device=device, dtype=ptdtype) #torch.FloatTensor(dev_embs)
            pred_dev, _ = model.forward(dev)
            loss_dev = pc['loss'](pred_dev, dev, pc['reduce'])
            loss_dev = float(loss_dev.cpu().data.numpy())
            visl.add_line(x=[epoch], y=[loss_dev], trace_name=make_exp_path_from_params(pc), win_name='epoch dev LOSS', opts=epoch_opts_dev)
            RMSE_dev = float(RMSE(pred_dev, dev).cpu().data.numpy())

            for batch in give_batch(train_embs, bs):
                b_i += len(batch)
                since_print += len(batch)
                e_i += 1
                x_b = torch.tensor(batch, requires_grad=False, device=device, dtype=ptdtype)
                pred, _ = model.forward(x_b)
                loss = pc['loss'](pred, x_b, pc['reduce'])
                loss.backward()
                optimizer.step() # update parameters
                optimizer.zero_grad()
                ls = float(loss.cpu().data.numpy())
                deque.append(ls)
                # alsways record both losses
                if epoch == 1: # epoch start loss
                    visl.add_line(x=[epoch], y=[ls], trace_name=make_exp_path_from_params(pc), win_name='epoch train LOSS ', opts=epoch_opts)
                    visl.add_line(x=[b_i], y=[np.mean(deque)], trace_name=make_exp_path_from_params(pc), win_name='batch train LOSS ', opts=batch_opts)
                if since_print >= print_interval:
                    since_print = 0
                    # record mean loss
                    visl.add_line(x=[b_i], y=[np.mean(deque)], trace_name=make_exp_path_from_params(pc), win_name='batch train LOSS', opts=batch_opts)
                since_print += 1
            # epoch
            visl.add_line(x=[epoch], y=[np.mean(deque)], trace_name=make_exp_path_from_params(pc), win_name='epoch train LOSS', opts=epoch_opts)
            # add word eval
            # plot all current single task scores
            opts_R = copy.deepcopy(epoch_opts) 
            opts_R['title'] = 'R SCORE'
            visl.add_line(x=[epoch], y=[20], trace_name=make_exp_path_from_params(pc), win_name=opts_R['title'], opts=opts_R)
            
            opts_C = copy.deepcopy(epoch_opts) 
            opts_C['title'] = 'C SCORE'
            visl.add_line(x=[epoch], y=[10], trace_name=make_exp_path_from_params(pc), win_name=opts_C['title'], opts=opts_C)
                
        param_conf['C'] = c_loss
        param_conf['R'] = r_loss
        param_conf.update(ANY_OTHER_SCORES_YOU_COLLECTED)
        # write csv
        guarded_csv.writerow(param_conf)
        parameter_search_vis(result_folder=exp_result_dir, vis=visl.vis, sort_by_column=measures, win_handle='parameter_search')

    print("Time elapsed:", (time.time() - start)/60)
    
    
# ICML 2019 experiment rand-search
hyper_params = OrderedDict([  ('model', [Model1, model2]), 
                              ('embs', ['dataset.tsv',     
                                        ]), 
                               ('vocab_size', ['added_on_the_fly']),
                               ('batch_size', [128]),
                               ('activation', [linear]),
                               ('epochs', [14]), # 77
                               ('train_frac', [.8]), # .7
                               ('emb_d', ['added_on_the_fly']),
                               ('loss', [nn.MSE]), # SIG_RMSE
                               ('optim', [{"obj":torch.optim.Adam, "params":{"eps":1e-08}}]),
                               ('lr', [2e-02]) 
                          ])

environment_name='EXPERIMENT_NAME'
run_experiments(hyper_params, environment_name)