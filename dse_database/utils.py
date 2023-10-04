import sys, pickle
from os.path import isfile, join, dirname, abspath
from os import scandir
import torch
import torch.nn as nn
from collections import OrderedDict, Counter, defaultdict
import numpy as np
from scipy.stats import mstats
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error, mean_absolute_error, max_error, \
    mean_absolute_percentage_error, classification_report, confusion_matrix
from scipy.stats import rankdata, kendalltau
import pandas as pd

import re

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


def save_pickle(data, filepath, print_msg=True):
    if print_msg:
        print('Saving to {}'.format(filepath))
    with open(filepath, 'wb') as handle:
        if sys.version_info.major < 3:  # python 2
            pickle.dump(data, handle)
        elif sys.version_info >= (3, 4):  # qilin & feilong --> 3.4
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            raise NotImplementedError()


def load_pickle(filepath, print_msg=True):
    fp = proc_filepath(filepath, '.pickle')
    if isfile(fp):
        with open(fp, 'rb') as handle:
            pickle_data = pickle.load(handle)
            return pickle_data
    elif print_msg:
        print('No file {}'.format(fp))


def proc_filepath(filepath, ext='.klepto'):
    if type(filepath) is not str:
        raise RuntimeError('Did you pass a file path to this function?')
    return append_ext_to_filepath(ext, filepath)


def append_ext_to_filepath(ext, fp):
    if not fp.endswith(ext):
        fp += ext
    return fp


def get_root_path():
    return dirname(abspath(__file__))

def get_subdir(cur_dir):
    return [f.path for f in scandir(cur_dir) if f.is_dir()]


def create_dir_if_not_exists(dir):
    import os
    if not os.path.exists(dir):
        os.makedirs(dir)

def save_fig(plt, dir, fn, print_path=False):
    plt_cnt = 0
    if dir is None or fn is None:
        return plt_cnt
    final_path_without_ext = '{}/{}'.format(dir, fn)
    for ext in ['png', 'eps']:
        full_path = final_path_without_ext + '.' + ext
        create_dir_if_not_exists(dirname(full_path))
        try:
            plt.savefig(full_path, bbox_inches='tight')
        except:
            warn('savefig')
        if print_path:
            print('Saved to {}'.format(full_path))
        plt_cnt += 1
    return plt_cnt


def create_act(act, num_parameters=None):
    if act == 'relu' or act == 'ReLU':
        return nn.ReLU()
    elif act == 'prelu':
        return nn.PReLU(num_parameters)
    elif act == 'sigmoid':
        return nn.Sigmoid()
    elif act == 'tanh':
        return nn.Tanh()
    elif act == 'identity' or act == 'None':
        class Identity(nn.Module):
            def forward(self, x):
                return x

        return Identity()
    if act == 'elu' or act == 'elu+1':
        return nn.ELU()
    else:
        raise ValueError('Unknown activation function {}'.format(act))


def print_stats(li, name):
    stats = OrderedDict()
    stats['#'] = len(li)
    stats['Avg'] = np.mean(li)
    stats['Std'] = np.std(li)
    stats['Min'] = np.min(li)
    stats['Max'] = np.max(li)
    print(name)
    for k, v in stats.items():
        print(f'\t{k}:\t{v}')



def plot_dist(data, label, save_dir, analyze_dist=True, bins=None):
    if analyze_dist:
        _analyze_dist(label, data)
    fn = f'distribution_{label}.png'
    plt.figure()
    sns.set()
    ax = sns.distplot(data, bins=bins, axlabel=label)
    plt.xlabel(label)
    ax.figure.savefig(join(save_dir, fn))
    plt.close()


def _analyze_dist(label, data):
    func = print
    func(f'--- Analyzing distribution of {label} (len={len(data)})')
    if np.isnan(np.sum(data)):
        func(f'{label} has nan')
    probs = [0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999, 0.9999, 0.99999]
    quantiles = mstats.mquantiles(data, prob=probs)
    func(f'{label} {len(data)}')
    s = '\t'.join([str(x) for x in probs])
    func(f'\tprob     \t {s}')
    s = '\t'.join(['{:.2f}'.format(x) for x in quantiles])
    func(f'\tquantiles\t {s}')
    func(f'\tnp.min(data)\t {np.min(data)}')
    func(f'\tnp.max(data)\t {np.max(data)}')
    func(f'\tnp.mean(data)\t {np.mean(data)}')
    func(f'\tnp.std(data)\t {np.std(data)}')
    
POINTS_MARKERS = ['o', '.', '.', '.', '', ',', 'x', '+', 'v', '^', '<', '>', 's', 'd']
POINTS_COLORS = ["red","green","blue","blue", "blue", "yellow","pink","black","orange","purple","beige","brown","gray","cyan","magenta"]

def _report_rmse_etc(points_dict, label='', print_result=True):
    data = defaultdict(list)
    tot_mape, tot_rmse, tot_mse, tot_mae, tot_max_err, tot_tau, tot_std = \
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    num_data = None
    try:
        for target_name, d in points_dict.items():
            true_li = [data for data,_ in d['pred']]
            pred_li = [data for _,data in d['pred']]
            num_data = len(true_li)
            mape = mean_absolute_percentage_error(true_li, pred_li)
            rmse = mean_squared_error(true_li, pred_li, squared=False)
            mse = mean_squared_error(true_li, pred_li, squared=True)
            mae = mean_absolute_error(true_li, pred_li)
            max_err = max_error(true_li, pred_li)

            true_rank = rankdata(true_li)
            pred_rank = rankdata(pred_li)
            tau = kendalltau(true_rank, pred_rank)[0]
            data['target'].append(target_name)
            data['mape'].append(mape)
            data['rmse'].append(rmse)
            data['mse'].append(mse)
            data['mae'].append(mae)
            data['max_err'].append(max_err)
            data['tau'].append(tau)

            tot_mape += mape
            tot_rmse += rmse
            tot_mse += mse
            tot_mae += mae
            tot_max_err += max_err
            tot_tau += tau

            pred_std = d.get('pred_std')
            if pred_std is not None:
                assert type(pred_std) is np.ndarray, f'{type(pred_std)}'
                pred_std = np.mean(pred_std)
                data['pred_std'].append(pred_std)
                tot_std += pred_std
        data['target'].append('tot/avg')
        data['mape'].append(tot_mape)
        data['rmse'].append(tot_rmse)
        data['mse'].append(tot_mse)
        data['mae'].append(tot_mae)
        data['max_err'].append(tot_max_err)
        data['tau'].append(tot_tau / len(points_dict))
        if 'pred_std' in data:
            data['pred_std'].append(tot_std / len(points_dict))
    except ValueError as v:
        print(f'Error {v}')
        data = defaultdict(list)

    df = pd.DataFrame(data)
    pd.set_option('display.max_columns', None)
    if print_result:
        print(num_data)
        print(df.round(4))
    return df

def multi_plot_dimension(target_list):
    num_figure = len(target_list)
    if num_figure == 1:
        y_dim = 1
        x_dim = 1
    elif num_figure == 2:
        y_dim = 1
        x_dim = 2
    elif num_figure == 3:
        y_dim = 1
        x_dim = 3
    elif num_figure == 4:
        y_dim = 2
        x_dim = 2
    elif num_figure == 5 or num_figure == 6:
        y_dim = 2
        x_dim = 3  
    return num_figure, x_dim, y_dim 
  
def plot_scatter_with_subplot(points_dict_multi_target, label, save_dir, target_list, connected = True):
    i = 0
    num_figure, x_dim, y_dim = multi_plot_dimension(target_list) 
    points_dict = {}
    ss = ['r-', 'b-', 'g-', 'c-', 'm-', 'k-', 'y-', 'w-']
    cs = [s[0] for s in ss]
    fig = plt.figure()
    # print(fig.get_figheight(), fig.get_figwidth())
    fig.set_figheight(18)
    fig.set_figwidth(24)
    m = {'p': 'o', 't': 'x'}
    for idx, target in enumerate(target_list):
        points_dict[f'p'] = points_dict_multi_target[target]['pred']
        points_dict[f't'] = points_dict_multi_target[target]['true']
        ax=plt.subplot(y_dim, x_dim, idx+1)
        ax.set_facecolor('xkcd:gray')
        i = 0
        for pname, points_ in points_dict.items(): # dict (true/pred) of dict (name: points)
            for gname, points in points_.items():
                x_li = [str(int(point[0])) for point in sorted(points)]
                y_li = [round(float(point[1]), 2) for point in sorted(points)]
                plt.scatter(np.array(x_li), np.array(y_li), label=f'{gname}-{pname}', color=cs[i % len(cs)], marker=m[pname])
                if connected:
                    plt.plot(np.array(x_li), np.array(y_li), ss[i % len(ss)])
                i += 1    
        plt.legend(loc='best')
        plt.title(f'{target}')
        plt.grid(True)
        plt.axis('on')
        points_dict = {}   
    
    plt.suptitle(f'{label}')    
    fn = f'points_{label}.png'
    plt.savefig(join(save_dir, fn), bbox_inches='tight')
    plt.close()


def plot_points_with_subplot(points_dict_multi_target, label, save_dir, target_list, use_sigma=False):
    i = 0
    num_figure, x_dim, y_dim = multi_plot_dimension(target_list) 
    points_dict = {}
    fig = plt.figure()
    fig.set_figheight(7)
    fig.set_figwidth(15)
    for idx, target in enumerate(target_list):
        points_dict[f'pred_points'] = points_dict_multi_target[target]['pred']
        
        if use_sigma:
            points_dict[f'mu-sigma_points'] = points_dict_multi_target[target]['sigma_mu']
            points_dict[f'mu+sigma_points'] = points_dict_multi_target[target]['sigma+mu']
        plt.subplot(y_dim, x_dim, idx+1)
        i = 0
        for pname, points in points_dict.items():
            xs = [point[0] for point in sorted(points)]
            ys = [point[1] for point in sorted(points)]
            plt.plot(xs, ys, POINTS_MARKERS[i % len(POINTS_MARKERS)],
                    color=POINTS_COLORS[i % len(POINTS_COLORS)],
                    label=f'Vitis20.2 vs SDx18.3')
            plt.xlabel('SDx18.3')
            if target == 'perf':
                plt.ylabel('Vitis20.2')
            xpoints = ypoints = plt.xlim()
            plt.plot(xpoints, ypoints, linestyle='--', color='k', lw=3, label='y=x', scalex=False, scaley=False)
            i += 1    
        plt.legend(loc='best')
        if target == 'perf': target = 'latency'
        plt.title(f'{target}')
        points_dict = {}   
    fn = f'points_{label}.png'
    plt.savefig(join(save_dir, fn), bbox_inches='tight')
    plt.close()