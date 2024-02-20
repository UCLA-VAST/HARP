from config import FLAGS
from saver import saver
from utils import MLP, load, get_save_path, argsort, get_root_path, get_src_path, \
     _get_y_with_target, _get_y
from data import _encode_edge_dict, _encode_edge_torch, _encode_X_torch, create_edge_index, _encode_X_dict
from model import Net
from parameter import DesignSpace, DesignPoint, DesignParameter, get_default_point, topo_sort_param_ids, compile_design_space, gen_key_from_design_point
from config_ds import build_config
from result import Result

from enum import Enum
import json
import os
from math import ceil, inf, exp, log2
import math
from os.path import join, basename, dirname

import time
import torch
from torch_geometric.data import Data, DataLoader
from logging import Logger
from typing import Deque, Dict, List, Optional, Set, Union, Generator, Any
import sys
from tqdm import tqdm
import networkx as nx
from collections import OrderedDict
from glob import glob, iglob
import pickle
from copy import deepcopy
import redis
from subprocess import Popen, DEVNULL, PIPE
import shutil
import numpy as np

import random
from pprint import pprint


TARGET = ['perf', 'util-DSP', 'util-BRAM', 'util-LUT', 'util-FF']
def persist(database, db_file_path) -> bool:
    dump_db = {
        key: database.hget(0, key)
        for key in database.hgetall(0)
    }
    with open(db_file_path, 'wb') as filep:
        pickle.dump(dump_db, filep, pickle.HIGHEST_PROTOCOL)

    return True

def run_procs(saver, procs, database, kernel, f_db_new):
    saver.info(f'Launching a batch with {len(procs)} jobs')
    try:
        while procs:
            prev_procs = list(procs)
            procs = []
            for p_list in prev_procs:
                text = 'None'
                # print(p_list)
                idx, key, p = p_list
                # text = (p.communicate()[0]).decode('utf-8')
                ret = p.poll()
                # Finished and unsuccessful
                if ret is not None and ret != 0:
                    text = (p.communicate()[0]).decode('utf-8')
                    saver.info(f'Job with batch id {idx} has non-zero exit code: {ret}')
                    saver.debug('############################')
                    saver.debug(f'Recieved output for {key}')
                    saver.debug(text)
                    saver.debug('############################')
                # Finished and successful
                elif ret is not None:
                    text = (p.communicate()[0]).decode('utf-8')
                    saver.debug('############################')
                    saver.debug(f'Recieved output for {key}')
                    saver.debug(text)
                    saver.debug('############################')

                    q_result = pickle.load(open(f'localdse/kernel_results/{kernel}_{idx}.pickle', 'rb'))

                    for _key, result in q_result.items():
                        pickled_result = pickle.dumps(result)
                        if 'lv2' in key:
                            database.hset(0, _key, pickled_result)
                        saver.info(f'Performance for {_key}: {result.perf} with return code: {result.ret_code} and resource utilization: {result.res_util}')
                    if 'EARLY_REJECT' in text:
                        for _key, result in q_result.items():
                            if result.ret_code != 'EARLY_REJECT':
                                result.ret_code = 'EARLY_REJECT'
                                result.perf = 0.0
                                pickled_result = pickle.dumps(result)
                                database.hset(0, _key.replace('lv2', 'lv1'), pickled_result)
                                #saver.info(f'Performance for {key}: {result.perf}')
                    persist(database, f_db_new)
                # Still running
                else:
                    procs.append([idx, key, p])
                
                time.sleep(1)
    except:
        saver.error(f'Failed to finish the processes')
        raise RuntimeError()        
        
        
class GNNModel():
    def __init__(self, saver, first_dse = False, multi_target = True, task = 'regression', num_layers = FLAGS.num_layers, D = FLAGS.D, target = FLAGS.target, model_path = None, model_id = 0, model_name = f'{FLAGS.model_tag}_model_state_dict.pth', encoder_name = 'encoders', pragma_dim = None):
        """
        >>> self.encoder.keys()
        dict_keys(['enc_ntype', 'enc_ptype', 'enc_itype', 'enc_ftype', 'enc_btype', 'enc_ftype_edge', 'enc_ptype_edge'])

        """
        self.log = saver
        
        base_path = 'models'    
        model = [f for f in iglob(join(get_root_path(), base_path), recursive=True) if f.endswith('.pth') and 'pragma_as_MLP' in f and task in f] 
        if model_path:
            if type(model_path) is list:
                self.model_path = model_path[0]
            else:
                self.model_path = model_path
        else:
            if task == 'regression':
                if FLAGS.model_path == None:
                    assert len(model) == 1
                    self.model_path = model[0]
                else:
                    if type(FLAGS.model_path) is list:
                        self.model_path = FLAGS.model_path[0]
                    else:
                        self.model_path = FLAGS.model_path
            else:
                if FLAGS.class_model_path == None:
                    assert len(model) == 1
                    self.model_path = model[0]
                else:
                    self.model_path = FLAGS.class_model_path

        if FLAGS.encoder_path == None:
            self.encoder_path = join(self.path, encoder_name)
        else:
            self.encoder_path = FLAGS.encoder_path

        print(self.model_path)
        shutil.copy(self.model_path, join(saver.logdir, f'{task}-{basename(self.model_path)}-{model_id}'))
        shutil.copy(f'{self.encoder_path}', join(saver.logdir, f'{task}-{basename(self.encoder_path)}-{model_id}.klepto'))

        self.num_features = FLAGS.num_features # 124
        self.model = Net(in_channels = self.num_features, edge_dim = FLAGS.edge_dim, init_pragma_dict=pragma_dim, task = task, num_layers = num_layers, D = D, target = target).to(FLAGS.device)
        if first_dse:
            saver.log_model_architecture(self.model)

        self.model.load_state_dict(torch.load(join(self.model_path), map_location=torch.device('cpu')))
        saver.info(f'loaded {self.model_path}')
        self.encoder = load(self.encoder_path)

          
    
    def encode_node(self, g, point: DesignPoint): 
        node_dict = _encode_X_dict(g, point=point)
        required_keys = ['X_contextnids', 'X_pragmanids', 'X_pseudonids', 'X_icmpnids', 'X_pragmascopenids', 'X_pragma_per_node']
        
        enc_ntype = self.encoder['enc_ntype']
        enc_ptype = self.encoder['enc_ptype']
        enc_itype = self.encoder['enc_itype']
        enc_ftype = self.encoder['enc_ftype']
        enc_btype = self.encoder['enc_btype']
        
        return _encode_X_torch(node_dict, enc_ntype, enc_ptype, enc_itype, enc_ftype, enc_btype), \
            {k: node_dict[k] for k in required_keys}
        
        
    def encode_edge(self, g):
        edge_dict = _encode_edge_dict(g)
        enc_ptype_edge = self.encoder['enc_ptype_edge']
        enc_ftype_edge = self.encoder['enc_ftype_edge']
        
        return _encode_edge_torch(edge_dict, enc_ftype_edge, enc_ptype_edge)
    
    def perf_as_quality(self, new_result: Result) -> float:
        """Compute the quality of the point by (1 / latency).

        Args:
            new_result: The new result to be qualified.

        Returns:
            The quality value. Larger the better.
        """
        return 1.0 / new_result.perf

    def finte_diff_as_quality(self, new_result: Result, ref_result: Result) -> float:
        """Compute the quality of the point by finite difference method.

        Args:
            new_result: The new result to be qualified.
            ref_result: The reference result.

        Returns:
            The quality value (negative finite differnece). Larger the better.
        """

        def quantify_util(result: Result) -> float:
            """Quantify the resource utilization to a float number.

            util' = 5 * ceil(util / 5) for each util,
            area = sum(2^1(1/(1-util))) for each util'

            Args:
                result: The evaluation result.

            Returns:
                The quantified area value with the range (2*N) to infinite,
                where N is # of resources.
            """

            # Reduce the sensitivity to (100 / 5) = 20 intervals
            utils = [
                5 * ceil(u * 100 / 5) / 100 for k, u in result.res_util.items()
                if k.startswith('util')
            ]

            # Compute the area
            return sum([2.0**(1.0 / (1.0 - u)) for u in utils])

        ref_util = quantify_util(ref_result)
        new_util = quantify_util(new_result)

        if (new_result.perf / ref_result.perf) > 1.05:
            # Performance is too worse to be considered
            return -float('inf')

        if new_util == ref_util:
            if new_result.perf < ref_result.perf:
                # Free lunch
                return float('inf')
            # Same util but slightly worse performance, neutral
            return 0

        return -(new_result.perf - ref_result.perf) / (new_util - ref_util)

    def eff_as_quality(self, new_result: Result, ref_result: Result) -> float:
        """Compute the quality of the point by resource efficiency.

        Args:
            new_result: The new result to be qualified.
            ref_result: The reference result.

        Returns:
            The quality value (negative finite differnece). Larger the better.
        """
        if (new_result.perf / ref_result.perf) > 1.05:
            # Performance is too worse to be considered
            return -float('inf')

        area = sum([u for k, u in new_result.res_util.items() if k.startswith('util')])

        return 1 / (new_result.perf * area)
    
    def test(self, loader, config, mode = 'regression'):
        self.model.eval()

        i = 0
        results: List[Result] = []
        target_list = FLAGS.target
        if not isinstance(FLAGS.target, list):
            target_list = [FLAGS.target]

        for data in loader:
            data = data.to(FLAGS.device)
            out_dict, loss, loss_dict, _ = self.model(data)
            
            if mode == 'regression':
                for i in range(len(out_dict['perf'])):
                    curr_result = Result()
                    curr_result.point = data[i].point
                    for target_name in target_list:
                        out = out_dict[target_name]
                        out_value = out[i].item()
                        if target_name == 'perf':
                            curr_result.perf = out_value
                            if FLAGS.encode_log:
                                curr_result.actual_perf = 2**out_value
                            else:
                                curr_result.actual_perf = out_value
                        elif target_name in curr_result.res_util.keys():
                            curr_result.res_util[target_name] = out_value
                        else:
                            raise NotImplementedError()
                    curr_result.quality = self.perf_as_quality(curr_result)
                    
                    # prune if over-utilizes the board
                    max_utils = config['max-util']
                    utils = {k[5:]: max(0.0, u) for k, u in curr_result.res_util.items() if k.startswith('util-')}
                    if FLAGS.prune_util:
                        curr_result.valid = all([(utils[res] / FLAGS.util_normalizer )< max_utils[res] for res in max_utils])
                    else:
                        curr_result.valid = True
                    results.append(curr_result)
            elif mode == 'class':
                _, pred = torch.max(out_dict['perf'], 1)
                labels = _get_y_with_target(data, 'perf') 
                return (pred == labels)
            else:
                raise NotImplementedError()
                    

        return results
  
  
        
class Explorer():
    def __init__(self, path_kernel: str, kernel_name: str, path_graph: str, first_dse: bool = False, run_dse: bool = True, prune_invalid = False, pragma_dim = None):
        """Constructor.

        Args:
            ds: DesignSpace
        """
        self.run_dse = run_dse
        self.log = saver
        self.kernel_name = kernel_name
        self.config_path = join(path_kernel, f'{kernel_name}_ds_config.json')
        self.config = self.load_config()

        
        if FLAGS.separate_T:
            pragma_dim = load(join(dirname(FLAGS.encoder_path), f'{FLAGS.v_db}_pragma_dim'))
            for gname in pragma_dim:
                self.max_pragma_length = pragma_dim[gname][1] ## it's a list of [#pragma per kernel, max #pragma for all kernels]
                break

        self.timeout = 60 * 60
        self.hls_timeout = 40
        self.ds, self.ds_size = compile_design_space(
            self.config['design-space']['definition'],
            None,
            self.log)

        self.batch_size = 1
        # Status checking
        self.num_top_designs = 10
        self.key_perf_dict = OrderedDict()
        self.best_results_dict = {}
        self.best_result: Result = Result()
        self.explored_point = 0
        self.ordered_pids = self.topo_sort_param_ids(self.ds)
        self.ensemble_GNNmodels = [] ## list of GNN models for regression. if ensemble is not used, only has one entry
        
        if FLAGS.ensemble > 1: ## number of models in ensemble, if 1, not ensemble
            for i in range(FLAGS.ensemble):
                model_path = FLAGS.model_path[i]
                model = GNNModel(self.log, first_dse = first_dse, multi_target=True, task='regression', num_layers = FLAGS.num_layers, D = FLAGS.D, model_path=model_path, model_id=i, pragma_dim = pragma_dim)
                self.ensemble_GNNmodels.append(model)
        else:
            self.GNNmodel = GNNModel(self.log, first_dse = first_dse, multi_target=True, task='regression', num_layers = FLAGS.num_layers, D = FLAGS.D, pragma_dim = pragma_dim)
            self.ensemble_GNNmodels.append(self.GNNmodel)


        if FLAGS.graph_type == '':
            pruner = 'extended'
            print(path_graph)
            gexf_file = sorted([f for f in iglob(path_graph + "/**/*", recursive=True) if f.endswith('.gexf') and f'{kernel_name}_' in f and pruner not in f])
        else:
            gexf_file = sorted([f for f in glob(path_graph + "/**/*") if f.endswith('.gexf') and f'{kernel_name}_' in f and FLAGS.graph_type in f])
        print(gexf_file)
        # print(gexf_file, glob(path_graph))
        assert len(gexf_file) == 1
        # self.graph_path = join(path_graph, f'{kernel_name}_processed_result.gexf')
        self.graph_path = join(path_graph, gexf_file[0])
        saver.info(f'graph path {self.graph_path}')
        self.graph = nx.read_gexf(self.graph_path)

        ## for ploting one of the objectives (all points)
        self.plot_data = {k: [] for k in FLAGS.target}
        
        self.prune_invalid = prune_invalid
        if self.prune_invalid:
            self.GNNmodel_valid = GNNModel(self.log, multi_target=False, task='class', num_layers = FLAGS.num_layers, D = FLAGS.D, pragma_dim = pragma_dim) 
        
        
    def load_config(self) -> Dict[str, Any]:
        """Load the DSE configurations.

        Returns:
            A dictionary of configurations.
        """

        try:
            if not os.path.exists(self.config_path):
                self.log.error(('Config JSON file not found: %s', self.config_path))
                raise RuntimeError()

            self.log.info('Loading configurations')
            with open(self.config_path, 'r', errors='replace') as filep:
                try:
                    user_config = json.load(filep)
                except ValueError as err:
                    self.log.error(('Failed to load config: %s', str(err)))
                    raise RuntimeError()

            config = build_config(user_config, self.log)
            if config is None:
                self.log.error(('Config %s is invalid', self.config_path))
                raise RuntimeError()
        except RuntimeError:
            sys.exit(1)

        return config
        

    def get_pragmas(self, point: DesignPoint) -> List[int]:
        pragmas = []
        for _, value in sorted(point.items()):
            if type(value) is str:
                if value.lower() == 'flatten':
                    value = 100 # 2
                elif value.lower() == 'off':
                    value = 1
                elif value.lower() == '':
                    value = 50 # 3
                else:
                    raise ValueError()
            elif type(value) is int:
                pass
            else:
                raise ValueError()
            pragmas.append(value)
        return pragmas
    
    def apply_design_point(self, g, point: DesignPoint, mode = 'regression', model=None) -> Data:
        if model is None: model = self.GNNmodel
        X, d_node = model.encode_node(g, point)
        edge_attr = model.encode_edge(g)
        edge_index = create_edge_index(g)
        pragmas = self.get_pragmas(point)
        if FLAGS.separate_T:
            pragmas.extend([0] * (self.max_pragma_length - len(pragmas)))

        resources = ['BRAM', 'DSP', 'LUT', 'FF']
        keys = ['perf', 'actual_perf', 'quality']
        d_node['pragmas'] = torch.FloatTensor(np.array([pragmas]))
        for r in resources:
            keys.append('util-' + r)
            keys.append('total-' + r)
        for key in keys:
            d_node[key] = 0
        if mode == 'class': ## default: point is valid
            d_node['perf'] = 1
        
        if FLAGS.task == 'regression':
            data = Data(
                x=X,
                edge_index=edge_index,
                edge_attr=edge_attr,
                point=point,
                X_contextnids=d_node['X_contextnids'],
                X_pragmanids=d_node['X_pragmanids'],                    
                X_pragmascopenids=d_node['X_pragmascopenids'],                    
                X_pseudonids=d_node['X_pseudonids'],    
                X_icmpnids=d_node['X_icmpnids'],    
                X_pragma_per_node=d_node['X_pragma_per_node'],            
                pragmas=d_node['pragmas'],
                perf=d_node['perf'],
                actual_perf=d_node['actual_perf'],
                quality=d_node['quality'],
                util_BRAM=d_node['util-BRAM'],
                util_DSP=d_node['util-DSP'],
                util_LUT=d_node['util-LUT'],
                util_FF=d_node['util-FF'],
                total_BRAM=d_node['total-BRAM'],
                total_DSP=d_node['total-DSP'],
                total_LUT=d_node['total-LUT'],
                total_FF=d_node['total-FF']
            )
        elif FLAGS.task == 'class':
            data = Data(
                x=X,
                edge_index=edge_index,
                edge_attr=edge_attr,
                point=point,
                X_contextnids=d_node['X_contextnids'],
                X_pragmanids=d_node['X_pragmanids'],
                X_pragmascopenids=d_node['X_pragmascopenids'],                    
                X_pseudonids=d_node['X_pseudonids'],    
                X_icmpnids=d_node['X_icmpnids'],    
                X_pragma_per_node=d_node['X_pragma_per_node'],
                pragmas=d_node['pragmas'],
                perf=d_node['perf']
            )
        else:
            raise NotImplementedError()

        
        return data
    
    

    def update_best(self, result: Result) -> None:
        """Keep tracking the best result found in this explorer.

        Args:
            result: The new result to be checked.

        """
        # if result.valid and result.quality > self.best_result.quality:
        if 'speedup' in FLAGS.norm_method:
            REF = min
        else:
            REF = max
        if self.key_perf_dict:
            key_refs_perf = REF(self.key_perf_dict, key=(lambda key: self.key_perf_dict[key]))
            refs_perf = self.key_perf_dict[key_refs_perf]
        else:
            if REF == min:
                refs_perf = float(-inf)
            else:
                refs_perf = float(inf)
        point_key = gen_key_from_design_point(result.point)
        updated = False
        if point_key not in self.key_perf_dict and result.valid and REF(result.perf, refs_perf) != result.perf: # if the new result is better than the references designs
        ## use the below condition when all the perf numbers are the same, such as for aes
        # if result.valid and (REF(result.perf, refs_perf) != result.perf or refs_perf == result.perf): # if the new result is better than the references designs
        # if result.valid and (not self.key_perf_dict or self.key_perf_dict[max(self.key_perf_dict, key=(lambda key: self.key_perf_dict[key]))] < result.perf): # if the new result is better than the references designs
            self.best_result = result
            self.log.info(('Found a better result at {}: Quality {:.1e}, Perf {:.1e}'.format(
                        self.explored_point, result.quality, result.perf)))
            if len(self.key_perf_dict.keys()) >= self.num_top_designs:
                ## replace maxmimum performance value
                key_refs_perf = REF(self.key_perf_dict, key=(lambda key: self.key_perf_dict[key]))
                self.best_results_dict.pop((self.key_perf_dict[key_refs_perf], key_refs_perf))
                self.key_perf_dict.pop(key_refs_perf)
                
            attrs = vars(result)
            self.log.info(', '.join("%s: %s" % item for item in attrs.items()))
            
            self.key_perf_dict[point_key] = result.perf
            self.best_results_dict[(result.perf, point_key)] = result
            updated = True
        
        if self.key_perf_dict.values():
            reward = REF([-p for p in self.key_perf_dict.values()])  
            return reward, updated
        else:
            return 0, updated

    def extract_plot_data(self, result: Result, updated = True) -> None:
        """extract the objective to be plotted
        """
        for target in FLAGS.target:
            if target == 'perf':
                data = result.perf
            elif 'util-' in target:
                data = result.res_util[target]
            else:
                raise NotImplementedError()
            if updated:
                self.plot_data[target].append(data)
            else: 
                self.plot_data[target].append(float('nan'))


    def gen_options(self, point: DesignPoint, pid: str, default = False) -> List[Union[int, str]]:
        """Evaluate available options of the target design parameter.

        Args:
            point: The current design point.
            pid: The target design parameter ID.

        Returns:
            A list of available options.
        """
        if default:
            dep_values = {dep: point[dep].default for dep in self.ds[pid].deps}
        else:
            dep_values = {dep: point[dep] for dep in self.ds[pid].deps}
        options = eval(self.ds[pid].option_expr, dep_values)
        if options is None:
            self.log.error(f'Failed to evaluate {self.ds[pid].option_expr} with dep {str(dep_values)}')
            print('Error: failed to manipulate design points')
            sys.exit(1)

        return options

    def get_order(self, point: DesignPoint, pid: str) -> int:
        """Evaluate the order of the current value.

        Args:
            point: The current design point.
            pid: The target design parameter ID.

        Returns:
            The order.
        """

        if not self.ds[pid].order:
            return 0

        order = eval(self.ds[pid].order['expr'], {self.ds[pid].order['var']: point[pid]})
        if order is None or not isinstance(order, int):
            self.log.warning(f'Failed to evaluate the order of {pid} with value {str(point[pid])}: {str(order)}')
            return 0

        return order

    def update_child(self, point: DesignPoint, pid: str) -> None:
        """Check values of affected parameters and update them in place if it is invalid.

        Args:
            point: The current design point.
            pid: The design parameter ID that just be changed.
        """

        pendings = [child for child in self.ds[pid].child if self.validate_value(point, child)]
        for child in pendings:
            self.update_child(point, child)

    def validate_point(self, point: DesignPoint) -> bool:
        """Check if the current point is valid and set it to the closest value if not.

        Args:
            point: The current design point.
            pid: The design parameter ID that just be changed.

        Returns:
            True if the value is changed.
        """

        changed = False
        for pid in point.keys():
            options = self.gen_options(point, pid)
            value = point[pid]
            if not options:  # All invalid (something not right), set to default
                self.log.warning(f'No valid options for {pid} with point {str(point)}')
                point[pid] = self.ds[pid].default
                changed = True
                continue

            if isinstance(value, int):
                # Note that we assume all options have the same type (int or str)
                cand = min(options, key=lambda x: abs(int(x) - int(value)))
                if cand != value:
                    point[pid] = cand
                    changed = True
                    continue

            if value not in options:
                point[pid] = self.ds[pid].default
                changed = True
                continue

        return changed
    
    def validate_value(self, point: DesignPoint, pid: str) -> bool:
        """Check if the current value is valid and set it to the closest value if not.

        Args:
            point: The current design point.
            pid: The design parameter ID that just be changed.

        Returns:
            True if the value is changed.
        """

        options = self.gen_options(point, pid)
        value = point[pid]
        if not options:  # All invalid (something not right), set to default
            self.log.warning(f'No valid options for {pid} with point {str(point)}')
            point[pid] = self.ds[pid].default
            return False

        if isinstance(value, int):
            # Note that we assume all options have the same type (int or str)
            cand = min(options, key=lambda x: abs(int(x) - int(value)))
            if cand != value:
                point[pid] = cand
                return True

        if value not in options:
            point[pid] = self.ds[pid].default
            return True
        return False

    def move_by(self, point: DesignPoint, pid: str, step: int = 1) -> int:
        """Move N steps of pid parameter's value in a design point in place.

        Args:
            point: The design point to be manipulated.
            pid: The target design parameter.
            step: The steps to move. Note that step can be positive or negatie,
                  but we will not move cirulatory even the step is too large.

        Returns:
            The actual move steps.
        """

        try:
            options = self.gen_options(point, pid)
            idx = options.index(point[pid])
        except (AttributeError, ValueError) as err:
            self.log.error(
                f'Fail to identify the index of value {point[pid]} of parameter {pid} at design point {str(point)}: {str(err)}')
            print('Error: failed to manipulate design points')
            sys.exit(1)

        target = idx + step
        if target >= len(options):
            target = len(options) - 1
        elif target < 0:
            target = 0

        if target != idx:
            point[pid] = options[target]
            self.update_child(point, pid)
        return target - idx
    
    def get_results(self, next_points: List[DesignPoint]) -> List[Result]:
        data_list = []
        model = None
        if FLAGS.ensemble > 1:
            model = self.ensemble_GNNmodels[0]
        if self.prune_invalid:
            for point in next_points:
                data_list.append(self.apply_design_point(self.graph, point, mode = 'class', model=model))

            test_loader = DataLoader(data_list, batch_size=self.batch_size)  # TODO
            valid = self.GNNmodel_valid.test(test_loader, self.config['evaluate'], mode='class')
            if valid == 0:
                # stop if the point is invalid
                self.log.debug(f'invalid point {point}')
                return [float(inf)] # TODO: add batch processing 
        
        data_list = []
        for point in next_points:
            data_list.append(self.apply_design_point(self.graph, point, model=model))

        test_loader = DataLoader(data_list, batch_size=self.batch_size)  # TODO
        if FLAGS.ensemble > 1:
            all_results = [model.test(test_loader, self.config['evaluate'], mode='regression')[0] for model in self.ensemble_GNNmodels]
            cur_res = all_results[0]
            max_utils = self.config['evaluate']['max-util']
            utils = None
            perf = 0
            # for curr_result in all_results:
            for i in range(FLAGS.ensemble):
                curr_result = all_results[i]
                curr_utils = {k: max(0.0, u) for k, u in curr_result.res_util.items()}
                if FLAGS.ensemble_weights != None:
                    assert len(FLAGS.ensemble_weights) == FLAGS.ensemble
                    curr_utils = {k: max(0.0, u * FLAGS.ensemble_weights[i]) for k, u in curr_result.res_util.items()}
                    curr_result.perf *= FLAGS.ensemble_weights[i]
                if utils is None:
                    utils = curr_utils
                else:
                    for k, u in utils.items():
                        utils[k] = u + curr_utils[k]
                perf += (curr_result.perf)

            if FLAGS.ensemble_weights == None:
                utils = {k: u / len(self.ensemble_GNNmodels) for k, u in utils.items()}
                perf = perf / len(self.ensemble_GNNmodels)
            cur_res.res_util = utils
            cur_res.perf = perf 
            if FLAGS.prune_util:
                cur_res.valid = all([(utils[f'util-{res}'] / FLAGS.util_normalizer )< max_utils[res] for res in max_utils])
            results = [cur_res]
        else:
            results = self.GNNmodel.test(test_loader, self.config['evaluate'], mode='regression')
        
        return results

    def get_hls_results(self, points: List[DesignPoint], database, f_db) -> List[Result]:
        ## TODO: assumes single HLS run
        procs = []
        batch_num = 1
        batch_id = 0
        src_dir = join(get_root_path(), 'dse_database/merlin_prj', f'{self.kernel_name}', 'xilinx_dse')
        work_dir = join('/expr', f'{self.kernel_name}', 'work_dir')
        for point in points:
            if len(procs) == batch_num:
                run_procs(saver, procs, database, self.kernel_name, f_db_new)
                batch_id == 0
                procs = []
            for key_, value in point.items():
                if type(value) is str or type(value) is int:
                    point[key_] = value
                else:
                    point[key_] = value.item()
            key = f'lv2:{gen_key_from_design_point(point)}'

            kernel = self.kernel_name
            f_config = self.config_path
            with open(f'./localdse/kernel_results/{self.kernel_name}_point_{batch_id}.pickle', 'wb') as handle:
                pickle.dump(point, handle, protocol=pickle.HIGHEST_PROTOCOL)
            new_work_dir = join(work_dir, f'batch_id_{batch_id}')
            p = Popen(f'cd {get_src_path()} \n source /share/atefehSZ/env.sh \n /share/atefehSZ/merlin_docker/docker-run-gnn.sh single {src_dir} {new_work_dir} {kernel} {f_config} {batch_id} {self.hls_timeout}', shell = True, stdout=PIPE)
            
            procs.append([batch_id, key, p])
            saver.info(f'Added {point} with batch id {batch_id}')
            batch_id += 1

        if len(procs) > 0:
            run_procs(saver, procs, database, self.kernel_name, f_db)

        pickle_obj = database.hget(0, f'lv2:{gen_key_from_design_point(point)}')
        return pickle.loads(pickle_obj)
    
    
    def topo_sort_param_ids(self, space: DesignSpace) -> List[str]:
        return topo_sort_param_ids(space)
    
    def traverse(self, point: DesignPoint, idx: int) -> Generator[DesignPoint, None, None]:
        """DFS traverse the design space and yield leaf points.

        Args:
            point: The current design point.
            idx: The current manipulated parameter index.

        Returns:
            A resursive generator for traversing.
        """

        if idx == len(self.ordered_pids):
            # Finish a point
            yield point
        else:
            yield from self.traverse(point, idx + 1)

            # Manipulate idx-th point
            new_point = self.clone_point(point)
            while self.move_by(new_point, self.ordered_pids[idx]) == 1:
                yield from self.traverse(new_point, idx + 1)
                new_point = self.clone_point(new_point)
    
    @staticmethod
    def clone_point(point: DesignPoint) -> DesignPoint:
        return dict(point)
    
    def run(self) -> None:
        """The main function of the explorer to launch the search algorithm.

        Args:
            algo_name: The corresponding algorithm name for running this exploration.
            algo_config: The configurable values for the algorithm.
        """
        raise NotImplementedError()
    
    
class ExhaustiveExplorer(Explorer):
    def __init__(self, path_kernel: str, kernel_name: str, path_graph: str, first_dse: bool = False, run_dse: bool = True, prune_invalid = FLAGS.prune_class, point: DesignPoint = None, pragma_dim = None):
        """Constructor.

        Args:
            ds: Design space.
        """
        super(ExhaustiveExplorer, self).__init__(path_kernel, kernel_name, path_graph, first_dse, run_dse, prune_invalid, pragma_dim)
        self.batch_size = 1
        self.log.info('Done init')
        
        if self.run_dse:
            self.run()
            attrs = vars(self.best_result)
            self.log.info('Best Results Found:')
            i = 1
            with open(join(saver.logdir, f'{kernel_name}.pickle'), 'wb') as handle:
                pickle.dump(self.best_results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
                handle.flush()
            for _, result in sorted(self.best_results_dict.items()):
                attrs = vars(result)
                self.log.info(f'Design {i}')
                self.log.info(', '.join("%s: %s" % item for item in attrs.items()))
                i += 1
        elif point is not None:
            results = self.get_results([point])
            attrs = vars(results[0])
            self.log.info(', '.join("%s: %s" % item for item in attrs.items()))
                
        
                

    def gen(self) -> Generator[List[DesignPoint], Optional[Dict[str, Result]], None]:
        #pylint:disable=missing-docstring

        self.log.info('Launch exhaustive search algorithm')

        traverser = self.traverse(get_default_point(self.ds), 0)
        iter_cnt = 0
        while True:
            next_points: List[DesignPoint] = []
            try:
                iter_cnt += 1
                self.log.debug(f'Iteration {iter_cnt}')
                while len(next_points) < self.batch_size:
                    next_points.append(next(traverser))
                    self.log.debug(f'Next point: {str(next_points[-1])}')
                yield next_points
            except StopIteration:
                if next_points:
                    yield next_points
                break

        self.log.info('No more points to be explored, stop.')
    
        
    def run(self) -> None:
        #pylint:disable=missing-docstring

        # Create a search algorithm generator
        gen_next = self.gen()

        timer = time.time()
        duplicated_iters = 0
        while (time.time() - timer) < self.timeout and self.explored_point < 75000:
            try:
                # Generate the next set of design points
                next_points = next(gen_next)
                self.log.debug(f'The algorithm generates {len(next_points)} design points')
            except StopIteration:
                break

            results = self.get_results(next_points)
            for r in results:
                if isinstance(r, Result):
                    attrs = vars(r)
                    self.log.debug(f'Evaluating Design')
                    self.log.debug(', '.join("%s: %s" % item for item in attrs.items()))
                    _, updated = self.update_best(r)
                    if FLAGS.plot_dse:
                        self.extract_plot_data(r, updated)
            self.explored_point += len(results)
            
        self.log.info(f'Explored {self.explored_point} points')