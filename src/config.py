from utils import get_user, get_host, get_root_path
import argparse
import torch
from glob import iglob
from os.path import join

decoder_arch = []

parser = argparse.ArgumentParser()
# TASK = 'class'
TASK = 'regression'
parser.add_argument('--task', default=TASK)

SUBTASK = 'dse'
# SUBTASK = 'inference'
# SUBTASK = 'train'
parser.add_argument('--subtask', default=SUBTASK)
parser.add_argument('--plot_dse', default=False)


#################### visualization ####################
parser.add_argument('--vis_per_kernel', default=True) ## only tsne visualization for now 


######################## data ########################

TARGETS = ['perf', 'quality', 'util-BRAM', 'util-DSP', 'util-LUT', 'util-FF',
           'total-BRAM', 'total-DSP', 'total-LUT', 'total-FF']


MACHSUITE_KERNEL = ['aes', 'gemm-blocked', 'gemm-ncubed', 'spmv-crs', 'spmv-ellpack', 'stencil_stencil2d',
                    'nw', 'md', 'stencil-3d']

poly_KERNEL = ['2mm', '3mm', 'adi', 'atax', 'bicg', 'bicg-large', 'covariance', 'doitgen', 
               'doitgen-red', 'fdtd-2d', 'fdtd-2d-large', 'gemm-p', 'gemm-p-large', 'gemver', 
               'gesummv', 'heat-3d', 'jacobi-1d', 'jacobi-2d', 'mvt', 'seidel-2d', 'symm', 
               'symm-opt', 'syrk', 'syr2k', 'trmm', 'trmm-opt', 'mvt-medium', 'correlation',
               'atax-medium', 'bicg-medium', 'gesummv-medium']


parser.add_argument('--force_regen', type=bool, default=False) ## must be set to True for the first time to generate the dataset

parser.add_argument('--min_allowed_latency', type=float, default=100.0) ## if latency is less than this, prune the point (used when synthesis is not valid)
EPSILON = 1e-3
parser.add_argument('--epsilon', default=EPSILON)
NORMALIZER = 1e7
parser.add_argument('--normalizer', default=NORMALIZER)
parser.add_argument('--util_normalizer', default=1)
MAX_NUMBER = 1e10
parser.add_argument('--max_number', default=MAX_NUMBER)

norm = 'speedup-log2' # 'const' 'log2' 'speedup' 'off' 'speedup-const' 'const-log2' 'none' 'speedup-log2'
parser.add_argument('--norm_method', default=norm)
parser.add_argument('--new_speedup', default=True) # new_speedup: same reference point across all, 
                                                    # old_speedup: base is the longest latency and different per kernel

parser.add_argument('--invalid', type = bool, default=False ) # False: do not include invalid designs

parser.add_argument('--encode_log', type = bool, default=False)
v_db = 'v21' # 'v20': v20 database, 'v18': v18 database
parser.add_argument('--v_db', default=v_db) # if set to true uses the db of the new version of the tool: 2020.2

test_kernels = None
parser.add_argument('--test_kernels', default=test_kernels)
target_kernel = None
# target_kernel = 'gemm-blocked'
parser.add_argument('--target_kernel', default=target_kernel)
if target_kernel == None:
    all_kernels = True
else:
    all_kernels = False
parser.add_argument('--all_kernels', type = bool, default=all_kernels)

dataset = 'harp' # machsuite and poly 
parser.add_argument('--dataset', default=dataset)

benchmark = ['machsuite', 'poly']
parser.add_argument('--benchmarks', default=benchmark)

tag = 'whole-machsuite-poly'
parser.add_argument('--tag', default=tag)


###################### graph type ######################
graph_type = 'original' # original DAC22 graph
graph_type = 'extended-pseudo-block-connected-hierarchy'
parser.add_argument('--graph_type', default=graph_type)

################## model architecture ##################
pragma_as_MLP, type_parallel, type_merge = True, '2l', '2l'
gnn_layer_after_MLP = 1
pragma_MLP_hidden_channels, merge_MLP_hidden_channels = None, None
if 'hierarchy' not in graph_type: ## separate_PT original graph
    gae_T, P_use_all_nodes, separate_pseudo, separate_T, dropout, num_features, edge_dim = True, True, False, True, 0.1, 154, 7
    model_ver = 'original-PT'
else:
    if pragma_as_MLP:
        if gnn_layer_after_MLP == 1: model_ver = 'pragma_as_MLP'
        
        if type_parallel == '2l': pragma_MLP_hidden_channels = '[in_D // 2]'
        elif type_parallel == '3l': pragma_MLP_hidden_channels = '[in_D // 2, in_D // 4]'
        
        if type_merge == '2l': merge_MLP_hidden_channels = '[in_D // 2]'
        elif type_merge == '3l': merge_MLP_hidden_channels = '[in_D // 2, in_D // 4]'
        else: raise NotImplementedError()
        gae_T, P_use_all_nodes, separate_pseudo, separate_T, dropout, num_features, edge_dim = False, True, True, False, 0.1, 153, 335
    else:
        gae_T, P_use_all_nodes, separate_pseudo, separate_T, dropout, num_features, edge_dim = True, False, False, True, 0.1, 156, 335   
        model_ver = 'hierarchy-PT'        

################# one-hot encoder ##################
encoder_path = None
encode_edge_position = True
use_encoder = True 
if use_encoder:
    encoder_path_list = [f for f in iglob(join(get_root_path(), 'models', '**'), recursive=True) if f.endswith('.klepto') and model_ver in f]
    assert len(encoder_path_list) == 1, print(encoder_path_list)
    encoder_path = encoder_path_list[0]
        
parser.add_argument('--encoder_path', default=encoder_path)


################ model architecture #################
## edge attributes
parser.add_argument('--encode_edge', type=bool, default=True)
parser.add_argument('--encode_edge_position', type=bool, default=encode_edge_position)

num_layers = 6
parser.add_argument('--num_layers', type=int, default=num_layers) 
parser.add_argument('--num_features', default=num_features) 
parser.add_argument('--edge_dim', default=edge_dim) 

multi_target = ['perf', 'util-LUT', 'util-FF', 'util-DSP', 'util-BRAM']
if SUBTASK == 'class':
    multi_target = ['perf']
parser.add_argument('--target', default=multi_target)
parser.add_argument('--MLP_common_lyr', default=0)
gnn_type = 'transformer'
parser.add_argument('--gnn_type', type=str, default=gnn_type)
parser.add_argument('--dropout', type=float, default=dropout)

jkn_mode = 'max'
parser.add_argument('--jkn_mode', type=str, default=jkn_mode)
parser.add_argument('--jkn_enable', type=bool, default=True)
node_attention = True
parser.add_argument('--node_attention', type=bool, default=node_attention)
if node_attention:
    parser.add_argument('--node_attention_MLP', type=bool, default=False)

    separate_P = True
    parser.add_argument('--separate_P', type=bool, default=separate_P)
    separate_icmp = False
    parser.add_argument('--separate_icmp', type=bool, default=separate_icmp)
    parser.add_argument('--separate_T', type=bool, default=separate_T)
    parser.add_argument('--separate_pseudo', type=bool, default=separate_pseudo)

    if separate_P:
        parser.add_argument('--P_use_all_nodes', type=bool, default=P_use_all_nodes)
    
## graph auto encoder
parser.add_argument('--gae_T', default = gae_T)
gae_P = False
parser.add_argument('--gae_P', default = gae_P)
if gae_P:
    parser.add_argument('--input_encode', default = False)
    d_type = 'type1'
    parser.add_argument('--decoder_type', default = d_type)

if pragma_as_MLP:
    assert graph_type == 'extended-pseudo-block-connected-hierarchy'
parser.add_argument('--gnn_layer_after_MLP', default=gnn_layer_after_MLP) ## number of message passing layers after MLP (pragma as MLP)
parser.add_argument('--pragma_as_MLP', default=pragma_as_MLP)
pragma_as_MLP_list = ['tile', 'pipeline', 'parallel']
parser.add_argument('--pragma_as_MLP_list', default=pragma_as_MLP_list)
pragma_scope = 'block'
parser.add_argument('--pragma_scope', default=pragma_scope)
keep_pragma_attribute = False if pragma_as_MLP else True
parser.add_argument('--keep_pragma_attribute', default=keep_pragma_attribute)
pragma_order = 'parallel_and_merge'
parser.add_argument('--pragma_order', default=pragma_order)
parser.add_argument('--pragma_MLP_hidden_channels', default=pragma_MLP_hidden_channels)
parser.add_argument('--merge_MLP_hidden_channels', default=merge_MLP_hidden_channels)


model_path = None
model_path_list = []
use_pretrain = True 
if use_pretrain:
    base_path = 'models'    
    keyword =  v_db
    includes = [keyword, model_ver, 'regression']
    excludes = ['class']
    model_base_path = join(get_root_path(), base_path, '**/*')
    model = [f for f in iglob(model_base_path, recursive=True) if f.endswith('.pth') and all(k not in f for k in excludes) and all(k in f for k in includes)]
    print(model)
    assert len(model) == 1
    model_path = model[0]
    model_path_list.append(model_path)

if model_path_list != []:
    model_path = model_path_list
parser.add_argument('--model_path', default=model_path) ## list of models when used in DSE, if more than 1, ensemble inference must be on

ensemble = 0
ensemble_weights = None
parser.add_argument('--ensemble', type=int, default=ensemble)
parser.add_argument('--ensemble_weights', default=ensemble_weights)
class_model_path = None
if SUBTASK == 'dse':
    keyword =  v_db
    includes = [keyword, model_ver, 'class']
    model = [f for f in iglob(model_base_path, recursive=True) if f.endswith('.pth') and all(k in f for k in includes)]
    assert len(model) == 1
    class_model_path = model[0]
parser.add_argument('--class_model_path', default=class_model_path)


################ transfer learning #################
feature_extract = False
parser.add_argument('--feature_extract', default=feature_extract) # if set to true GNN encoder (or part of it) will be fixed and only MLP will be trained
if feature_extract:
    parser.add_argument('--random_MLP', default=False) # true: initialize MLP randomly
fix_gnn_layer = None ## if none, all layers will be fixed
fix_gnn_layer = 1 ## number of gnn layers to freeze, feature_extract should be set to True
parser.add_argument('--fix_gnn_layer', default=fix_gnn_layer) # if not set to none, feature_extract should be True
FT_extra = False
parser.add_argument('--FT_extra', default=FT_extra) ## fine-tune only on the new data points


################ training details #################
parser.add_argument('--save_model', type = bool, default=True)
resample = False
val_ratio = 0.15
parser.add_argument('--resample', default=resample) ## when resample is turned on, it will divide the dataset in round-robin and train multiple times to have all the points in train/test set
parser.add_argument('--val_ratio', type=float, default=val_ratio) # ratio of database for validation set
parser.add_argument('--activation', default='elu')     
parser.add_argument('--D', type=int, default=64)
parser.add_argument('--lr', default=0.001) ## default=0.001
scheduler, warmup, weight_decay = None, None, 0
scheduler, warmup, weight_decaty = 'cosine', 'linear', 1e-4
parser.add_argument('--weight_decay', default=weight_decay) ## default=0.0001, larger than 1e-4 didn't help original graph P+T
parser.add_argument("--scheduler", default=scheduler)
parser.add_argument("--warmup", default=warmup)

parser.add_argument('--random_seed', default=123) ## default=100
batch_size = 64
parser.add_argument('--batch_size', type=int, default=batch_size)

loss = 'MSE' # RMSE, MSE, 
parser.add_argument('--loss', type=str, default=loss) 

if model_path == None:
    if TASK == 'regression':
        epoch_num = 1500
    else:
        epoch_num = 200
else:
    epoch_num = 200

parser.add_argument('--epoch_num', type=int, default=epoch_num)

gpu = 0
device = str('cuda:{}'.format(gpu) if torch.cuda.is_available() and gpu != -1
             else 'cpu')
parser.add_argument('--device', default=device)



################# DSE details ##################
explorer = 'exhaustive'
parser.add_argument('--explorer', default=explorer)

model_tag = 'test'
parser.add_argument('--model_tag', default=model_tag)

parser.add_argument('--prune_util', default=True) # only DSP and BRAM
parser.add_argument('--prune_class', default=True)

parser.add_argument('--print_every_iter', type=int, default=100)

plot = True
parser.add_argument('--plot_pred_points', type=bool, default=plot)

"""
Other info.
"""
parser.add_argument('--user', default=get_user())

parser.add_argument('--hostname', default=get_host())

FLAGS = parser.parse_args()
