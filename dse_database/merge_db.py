import redis
import pickle5 as pickle
from collections import Counter
from os.path import join, basename, dirname
from glob import glob, iglob
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
import csv
import ast

from utils import get_root_path, create_dir_if_not_exists
from result import Result



MACHSUITE_KERNEL = ['aes', 'gemm-blocked', 'gemm-ncubed', 'spmv-crs', 'spmv-ellpack', 'stencil',
                    'nw', 'md', 'stencil-3d']

poly_KERNEL = ['2mm', '3mm', 'adi', 'atax', 'bicg', 'bicg-large', 'covariance', 'doitgen', 
               'doitgen-red', 'fdtd-2d', 'fdtd-2d-large', 'gemm-p', 'gemm-p-large', 'gemver', 
               'gesummv', 'heat-3d', 'jacobi-1d', 'jacobi-2d', 'mvt', 'seidel-2d', 'symm', 
               'symm-opt', 'syrk', 'syr2k', 'trmm', 'trmm-opt', 'mvt-medium', 'correlation',
               'atax-medium', 'bicg-medium', 'gesummv-medium']
VER = 'v18'
VER = 'v20'



def init_db(port=6379):
    database = redis.StrictRedis(host='localhost', port=port)
    database.flushdb()
    return database

def store_db(database, db_id, db_file_path) -> bool:
    dump_db = {
        key: database.hget(db_id, key)
        for key in database.hgetall(db_id)
    }
    with open(db_file_path, 'wb') as filep:
        pickle.dump(dump_db, filep, pickle.HIGHEST_PROTOCOL)

    return True

def commit_db(database, db_id, key, result):
    pickled_result = pickle.dumps(result)
    database.hset(db_id, key, pickled_result)
    
def get_db_files(db_path, kernel = None):
    db_files = [f for f in iglob(db_path, recursive=True) if f.endswith('.db') and f'{kernel}_' in f and VER in f] 
    
    return db_files

def add_if_not_exists_or_now_valid(database, new_database, len_pragma = 0):
    keys_new = [k.decode('utf-8') for k in new_database.hkeys(0)]
    for key in sorted(keys_new):
        if key == 'lv1:' or key == 'lv2:':
            database.hdel(0, key)
            continue
        if database.hexists(0, key):
            pickled_obj_new = new_database.hget(0, key)
            pickled_obj = database.hget(0, key)
            obj_new = pickle.loads(pickled_obj_new)
            obj = pickle.loads(pickled_obj)
            
            if type(obj_new) is int or type(obj_new) is dict:
                if type(obj_new) is dict:
                    if key == 'scope-map':
                        continue
                    assert obj == obj_new, f'{obj}, {obj_new}'
                elif obj_new > obj:
                    database.hset(0, key, pickle.dumps(obj_new))
                continue
            if 'lv' in key: assert len(obj_new.point) == len_pragma, print(len(obj.point), len_pragma)
            if obj_new.ret_code == 'EARLY_REJECT':
                obj_new.ret_code = Result.RetCode.EARLY_REJECT
            if obj.ret_code.name == obj_new.ret_code.name and obj.perf == obj_new.perf:
                continue
            elif obj.ret_code.name == obj_new.ret_code.name and obj.perf != obj_new.perf:
                print(f'invalid design config with key: {key}')
                print(f'the prev obj had perf {obj.perf} with ret code: {obj.ret_code.name}')
                print(f'but now has perf {obj_new.perf} with ret code: {obj_new.ret_code.name}')
                print()
                raise RuntimeError()
            elif obj_new.ret_code.name == 'PASS' or obj_new.ret_code.name == 'EARLY_REJECT':
                if pickled_obj_new:
                    database.hset(0, key, pickle.dumps(obj_new))
                if obj.ret_code.name == 'TIMEOUT':
                    continue
                print(f'for {key}')
                print(f'replaced prev point with perf {obj.perf} and ret code: {obj.ret_code.name}')
                print(f'with perf {obj_new.perf} with ret code: {obj_new.ret_code.name}')
                print()
            else:
                if obj_new.ret_code.name == 'TIMEOUT' and (obj.ret_code.name == 'PASS' or obj.ret_code.name == 'EARLY_REJECT'):
                    continue
                print(f'for {key}')
                print(f'ignored new point with perf {obj_new.perf} with ret code: {obj_new.ret_code.name}')
                print(f'prev point was with perf {obj.perf} and ret code: {obj.ret_code.name}')
                print()
        else:
            pickled_obj = new_database.hget(0, key)
            obj = pickle.loads(pickled_obj)
            if 'lv' in key: assert len(obj.point) == len_pragma, print(len(obj.point), len_pragma)
            if pickled_obj:
                if type(obj) is int or type(obj) is dict:
                    pass
                elif obj.ret_code == 'EARLY_REJECT':
                    obj.ret_code = Result.RetCode.EARLY_REJECT
                database.hset(0, key, pickle.dumps(obj))
    setup = database.hget(0, 'setup')
    if setup: obj_tool = pickle.loads(setup)
    else: 
        if VER == 'v18': database.hset(0, 'setup', pickle.dumps({'tool_version': 'SDx-18.3'}))
        elif VER == 'v20': database.hset(0, 'setup', pickle.dumps({'tool_version': 'Vitis-20.2'}))
        elif VER == 'v21': database.hset(0, 'setup', pickle.dumps({'tool_version': 'Vitis-21.1'}))
    if setup:
        if VER == 'v18': assert obj_tool['tool_version']== 'SDx-18.3'
        elif VER == 'v20': assert obj_tool['tool_version']== 'Vitis-20.2'
        elif VER == 'v21': assert obj_tool['tool_version']== 'Vitis-21.1'
        else: raise NotImplementedError()



def read_db(database, db_files):
    '''
    combine: merge all the databases as one or read them separately
    '''
    len_pragma = 0
    database.flushdb()
    for idx, file in enumerate(db_files):
        print('###########')
        print('now reading', file)
        print('###########')
        f_db = open(file, 'rb')
        data = pickle.load(f_db)
        if idx == 0:
            database.hset(0, mapping=data)
            keys = [k.decode('utf-8') for k in database.hkeys(idx) if 'lv2' in k.decode('utf-8')]
            obj = pickle.loads(database.hget(idx, keys[0]))
            len_pragma = len(obj.point)
        else:
            new_database = init_db(port=7777)
            new_database.hset(0, mapping=data)
            add_if_not_exists_or_now_valid(database, new_database, len_pragma)
        f_db.close()
            
def get_keys(database, num_db):
    all_keys = {}
    for n in range(num_db):
        keys = [k.decode('utf-8') for k in database.hkeys(n)]
        all_keys[n] = sorted(keys)
    return all_keys

def get_keys_with_id(database, id):
    all_keys = {}
    keys = [k.decode('utf-8') for k in database.hkeys(id)]
    all_keys[id] = sorted(keys)
    return all_keys[id]

def compare_points_and_keys(all_points, all_keys, first, second): 
    res = (all_points[first]).items() <= (all_points[second]).items()
    res_key = set(all_keys[first]).issubset(set(all_keys[second]))
    
    return res, res_key 
        

# run "redis-server" on command line first!
def merge_db_object(db_path, kernel = None, store_results = True):
    # create a redis database
    database = init_db()
    db_files = get_db_files(db_path, kernel)
    if len(db_files) == 0:
        print(f'Warning: no database file found for {kernel}')
        return ## no db was found for the given test
    db_files = sorted(db_files)
    pprint(db_files)
    # load the database and get the keys
    # the key for each entry shows the value of each of the pragmas in the source file
    read_db(database, db_files)
    idx_db = 0
    all_keys_0 = get_keys_with_id(database, idx_db)
    print(f'total number of keys: {len(all_keys_0)}')

    if len(all_keys_0) > 0 and store_results:
        store_db(database, idx_db, join(COMMON_DIR, f'{kernel}_result_merged-0.db'))
    


if __name__ == '__main__':
    store = True
    if store:        
        target = ['machsuite', 'poly']
        for dataset in target:
            COMMON_DIR = join(get_root_path(), f'{dataset}/databases/{VER}/') 
            if dataset == 'machsuite':
                KERNELS = MACHSUITE_KERNEL
            elif dataset == 'poly':
                KERNELS = poly_KERNEL
            create_dir_if_not_exists(COMMON_DIR)
            for kernel in KERNELS:
                if kernel != '2mm':
                    continue
                print(f'now processing {kernel}')
                merge_db_object(db_path=COMMON_DIR+ '/**/*', kernel=kernel)

                print('***********************************************')

   

    """
    # create a redis database
    database = redis.StrictRedis(host = 'localhost', port = 6379)
    
    # open a sample database
    f_db = open("./all_dbs/hog_result.db", "rb")
    
    # load the database and get the keys
    # the key for each entry shows the value of each of the pragmas in the source file
    data = pickle.load(f_db)
    database.hset(0, mapping=data)
    keys = [k.decode('utf-8') for k in database.hkeys(0)]
    # get a value of a specific key (key i)
    i = 6
    pickle_obj = database.hget(0, keys[i])
    obj = pickle.loads(pickle_obj)
    
    print(obj.ret_code, obj.perf, obj.res_util)  ## --> return code for the current configuration. checkout autodse/result.py for all available options
    # other variables that exists:
    ## obj.perf  --> the cycle counts of the given configuration
    ## obj.res_util --> the resource utilization for each type of the resourse (BRAM, DSP, LUT, FF)
    ##		    the variables "util_*" show the percentage of utilization
    ##		    the variables "total_*" show the exact number of resources used
    ## obj.eval_time --> how long it took for the tool to synthesize the given configuration
    ## obj.point --> the same as key. shows the value of each pragma
    ## obj.quality --> quality of the given design point as a measure of finite difference as defined in the paper 
    ##		   (ratio of difference of cycle and utilization compared to a base design)
    """
