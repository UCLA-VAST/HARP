import pickle
import pickle5
import redis
from os.path import join, dirname, basename, exists
import os
import argparse
from glob import iglob
from copy import deepcopy

from utils import get_ts, create_dir_if_not_exists, get_src_path, get_host
import time
from subprocess import Popen, PIPE
from result import Result

model_tag = 'pragma_as_MLP'
VER = 'v20' 
class MyTimer():
    def __init__(self) -> None:
        self.start = time.time()
    
    def elapsed_time(self):
        end = time.time()
        minutes, seconds = divmod(end - self.start, 60)
        
        return int(minutes)

class Saver():
    def __init__(self, kernel):
        self.logdir = join(
            get_src_path(),
            'logs',
            f'{VER}/hls_results', f'{kernel}_{get_ts()}')
        create_dir_if_not_exists(self.logdir)
        self.timer = MyTimer()
        print('Logging to {}'.format(self.logdir))

    def _open(self, f):
        return open(join(self.logdir, f), 'w')
    
    def info(self, s, silent=False):
        elapsed = self.timer.elapsed_time()
        if not silent:
            print(f'[{elapsed}m] INFO: {s}')
        if not hasattr(self, 'log_f'):
            self.log_f = self._open('log.txt')
        self.log_f.write(f'[{elapsed}m] INFO: {s}\n')
        self.log_f.flush()
        
    def error(self, s, silent=False):
        elapsed = self.timer.elapsed_time()
        if not silent:
            print(f'[{elapsed}m] ERROR: {s}')
        if not hasattr(self, 'log_e'):
            self.log_e = self._open('error.txt')
        self.log_e.write(f'[{elapsed}m] ERROR: {s}\n')
        self.log_e.flush()
        
    def warning(self, s, silent=False):
        elapsed = self.timer.elapsed_time()
        if not silent:
            print(f'[{elapsed}m] WARNING: {s}')
        if not hasattr(self, 'log_f'):
            self.log_f = self._open('log.txt')
        self.log_f.write(f'[{elapsed}m] WARNING: {s}\n')
        self.log_f.flush()
        
    def debug(self, s, silent=True):
        elapsed = self.timer.elapsed_time()
        if not silent:
            print(f'[{elapsed}m] DEBUG: {s}')
        if not hasattr(self, 'log_d'):
            self.log_d = self._open('debug.txt')
        self.log_d.write(f'[{elapsed}m] DEBUG: {s}\n')
        self.log_d.flush()

def gen_key_from_design_point(point) -> str:

    return '.'.join([
        '{0}-{1}'.format(pid,
                         str(point[pid]) if point[pid] else 'NA') for pid in sorted(point.keys())
    ])

def kernel_parser() -> argparse.Namespace:
    """Parse user arguments."""

    parser_run = argparse.ArgumentParser(description='Running Queries')
    parser_run.add_argument('--kernel',
                        required=True,
                        action='store',
                        help='Kernel Name')
    parser_run.add_argument('--benchmark',
                        required=True,
                        action='store',
                        help='Benchmark Name')
    parser_run.add_argument('--root-dir',
                        required=True,
                        action='store',
                        default='.',
                        help='GNN Root Directory')
    parser_run.add_argument('--redis-port',
                        required=True,
                        action='store',
                        default='6379',
                        help='The port number for redis database')
    parser_run.add_argument('--version',
                        required=True,
                        action='store',
                        default='v18',
                        help='The version of the Xilinx tool')
    parser_run.add_argument('--server',
                        required=False,
                        action='store',
                        default=get_host(),
                        help='The container ID')
    

    return parser_run.parse_args()
    
def persist(database, db_file_path, id=0) -> bool:
    dump_db = {
        key: database.hget(id, key)
        for key in database.hgetall(id)
    }
    with open(db_file_path, 'wb') as filep:
        pickle.dump(dump_db, filep, pickle.HIGHEST_PROTOCOL)

    return True

def update_best(result_summary, key, result):
    thresh = 0.80000001
    if result.perf > 16.0 and result.perf < result_summary['min_perf']:
        is_min = True
    else:
        return
    max_utils = {'BRAM': thresh, 'DSP': thresh, 'LUT': thresh, 'FF': thresh}
    utils = {k[5:]: max(0.0, u) for k, u in result.res_util.items() if k.startswith('util-')}
    valid = all([(utils[res])< max_utils[res] for res in max_utils])
    if valid:
        result_summary['min_perf'] = result.perf
        result_summary['key_min_perf'] = key
        result_summary[key] = deepcopy(result)

def run_procs(saver, procs, database, kernel, f_db_new, result_summary, server=None):
    saver.info(f'Launching a batch with {len(procs)} jobs')
    try:
        while procs:
            prev_procs = list(procs)
            procs = []
            for p_list in prev_procs:
                text = 'None'
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

                    q_result = pickle.load(open(f'localdse/kernel_results/{kernel}_{idx}_{server}.pickle', 'rb'))

                    for _key, result in q_result.items():
                        pickled_result = pickle.dumps(result)
                        if 'lv2' in key:
                            database.hset(0, _key, pickled_result)
                            database.hset(1, _key, pickled_result)
                        saver.info(f'Performance for {_key}: {result.perf} with return code: {result.ret_code} and resource utilization: {result.res_util}')
                        update_best(result_summary, key, result)
                    persist(database, f_db_new)
                # Still running
                else:
                    procs.append([idx, key, p])
                
                time.sleep(1)
    except:
        saver.error(f'Failed to finish the processes')
        raise RuntimeError()


if __name__ == '__main__':
    args = kernel_parser()
    saver = Saver(args.kernel)
    CHECK_EARLY_REJECT = False

    os.makedirs('./localdse/kernel_results/', exist_ok=True)
    ## update makefile path
    p = Popen(f"cd ../dse_database/merlin_prj \n python3 create_prj.py", shell = True, stdout=PIPE)
    src_dir = join(args.root_dir, 'dse_database/merlin_prj', f'{args.kernel}', 'xilinx_dse')
    work_dir = join('logs/expr', f'{args.kernel}', 'work_dir')
    f_config = join(args.root_dir, 'dse_database', args.benchmark, 'config', f'{args.kernel}_ds_config.json')
    f_pickle_path = join(args.root_dir, 'src/logs/', '**/*') 
    f_pickle_list = [f for f in iglob(f_pickle_path, recursive=True) if f.endswith('.pickle') and f'{args.kernel}.' in f and args.version in f] 
    print(f_pickle_list, f_pickle_path)
    assert len(f_pickle_list) == 1
    f_pickle = f_pickle_list[0]
    db_dir = join(args.root_dir, 'dse_database', args.benchmark, 'databases', '**')
    result_dict = pickle5.load(open(f_pickle, "rb" ))
    create_dir_if_not_exists(dirname(work_dir))
    create_dir_if_not_exists(work_dir)

    found_db = False
    if args.version in ['v18', 'v20', 'v21']:
        f_db_list = [f for f in iglob(db_dir, recursive=True) if f.endswith('.db') and f'{args.kernel}_' in f and args.version in f]
    else:
        raise NotImplementedError()
    if len(f_db_list) == 1:
        f_db = f_db_list[0]
        print(f_db)
        f_db_new = join(dirname(saver.logdir), 'all_db', basename(f_db).replace(f'_updated', f'_updated-merged'))
        f_db_new_this_round = join(dirname(saver.logdir), 'all_db', basename(f_db).replace(f'_updated', f'_updated-this-round'))
        create_dir_if_not_exists(dirname(f_db_new))
        found_db = True

    database = redis.StrictRedis(host='localhost', port=int(args.redis_port))
    database.flushdb()
    try:
        file_db = open(f_db, 'rb')
        data = pickle.load(file_db)
        database.hset(0, mapping=data)
    except:
        f_db_new = f'{args.kernel}_result_updated.db'
        saver.info('No prior databases')

    share = '/share/atefehSZ'
    batch_num = 5
    batch_id = 0
    procs = []
    saver.info(f"""processing {f_pickle} 
        from db: {f_db} and 
        updating to {f_db_new}""")
    saver.info(f"total of {len(result_dict.keys())} solution(s)")

    if args.version == 'v18':
        database.hset(0, 'setup', pickle.dumps({'tool_version': 'SDx-18.3'}))
        database.hset(1, 'setup', pickle.dumps({'tool_version': 'SDx-18.3'}))
    elif args.version == 'v20':
        database.hset(0, 'setup', pickle.dumps({'tool_version': 'Vitis-20.2'}))
        database.hset(1, 'setup', pickle.dumps({'tool_version': 'Vitis-20.2'}))
    elif args.version == 'v21':
        database.hset(0, 'setup', pickle.dumps({'tool_version': 'Vitis-21.1'}))
        database.hset(1, 'setup', pickle.dumps({'tool_version': 'Vitis-21.1'}))
    else:
        raise NotImplementedError()
        
    min_perf = float("inf")
    result_summary = {'key_min_perf': None, 'min_perf': float("inf")}
    hls_result_dict = {}
    for result_key, result in sorted(result_dict.items()):
        if hasattr(result, 'point'):
            point_ = result.point
        else:
            point_ = result
        for key_, value in point_.items():
            if type(value) is str or type(value) is int:
                point_[key_] = value
            else:
                point_[key_] = value.item()
        key = f'lv2:{gen_key_from_design_point(point_)}'
        lv1_key = key.replace('lv2', 'lv1')
        isEarlyRejected = False
        rerun = False
        if CHECK_EARLY_REJECT and database.hexists(0, lv1_key):
            pickle_obj = database.hget(0, lv1_key)
            obj = pickle.loads(pickle_obj)
            if obj.ret_code.name == 'EARLY_REJECT':
                isEarlyRejected = True
        
        if database.hexists(0, key):
            pickled_obj = database.hget(0, key)
            obj = pickle.loads(pickled_obj)
            if obj.perf == 0.0:
                rerun = True

        if rerun or (not isEarlyRejected and not database.hexists(0, key)):
            hls_result_dict[result_key] = result
            pass
        elif isEarlyRejected:
            pickled_obj = database.hget(0, lv1_key)
            obj = pickle.loads(pickled_obj)
            result.actual_perf = 0
            result.ret_code = Result.RetCode.EARLY_REJECT
            result.valid = False
            saver.info(f'LV1 Key exists for {key}, EARLY_REJECT')
        else:
            pickled_obj = database.hget(0, key)
            obj = pickle.loads(pickled_obj)
            result.actual_perf = obj.perf
            saver.info(f'Key exists. Performance for {key}: {result.actual_perf} with return code: {result.ret_code} and resource utilization: {obj.res_util}')
            update_best(result_summary, key, obj)
            database.hset(1, key, pickled_obj)
            
        
    if len(hls_result_dict) == 0:
        persist(database, f_db_new_this_round, id=1)

    for _, result in sorted(hls_result_dict.items()):
        if hasattr(result, 'point'):
            point_ = result.point
        else:
            point_ = result
        if 'ellpack' in args.kernel:
            break
        if len(procs) == batch_num:
            run_procs(saver, procs, database, args.kernel, f_db_new, result_summary, args.server)
            batch_id == 0
            procs = []
        for key_, value in point_.items():
            if type(value) is str or type(value) is int:
                point_[key_] = value
            else:
                point_[key_] = value.item()
        key = f'lv2:{gen_key_from_design_point(point_)}'
        # print(key)
        lv1_key = key.replace('lv2', 'lv1')
        isEarlyRejected = False
        rerun = False
        if CHECK_EARLY_REJECT and database.hexists(0, lv1_key):
            pickle_obj = database.hget(0, lv1_key)
            obj = pickle.loads(pickle_obj)
            if obj.ret_code.name == 'EARLY_REJECT':
                isEarlyRejected = True
        
        if database.hexists(0, key):
            pickled_obj = database.hget(0, key)
            obj = pickle.loads(pickled_obj)
            if obj.perf == 0.0:
                rerun = True

        assert rerun or (not isEarlyRejected and not database.hexists(0, key))

        kernel = args.kernel
        result_file = f'./localdse/kernel_results/{args.kernel}_point_{batch_id}_{args.server}.pickle'
        hls_results_file = f'localdse/kernel_results/{args.kernel}_{batch_id}_{args.server}.pickle'
        
        if exists(hls_results_file): os.remove(hls_results_file)
            
        with open(result_file, 'wb') as handle:
            pickle.dump(point_, handle, protocol=pickle.HIGHEST_PROTOCOL)
        new_work_dir = join(work_dir, f'batch_id_{batch_id}')
        if args.version == 'v18':
            env, docker, time_ = 'env', 'docker', "faketime -f '-2y'"
        elif args.version == 'v20' or args.version == 'v21':
            env, docker, time_ = 'vitis_env', 'vitis_docker', ''
        else:
            raise NotImplementedError()
        
        ### UPDATE ME: update this line by how you run the Merlin compiler
        p = Popen(f"cd {get_src_path()} \n source {share}/{env}.sh \n /expr/merlin_docker/{docker}-run-gnn.sh {time_} python3 -m autodse.explorer.single_query --src-dir {src_dir} --work-dir {new_work_dir} --kernel {kernel} --config {f_config} --id {batch_id} --server {args.server} --timeout 160", shell = True, stdout=PIPE)
        
        procs.append([batch_id, key, p])
        saver.info(f'Added {point_} with batch id {batch_id}')
        batch_id += 1

    if len(procs) > 0:
        run_procs(saver, procs, database, args.kernel, f_db_new, result_summary, args.server)

    persist(database, f_db_new_this_round, id=1)

    if result_summary['key_min_perf'] != None:
        saver.info('#####################################')
        key = result_summary['key_min_perf']
        result = result_summary[key]
        saver.info(f'Min perf is {result.perf} for {key} with return code: {result.ret_code} and resource utilization: {result.res_util}')
    else:
        saver.info('#####################################')
        saver.info('No valid point generated.')
        
                
        

    try:
        file_db.close()
    except:
        print('file_db is not defined')

