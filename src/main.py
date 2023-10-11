from config import FLAGS
from train import train_main, inference
from dse import ExhaustiveExplorer
from saver import saver
from utils import get_root_path, load, get_src_path, plot_dist, plot_models_per_graph

from os.path import join, dirname
from glob import iglob

import config
TARGETS = config.TARGETS
MACHSUITE_KERNEL = config.MACHSUITE_KERNEL
poly_KERNEL = config.poly_KERNEL

from data import get_data_list, MyOwnDataset
import data

if __name__ == '__main__':

    if not FLAGS.force_regen:
        dataset = MyOwnDataset()
        print('read dataset')
    else:   
        pragma_dim = 0
        dataset, pragma_dim = get_data_list()

    if FLAGS.encoder_path is not None:
        pragma_dim = load(join(dirname(FLAGS.encoder_path), 'v18_pragma_dim'))
        
        
    def inf_main(dataset):
        if type(FLAGS.model_path) is None:
            saver.error('model_path must be set for running the inference.')
            raise RuntimeError()
        else:
            for ind, model_path in enumerate(FLAGS.model_path):
                if FLAGS.val_ratio > 0.0:
                    inference(dataset, init_pragma_dict=pragma_dim, model_path=model_path, model_id=ind, test_ratio=FLAGS.val_ratio)
                    inference(dataset, init_pragma_dict=pragma_dim, model_path=model_path, model_id=ind, test_ratio=FLAGS.val_ratio, is_val_set=True)
                inference(dataset, init_pragma_dict=pragma_dim, model_path=model_path, model_id=ind, test_ratio=FLAGS.val_ratio, is_train_set=True)
                if ind + 1 < len(FLAGS.model_path):
                    saver.new_sub_saver(subdir=f'run{ind+2}')
                    saver.log_info('\n\n')


    if FLAGS.subtask == 'inference':
        inf_main(dataset)
    elif FLAGS.subtask == 'dse':
        if FLAGS.dataset == 'harp':
            first_dse = True
            if FLAGS.plot_dse: graph_types = ['initial', 'extended', 'hierarchy']
            else: graph_types = [FLAGS.graph_type]

            for dataset in ['machsuite', 'poly']:
                path = join(get_root_path(), 'dse_database', dataset, 'config')
                path_graph = join(get_root_path(), 'dse_database', 'generated_graphs', dataset, 'processed')
                if dataset == 'machsuite':   
                    KERNELS = MACHSUITE_KERNEL
                elif dataset == 'poly':
                    KERNELS = poly_KERNEL
                else:
                    raise NotImplementedError()
                
                for kernel in KERNELS:
                    if not FLAGS.all_kernels and not FLAGS.target_kernel in kernel:
                        continue
                    plot_data = {}
                    for graph_type in graph_types:
                        saver.info('#'*65)
                        saver.info(f'Now processing {graph_type} graph')
                        saver.info('*'*65)
                        saver.info(f'Starting DSE for {kernel}')
                        saver.debug(f'Starting DSE for {kernel}')
                        FLAGS.target_kernel = kernel
                        if FLAGS.explorer == 'exhaustive':
                            explorer = ExhaustiveExplorer(path, kernel, path_graph, first_dse = first_dse, run_dse = True, pragma_dim = pragma_dim)
                            if FLAGS.plot_dse: plot_data[graph_type] = explorer.plot_data
                        else:
                            raise NotImplementedError()
                        saver.info('*'*65)
                        saver.info(f'')
                        first_dse = False

                    if FLAGS.plot_dse:
                        plot_models_per_graph(saver.plotdir, kernel, graph_types, plot_data, FLAGS.target)
        else:
            raise NotImplementedError()
    elif FLAGS.subtask == 'train':
        test_ratio, resample_list = FLAGS.val_ratio, [-1]
        if FLAGS.resample:
            test_ratio, resample_list = 0.25, range(4)
        for ind, r in enumerate(resample_list):
            saver.info(f'Starting training with resample {r}')
            test_data = train_main(dataset, pragma_dim, test_ratio=test_ratio, resample=r)
            if ind + 1 < len(resample_list):
                saver.new_sub_saver(subdir=f'run{ind+2}')
                saver.log_info('\n\n')
                
    else:
        raise NotImplementedError()

    saver.close()
