from os.path import abspath, dirname, exists, join
from os import makedirs
import os
import shutil
from tempfile import mkstemp

MACHSUITE_KERNEL = ['aes', 'gemm-blocked', 'gemm-ncubed', 'spmv-crs', 'spmv-ellpack', 'stencil_stencil2d',
                    'nw', 'md', 'stencil-3d']

poly_KERNEL = ['2mm', '3mm', 'adi', 'atax', 'bicg', 'bicg-large', 'covariance', 'doitgen', 
               'doitgen-red', 'fdtd-2d', 'fdtd-2d-large', 'gemm-p', 'gemm-p-large', 'gemver', 
               'gesummv', 'heat-3d', 'jacobi-1d', 'jacobi-2d', 'mvt', 'seidel-2d', 'symm', 
               'symm-opt', 'syrk', 'syr2k', 'trmm', 'trmm-opt', 'mvt-medium', 'correlation',
               'atax-medium', 'bicg-medium', 'gesummv-medium', 'symm-opt-medium',
               'gemver-medium']

kernels = {'machsuite': MACHSUITE_KERNEL, 'poly': poly_KERNEL}

def get_cur_dir():
    return dirname(abspath(__file__))

def get_base_project_dir():
    return join('/share/atefehSZ/RL/original-software-gnn/software-gnn/dse_database/save/merlin_prj')

def get_local_project_dir():
    return join('.')

def modify_makefile_mcc_common(orig_file):
    '''
        change the mcc_common path in makefile
    '''

    fnew, abs_path = mkstemp()
    with open(abs_path, 'w') as fpnew:
        with open(orig_file) as f:
            for line in f:
                if 'MCC_COMMON_DIR=' in line:
                    fpnew.write(line.replace(line.strip().split('=')[-1], f'{dirname(abspath(__file__))}/mcc_common'))
                else:
                    fpnew.write(line)
    
    shutil.copymode(orig_file, abs_path)
    shutil.copy(abs_path, orig_file)
    
def modify_makefile_kernel_name(orig_file, kernel_name):
    '''
        change the kernel path in makefile
    '''

    fnew, abs_path = mkstemp()
    with open(abs_path, 'w') as fpnew:
        with open(orig_file) as f:
            for line in f:
                if 'KERNEL_SRC_FILES=' in line:
                    print('here', line.replace(line.strip().split('=')[-1], f'./{kernel_name}/{kernel_name}.c'))
                    fpnew.write(line.replace(line.strip().split('=')[-1], f'./{kernel_name}/{kernel_name}.c'))
                else:
                    fpnew.write(line)
    
    shutil.copymode(orig_file, abs_path)
    shutil.copy(abs_path, orig_file)
    
    
def remove_include_file(orig_file):
    '''
        change the include file from kernel file
    '''

    fnew, abs_path = mkstemp()
    with open(abs_path, 'w') as fpnew:
        with open(orig_file) as f:
            for line in f:
                if '#include "merlin_type_define.h"' in line:
                    continue
                else:
                    fpnew.write(line)
    
    shutil.copymode(orig_file, abs_path)
    shutil.copy(abs_path, orig_file)
    
    
def create_merlin_prj(benchmark, kernel_name):
    dst = join(get_local_project_dir(), kernel_name)
    shutil.rmtree(dst)
    kernel_path = join('../', benchmark, 'sources', f'{kernel_name}_kernel.c')
    remove_include_file(kernel_path)
    ds_info_path = join('../', benchmark, 'config', f'{kernel_name}_ds_config.json')
    new_project_dir = join(dst, 'xilinx_dse', kernel_name)
    makefile_dir = join(dst, 'xilinx_dse')
    makedirs(new_project_dir, exist_ok=True)
    shutil.copyfile(kernel_path, join(new_project_dir, f'{kernel_name}.c'))
    shutil.copyfile(ds_info_path, join(new_project_dir, f'ds_info.json'))
    makefile_path = join('../Makefile')
    shutil.copy(makefile_path, makefile_dir)
    modify_makefile_mcc_common(join(makefile_dir, 'Makefile'))
    modify_makefile_kernel_name(join(makefile_dir, 'Makefile'), kernel_name)


def modify_merlin_prj(benchmark, kernel_name):
    dst = join(get_local_project_dir(), kernel_name)
    makefile_dir = join(dst, 'xilinx_dse')
    modify_makefile_mcc_common(join(makefile_dir, 'Makefile'))


# for dir_ in os.walk(get_cur_dir()):
#     if len(dir_[2]) > 0:
#         for file_ in dir_[2]:
#             if 'rose' in file_ and exists(os.path.join(dir_[0], file_)) and 'ds_info.json' not in file_:
#                 print(os.path.join(dir_[0], file_))
#                 # assert 'xilinx_dse' in os.path.join(dir_[0], file_)
                
#                 # modify_makefile_mcc_common(os.path.join(dir_[0], file_))
#                 os.system('rm -rf "%s"' % os.path.join(dir_[0], file_))

for benchmark in kernels:
    cur_kernels = kernels[benchmark]
    for k in cur_kernels:
        create_merlin_prj(benchmark, k)
                
