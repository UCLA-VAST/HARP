from utils import get_root_path, create_dir_if_not_exists
from graph_gen import run_graph_gen

import shutil
from tempfile import mkstemp
from os.path import join, dirname, exists
import os

# project_dir: the path to xilinx_dse
# kernel_path: get it with parse_makefile_kernel_name
# db_base_path
# db_shared_path
# new_kernel_name
# benchmark

kernel_name = 'symm-opt-medium'
category = 'linear-algebra/blas' ## 'linear-algebra/kernels' ## 'linear-algebra/blas', 'datamining'
project_dir=f'/share/atefehSZ/polybench-c-4.2.1-beta/{category}/{kernel_name}_dse/xilinx_dse'
new_kernel_name='symm-opt-medium'
benchmark='poly'

MACHSUITE_KERNEL = ['md']
MACHSUITE_KERNEL = []
poly_KERNEL = [kernel_name]

# MACHSUITE_KERNEL = ['aes', 'gemm-blocked', 'gemm-ncubed', 'spmv-crs', 'spmv-ellpack', 'stencil',
#                     'nw', 'md', 'stencil-3d']

# poly_KERNEL = ['2mm', '3mm', 'adi', 'atax', 'bicg', 'bicg-large', 'covariance', 'doitgen', 
#                'doitgen-red', 'fdtd-2d', 'fdtd-2d-large', 'gemm-p', 'gemm-p-large', 'gemver', 
#                'gesummv', 'heat-3d', 'jacobi-1d', 'jacobi-2d', 'mvt', 'seidel-2d', 'symm', 
#                'symm-opt', 'syrk', 'syr2k', 'trmm', 'trmm-opt', 'mvt-medium', 'correlation',
#                'atax-medium', 'bicg-medium', 'gesummv-medium']
ALL_KERNEL = {'machsuite': MACHSUITE_KERNEL, 'poly': poly_KERNEL}

def get_local_project_dir():
    return join(get_root_path(), 'save/merlin_prj')

def get_makefile_path(project_dir):
    return join(project_dir, 'Makefile')

def parse_makefile_kernel_name(project_dir):
    ''''
        parse the makefile to get the location to the kernel and its name
    '''
    m_file = get_makefile_path(project_dir)
    with open(m_file, 'r') as f:
        for line in f:
            if 'KERNEL_SRC_FILES' in line:
                file_path = line.strip().split('=')[-1]
                break
    abs_path = join(dirname(m_file), file_path)
    name = file_path.split('/')[-1]
    return {'path': abs_path,
            'name': name}
    
    
def modify_makefile_kernel_name(project_dir, kernel_name):
    '''
        change the mcc_common path in makefile
    '''
    orig_file = get_makefile_path(project_dir)

    fnew, abs_path = mkstemp()
    with open(abs_path, 'w') as fpnew:
        with open(orig_file) as f:
            for line in f:
                if 'KERNEL_SRC_FILES=' in line:
                    fpnew.write(line.replace(line.strip().split('=')[-1], f'./{kernel_name}'))
                else:
                    fpnew.write(line)
    
    shutil.copymode(orig_file, abs_path)
    shutil.copy(abs_path, orig_file)

def modify_makefile_mcc_common(project_dir):
    '''
        change the mcc_common path in makefile
    '''
    orig_file = get_makefile_path(project_dir)

    fnew, abs_path = mkstemp()
    with open(abs_path, 'w') as fpnew:
        with open(orig_file) as f:
            for line in f:
                if 'MCC_COMMON_DIR=' in line:
                    fpnew.write(line.replace(line.strip().split('=')[-1], '/expr/mcc_common'))
                else:
                    fpnew.write(line)
    
    shutil.copymode(orig_file, abs_path)
    shutil.copy(abs_path, orig_file)


def copy_project_dir(project_dir, new_kernel_name):
    dst = join(get_local_project_dir(), new_kernel_name)
    create_dir_if_not_exists(dst)
    new_project_dir = join(dst, 'xilinx_dse')
    if exists(new_project_dir):
        print(f'deleting the existing files in {new_project_dir}')
        shutil.rmtree(new_project_dir)
    shutil.copytree(project_dir, new_project_dir)
    modify_makefile_mcc_common(new_project_dir)
    kernel_info = parse_makefile_kernel_name(new_project_dir)
    return kernel_info


def copy_kernel_file(kernel_path, kernel_name, benchmark, new_kernel_name):
    ext = kernel_name.split('.')[-1]
    new_file_path = join(get_root_path(), benchmark, 'sources', f'{new_kernel_name}_kernel.{ext}')
    if exists(new_file_path): os.remove(new_file_path)
    shutil.copyfile(kernel_path, new_file_path)


def copy_ds_info_file(kernel_path, benchmark, new_kernel_name):
    file_path = join(dirname(kernel_path), 'ds_info.json')
    new_file_path = join(get_root_path(), benchmark, 'config', f'{new_kernel_name}_ds_config.json')
    if exists(new_file_path): os.remove(new_file_path)
    os.system(f'cp {file_path} {new_file_path}')

def create_programl_project(kernel_path, kernel_name, benchmark, new_kernel_name):
    ext = kernel_name.split('.')[-1]
    new_dir = join(get_root_path(), 'generated_graphs', benchmark, f'{new_kernel_name}')
    if exists(new_dir):
        print(f'deleting the existing files in {new_dir}')
        shutil.rmtree(new_dir)
    create_dir_if_not_exists(new_dir)
    new_file_path = join(new_dir, f'{new_kernel_name}.{ext}')
    shutil.copyfile(kernel_path, new_file_path)
    
    
def create_merlin_prj(kernel_name, kernel_path):
    dst = join(get_local_project_dir(), kernel_name)
    new_project_dir = join(dst, 'xilinx_dse')
    create_dir_if_not_exists(new_project_dir)
    shutil.copy(kernel_path, new_project_dir)
    makefile_path = get_makefile_path(join(get_root_path(), 'merlin_prj'))
    shutil.copy(makefile_path, new_project_dir)
    modify_makefile_mcc_common(new_project_dir)
    modify_makefile_kernel_name(new_project_dir, basename(kernel_path))

if __name__ == '__main__':
    modify_kernel = False
    create_prj = False
    if create_prj:
        kernel_info = {'name': '2mm',
                   'path': './demo/2mm.c'}

        #### 1. create Merlin project
        create_merlin_prj(kernel_name=kernel_info['name'], 
                        kernel_path=kernel_info['path'])
        print('finished step 1 (merlin project creation)')
    elif modify_kernel:
        for b in ['machsuite', 'poly']:
            all_kernels = ALL_KERNEL[b]
            for kernel in all_kernels:
                dst = join(get_local_project_dir(), kernel)
                new_project_dir = join(dst, 'xilinx_dse')
                if not exists(new_project_dir):
                    print(f'no {new_project_dir}')
                    raise NotImplementedError()
                modify_makefile_mcc_common(new_project_dir)
    else:
        #### 1. copy the xilinx_dse into save/merlin_prj
        kernel_info = copy_project_dir(
                    project_dir=project_dir, 
                    new_kernel_name=new_kernel_name)
        print('finished step 1 (merlin project creation/copy)')

        #### 2. copy the kernel to its benchmark directory
        copy_kernel_file(kernel_path=kernel_info['path'],
                        kernel_name=kernel_info['name'],
                        benchmark=benchmark,
                        new_kernel_name=new_kernel_name)
        print('finished step 2 (kernel file copy)')

        #### 3. copy the DS config to its benchmark directory
        copy_ds_info_file(kernel_path=kernel_info['path'],
                        benchmark=benchmark,
                        new_kernel_name=new_kernel_name)
        print('finished step 3 (ds file copy)')

        #### 4. generate the graphs (initial, auxiliary, and hierarcy) and store the node/edge counts
        create_programl_project(kernel_path=kernel_info['path'],
                        kernel_name=kernel_info['name'],
                        benchmark=benchmark,
                        new_kernel_name=new_kernel_name)
        ## the below function can only be run on u7
        run_graph_gen(mode='initial', connected=True, target=[benchmark], ALL_KERNEL=ALL_KERNEL)
        run_graph_gen(mode='auxiliary', connected=False, target=[benchmark], ALL_KERNEL=ALL_KERNEL)
        run_graph_gen(mode='auxiliary', connected=True, target=[benchmark], ALL_KERNEL=ALL_KERNEL)
        run_graph_gen(mode='hierarchy', connected=True, target=[benchmark], ALL_KERNEL=ALL_KERNEL)
        print('finished step 4 (graph generation)')

        #### 5. copy and merge the databases

