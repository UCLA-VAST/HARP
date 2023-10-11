## poly kernels
kernels=(atax bicg doitgen-red gemm-p gemver trmm-opt bicg-medium doitgen gesummv 2mm 3mm symm syrk mvt trmm correlation fdtd-2d adi atax-medium gesummv-medium mvt-medium jacobi-1d jacobi-2d heat-3d) 
benchmark=poly
## machsuite kernels
kernels=(aes gemm-blocked gemm-ncubed spmv-ellpack stencil nw stencil-3d) 
benchmark=machsuite
echo $HOSTNAME


for kernel in  ${kernels[@]}  
do
    python3 parallel_run_tool_dse.py --version 'v20' --kernel $kernel --benchmark $benchmark --root-dir ../ --redis-port "7777" --server $HOSTNAME
done
