## poly kernels
kernels=(atax bicg doitgen-red gemm-p gemver trmm-opt bicg-medium 
        doitgen gesummv 2mm 3mm symm syrk mvt trmm correlation 
        fdtd-2d adi atax-medium gesummv-medium mvt-medium jacobi-1d 
        jacobi-2d heat-3d) 
num_kernels_poly=${#kernels[@]}

benchmark=()
for ((i=0; i<num_kernels_poly; i++)); do
    benchmark+=(poly)
done

## machsuite kernels
kernels+=(aes gemm-blocked gemm-ncubed spmv-ellpack stencil nw stencil-3d) 
num_kernels_mach=${#kernels[@]}

for ((i=num_kernels_poly; i<num_kernels_mach; i++)); do
    benchmark+=(machsuite)
done
echo $HOSTNAME


for i in  ${!kernels[@]}  
do
    echo ${kernels[$i]} , ${benchmark[$i]} 
    python3 parallel_run_tool_dse.py --version 'v21' --kernel ${kernels[$i]} --benchmark ${benchmark[$i]} --root-dir ../ --redis-port "6379" --server $HOSTNAME
done
