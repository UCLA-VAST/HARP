cd $2
if [ $3 == 'multi_modality' ]
then
    clang-13 -emit-llvm -fno-discard-value-names -g -S -c $1.c -o $1.ll
else
    clang-13 -emit-llvm -fno-discard-value-names -S -c $1.c -o $1.ll
fi
# llvm2graph < $1.ll > $1.pbtxt
# graph2json < $1.pbtxt > $1.json



