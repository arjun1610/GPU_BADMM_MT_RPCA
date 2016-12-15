rm badmm_mt 
nvcc -o badmm_mt thrust_badmm_mt.cu badmm_kernel.cu -lcublas -lcusolver
./badmm_mt 200C.dat
