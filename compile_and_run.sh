 nvcc -o badmm_mt badmm_mt.cu badmm_kernel.cu -lcublas -lcusolver
./badmm_mt 200C.dat
