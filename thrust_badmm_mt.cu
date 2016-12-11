/***********************************************************
By Huahua Wang, the University of Minnesota, twin cities
***********************************************************/

#include <stdio.h>
#include "badmm_kernel.cuh"
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include "cublas.h"
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

//#define MAX_GRID_SIZE 65535
//#define MAX_BLOCK_SIZE 1024

typedef struct GPUInfo
{
    unsigned int MAX_GRID_SIZE;
    unsigned int MAX_BLOCK_SIZE;
}GPUInfo;

typedef struct ADMM_para
{
    float rho;          // penalty parameter

    float* iter_obj;
    float* iter_time;
    float* iter_err;
    unsigned int MAX_ITER;          // MAX_ITER
    float tol;
    float abstol;
    float reltol;
    float lambda;

}ADMM_para;

typedef struct BADMM_massTrans
{
    int m;
    int n;
    int N;

    // we'll call our matrix A
    float* A;                       // row major order
    float* a;
    float* b;

    float g2;
    float g3;

    int print_step;

    bool SAVEFILE;
}BADMM_massTrans;

void matInit(float* &X, unsigned int size, float value);

/*
 * Thrust update functions for tranformations
 */
struct B_update {
    const float N_loc;

    B_update(float _N_loc) : N_loc(_N_loc) {}

    __host__ __device__
    float operator()(thrust::tuple<float,float,float,float,float> t) {
        float x1, x2, x3, a, u;
        thrust::tie(x1, x2, x3, a, u) = t;
        return (((x1 + x2 + x3) / 3.0f) - (a / N_loc) + u);
    }
};

struct X1_update {
    const float lamb;

    X1_update(float _lamb) : lamb(_lamb) {}

    __host__ __device__
    float operator()(const float &x1, const float &u) {
        return ((1.0f / (1.0f + lamb)) * (x1 - u));
    }
};


struct X2_update {
    const float temp;

    X2_update(float _temp) : temp(_temp) {}

    __host__ __device__
    float operator()(const float &x2, const float &u) {
        float v = x2 - u;
        float ans = (v - temp > 0.0f? v - temp : 0.0f) - (-v - temp > 0.0f? -v - temp : 0.0f);
        return ans;
    }
};

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

/*********************************************
Bregman ADMM for mass transportation problem
All matrices are in row major order
**********************************************/
void gpuBADMM_MT( BADMM_massTrans* &badmm_mt, ADMM_para* &badmm_para, GPUInfo* gpu_info)
{
    float *X_1, *X_2, *X_3, *U, *B;     // host (boyd code)

    // device
    float *d_A, *d_X_1, *d_X_2, *d_X_3, *d_U;
    float *d_z, *d_z_old, *d_Xmean, *d_X;
    float *d_svd_U, *d_svd_S, *d_svd_VH, *d_temp ;

    unsigned int m,n,N;
    m = badmm_mt->m;
    n = badmm_mt->n;
    N = badmm_mt->N;

    unsigned long int size = m*n;
    float fill_value = 0.0f;

    // local variables below for updates
    // set g2_max = norm(A(:), inf);
    // set g3_max = norm(A);

    // THEN UNCOMMENT
    // badmm_mt->g2 = 0.15 * g2_max;
    // badmm_mt->g3 = 0.15 * g3_max;
    // Let's hard code correct values from boyd for now

    badmm_mt->g2 = 0.14999999999;
    badmm_mt->g3 = 206.356410537;

    // GPU matrix
    cudaMalloc(&d_X_1, size*sizeof(float));
    cudaMalloc(&d_X_2, size*sizeof(float));
    cudaMalloc(&d_X_3, size*sizeof(float));
    cudaMalloc(&d_A, size*sizeof(float));
    cudaMalloc(&d_U, size*sizeof(float));

    cudaMalloc(&d_svd_U, m*m*sizeof(float));
    cudaMalloc(&d_svd_S, n*sizeof(float)); // size of min(m,n); cublasgesvd works for m>n
    cudaMalloc(&d_svd_VH, n*n*sizeof(float));
    cudaMalloc(&d_temp, size*sizeof(float));

    printf("Copying data from CPU to GPU ...\n");

    // copy A to GPU
    cudaMemcpy(d_A, badmm_mt->A, sizeof(float)*size, cudaMemcpyHostToDevice);

    // direct device allocation
    // this should be done on device kernel directly.
    thrust::device_ptr<float> dp_X1(d_X_1);
    thrust::device_ptr<float> dp_X2(d_X_2);
    thrust::device_ptr<float> dp_X3(d_X_3);
    thrust::device_ptr<float> dp_A(d_A);
    thrust::device_ptr<float> dp_U(d_U);

    // dont know if this initialization is compulsory.
    thrust::fill(dp_X1, dp_X1 + size, fill_value);
    thrust::fill(dp_X2, dp_X2 + size, fill_value);
    thrust::fill(dp_X3, dp_X3 + size, fill_value);
    thrust::fill(dp_U, dp_U + size, fill_value);
    // if necessary add SVD matrices initialization here.


    // grid and block size
    unsigned int block_size = size > gpu_info->MAX_BLOCK_SIZE ? gpu_info->MAX_BLOCK_SIZE : size;
    unsigned long int n_blocks = (int) (size+block_size-1)/block_size;
    if(n_blocks > gpu_info->MAX_GRID_SIZE) n_blocks = gpu_info->MAX_GRID_SIZE;
    printf("Block size %f b_blocks %f\n", block_size, n_blocks);

    unsigned int stride = block_size*n_blocks;

    printf("nblcoks = %d, block_size = %d, size = %d, stride = %d\n", n_blocks, block_size, size, stride);
    printf("BregmanADMM for mass transportation is running ...\n");

    cublasInit();

    float iter_obj;
    int iter, count = 0;
    // GPU time
    float milliseconds = 0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for ( iter = 0; iter < badmm_para->MAX_ITER; iter++ )
    {
        // update B here as in admm boyd - B and U are the same - we'll use U for both
        thrust::transform(
            thrust::make_zip_iterator(thrust::make_tuple(dp_X1, dp_X2, dp_X3, dp_A, dp_U)),
            thrust::make_zip_iterator(thrust::make_tuple(dp_X1+size, dp_X2+size, dp_X3+size, dp_A+size, dp_U+size)),
            dp_U,
            B_update(N)
        );

        // X_1 update
        thrust::transform(
            dp_X1, dp_X1 + size, dp_U, dp_X1, X1_update(badmm_para->lambda)
        );

        // X_2 update
        float temp = badmm_para->lambda * badmm_mt->g2;
        thrust::transform(
            dp_X2, dp_X2 + size, dp_U, dp_X2, X2_update(temp)
        );


        // X_3 update
        // perform X_3 - B and store in X_3

        thrust::transform(
            dp_X3, dp_X3 + size, dp_U, dp_X3, thrust::minus<float>() 
        );

        // svd code coming in
        // --- CUDA solver initialization
        int *devInfo;
        //can do a gpuErrchk on all the cuda Mallocs
        cudaMalloc(&devInfo, sizeof(int));
        cusolverStatus_t stat;
        cusolverDnHandle_t solver_handle;
        cusolverDnCreate(&solver_handle);

        int work_size = 0;
        stat = cusolverDnSgesvd_bufferSize(solver_handle, m, n, &work_size);
        if(stat != CUSOLVER_STATUS_SUCCESS ) std::cout << "Initialization of cuSolver failed. \n";

        float *work;
        gpuErrchk(cudaMalloc(&work, work_size * sizeof(float)));

        // --- CUDA SVD execution
        stat = cusolverDnSgesvd(solver_handle, 'A', 'A', m, n, d_X_3, m, d_svd_S, d_svd_U, m, d_svd_VH, n, work, work_size, NULL, devInfo);
        cudaDeviceSynchronize();

        int devInfo_h = 0;
        gpuErrchk(cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
        // std::cout << "devInfo = " << devInfo_h << "\n";

        switch(stat){
            case CUSOLVER_STATUS_SUCCESS:           std::cout << "SVD computation success\n";                       break;
            case CUSOLVER_STATUS_NOT_INITIALIZED:   std::cout << "Library cuSolver not initialized correctly\n";    break;
            case CUSOLVER_STATUS_INVALID_VALUE:     std::cout << "Invalid parameters passed\n";                     break;
            case CUSOLVER_STATUS_INTERNAL_ERROR:    std::cout << "Internal operation failed\n";                     break;
        }

        // if (devInfo_h == 0 && stat == CUSOLVER_STATUS_SUCCESS) std::cout    << "SVD successful\n\n";

        // --- Moving the results from device to host
        // cudaMemcpy(h_S, d_S, N * sizeof(float), cudaMemcpyDeviceToHost);

        // for(int i = 0; i < N; i++) std::cout << "d_S["<<i<<"] = " << h_S[i] << std::endl;
        cusolverDnDestroy(solver_handle);
        // X3 update here
        // first calculate the prox_l1 // declare B as NULL here.
        // reusing prox_l1 here
        /* DO NOT DELETE - THIS IS NOT OLD CODE - JUST NEW CODE NOT READY TO RUN YET
        // **************************************************************
        // ANYTHING NOT READY TO COMPILE IS BELOW; EVERYTHING ABOVE SHOULD COMPILE AND RUN
        // KEEP MOVING THIS START OF COMMENT TO TEST MORE STUFF FOR COMPILATION
        X2_update<<<n_blocks, block_size>>>(d_svd_S, NULL, badmm_para->lambda * badmm_para->g3, size );

        // TODO-  this has to be updated 
        X3_update<<<n_blocks, block_size>>>(d_X_3, d_svd_U, d_svd_S,  d_svd_VH, size);

        // Concat all the X_i's to X for termination checks
        concat_X<<<n_blocks, block_size>>>(d_X, d_X_1, d_X_2, d_X_3, badmm_mt->N, size);

        cublasScopy(size, d_z_old, 1, d_z, 1);

        // % diagnostics, reporting, termination checks
        // matlab code continue here.
        // // matric vector multiplication, probably not required.
        // cublasSgemv( 'T',n,m, 1.0,d_X,n,col_ones,1, 0,d_rowSum,1);  // fortran, column-major
        // if (badmm_mt->a)
        //     rowNorm_a<<<n_blocks,block_size>>>(d_X, d_rowSum, d_a, size, n);
        // else
        //     rowNorm<<<n_blocks,block_size>>>(d_X, d_rowSum, size, n);

        // Z update
        // this line also uses average of the three Xis,
        // change this to cuda code
        // z = x + repmat(-avg(X_1, X_2, X_3) + A./N, 1, N);

        //zexp<<<n_blocks,block_size>>>( d_Z, d_X, d_Y, size);

        // U - update
        cublasScopy(size, d_U, 1, d_B, 1);
        // matrix vector multiplication
        cublasSgemv('N',n,m, 1.0,d_Z,n,row_ones,1, 0.0,d_colSum,1);
        if (badmm_mt->b)
            colNorm_b<<<n_blocks,block_size>>>(d_Z, d_colSum, d_b, size, n);
        else
            colNorm<<<n_blocks,block_size>>>(d_Z, d_colSum, size, n);

        // dual update
        dual<<<n_blocks,block_size>>>( d_Yerr, d_Y, d_X, d_Z, size);

        // check stopping conditions
        dev_ptr = thrust::device_pointer_cast(d_X);
        dev_ptr1 = thrust::device_pointer_cast(d_Xold);
        Xerr = thrust::transform_reduce(thrust::make_zip_iterator(thrust::make_tuple(dev_ptr, dev_ptr1)), thrust::make_zip_iterator(thrust::make_tuple(dev_ptr+size, dev_ptr1+size)), zdiffsq(), 0.0f, thrust::plus<float>());

        dev_ptr = thrust::device_pointer_cast(d_X);
        // for relative err condition
//        iternorm = thrust::inner_product(dev_ptr, dev_ptr+size, dev_ptr, 0.0f);
//        Xerr = sqrt(Xerr/iternorm);

        dev_ptr = thrust::device_pointer_cast(d_Yerr);
        Yerr = thrust::reduce(dev_ptr, dev_ptr+stride);
        dev_ptr = thrust::device_pointer_cast(d_Y);
        // for relative err condition
//        iternorm = thrust::inner_product(dev_ptr, dev_ptr+size, dev_ptr, 0.0f);
//        Yerr = sqrt(Yerr/iternorm);

        if ( Yerr < badmmpara->tol && Xerr < badmmpara->tol ) {
            break;
        }

        if( badmm_mt->print_step && !((iter+1)%badmm_mt->print_step) )
        {
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&milliseconds, start, stop);

            // calculate primal objective value
            dev_ptr = thrust::device_pointer_cast(d_Z);
            iter_obj = thrust::inner_product(d_Cptr, d_Cptr+size, dev_ptr, 0.0f);

            badmmpara->iter_time[count] = milliseconds;
            badmmpara->iter_err[count] = Xerr + Yerr;
            badmmpara->iter_obj[count] = iter_obj * (-badmmpara->rho);
            count++;

            printf("iter = %d, objval = %f, primal_err = %f, dual_err = %f, time = %f\n", iter, iter_obj * (-badmmpara->rho), Xerr, Yerr, milliseconds);
        }

        // *******************************************************
        // NON COMPILED BLOCK ENDS HERE
        */
    }
    // calculate primal objective value
    // dev_ptr = thrust::device_pointer_cast(d_Z);
    // iter_obj = thrust::inner_product(d_Cptr, d_Cptr+size, dev_ptr, 0.0f);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    // average X+Z
//    cublasSaxpy (size, 1, d_Z, 1, d_X, 1);
//    cublasSscal( size, 0.5, d_X, 1);

/*
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaMemcpyAsync(X, d_X, sizeof(float)*size, cudaMemcpyDeviceToHost,stream);

    badmmpara->iter_err[count] = Xerr + Yerr;
    badmmpara->iter_time[count] = milliseconds;
    badmmpara->iter_obj[count] = iter_obj * (-badmmpara->rho);
    printf("iter = %d, objval = %f, Xerr = %f, Yerr = %f, milliseconds:%f\n", iter, iter_obj * (-badmmpara->rho), Xerr, Yerr, milliseconds);


    if (badmm_mt->SAVEFILE)
    {
        char filename[40];
        FILE *f;
        sprintf(filename, "X_out.dat");
        f = fopen(filename, "wb");
        fwrite (X,sizeof(float),size,f);
        fclose(f);
    }

    cudaFree(d_X);

    delete[]X;
    */

    cudaDeviceReset();
}

int main(const int argc, const char **argv)
{

    BADMM_massTrans* badmm_mt = NULL;

    badmm_mt = (struct BADMM_massTrans *) malloc( sizeof(struct BADMM_massTrans) );

    badmm_mt->print_step = 0;           // default: not print
    badmm_mt->SAVEFILE = 1;             // default: save
    // we'll call it A
    badmm_mt->A = NULL;
    badmm_mt->a = NULL;
    badmm_mt->b = NULL;

    long size;
    int Asize[2];

    unsigned int dim;

    // dim = 1;
    // dim = 5;
    // dim = 10;
    // dim = 15;

    char* str;
    if ( argc > 1 ) dim = strtol(argv[1],&str,10);

    // dim = dim*1024;

    // read file
    char filename[40];
    FILE *f;

    // read A
    sprintf(filename, "%dC.dat",dim);

	printf("%s", filename);
	f = fopen ( filename , "rb" );

    if ( f == NULL ) {
        printf("Error! Can not find C file!");
        return 0;
    }

    fread(Asize,sizeof(int),2, f);
    badmm_mt->m = Asize[0];
    badmm_mt->n = Asize[1];
    badmm_mt->N = 3;
    size = badmm_mt->m*badmm_mt->n;
    badmm_mt->A = new float[size];
    fread (badmm_mt->A,sizeof(float),size,f);
    fclose(f);

    printf("Cost Matrix C: rows = %d, cols = %d, total size = %d\n", badmm_mt->m, badmm_mt->n, size);



    // DONT NEED FOR RPCA
    // DELETE ALL
    // read a
    sprintf(filename, "%da.dat",dim);
	f = fopen ( filename , "rb" );
    if ( f != NULL )
    {
        badmm_mt->a = new float[badmm_mt->m];
        fread (badmm_mt->a,sizeof(float),badmm_mt->m,f);
        fclose(f);
    }

    // read b
    sprintf(filename, "%db.dat",dim);
	f = fopen ( filename , "rb" );
    if ( f != NULL )
    {
        badmm_mt->b = new float[badmm_mt->n];
        fread (badmm_mt->b,sizeof(float),badmm_mt->n,f);
        fclose(f);
    }
    // UNTIL HERE


    int iter_size;

    ADMM_para* badmm_para = NULL;

    badmm_para = (struct ADMM_para *) malloc( sizeof(struct ADMM_para) );

    // default value
    badmm_para->lambda = 1;
    badmm_para->rho = 1.0 / badmm_para->lambda;
    badmm_para->MAX_ITER = 100;
    badmm_para->tol = 1e-4;
    badmm_para->abstol = 1e-4;
    badmm_para->reltol = 1e-2;

    if ( argc > 2 ) badmm_para->rho = strtod(argv[2],&str);
    if ( argc > 3 ) badmm_para->MAX_ITER = strtol(argv[3],&str,10);
    if ( argc > 4 ) badmm_para->tol = strtod(argv[4],&str);
    if ( argc > 5 ) badmm_mt->print_step = strtol(argv[5],&str,10);
    if ( argc > 6 ) badmm_mt->SAVEFILE = strtol(argv[6],&str,10);

    if ( badmm_para->rho == 0.0 ) badmm_para->rho = 0.001;
    if ( badmm_para->MAX_ITER == 0 ) badmm_para->MAX_ITER = 2000;
    if ( badmm_para->tol == 0.0 ) badmm_para->tol = 1e-4;

    iter_size = 1;
    if(badmm_mt->print_step)
    {
        iter_size = (int)badmm_para->MAX_ITER/badmm_mt->print_step + 1;
    }
    badmm_para->iter_obj = new float[iter_size];
    badmm_para->iter_time = new float[iter_size];
    badmm_para->iter_err = new float[iter_size];

    printf("Please be patient! Getting GPU information is slow .....\n");

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop,0);       // default device

    GPUInfo gpu_info;
    gpu_info.MAX_GRID_SIZE = prop.maxGridSize[0];
    gpu_info.MAX_BLOCK_SIZE = prop.maxThreadsPerBlock;

    // if out of GPU memory, return
    float mem = (size*5*4+(badmm_mt->m+badmm_mt->n)*3*4+gpu_info.MAX_GRID_SIZE*gpu_info.MAX_BLOCK_SIZE*2*4)/pow(2,30);
    float GPUmem = (long)prop.totalGlobalMem/pow(2,30);
    printf("gridDim = %d, blockDim = %d, memory required = %fGB, GPU memory = %fGB\n", gpu_info.MAX_GRID_SIZE, gpu_info.MAX_BLOCK_SIZE, mem, GPUmem );
    if ( GPUmem < mem )
    {
        printf("Not enough memory on GPU to solve the problem !\n");
        return 0;
    }

    printf("rho = %f, Max_Iteration = %d, tol = %f, print every %d steps, save result: %d\n", badmm_para->rho, badmm_para->MAX_ITER, badmm_para->tol, badmm_mt->print_step, badmm_mt->SAVEFILE);

    gpuBADMM_MT( badmm_mt, badmm_para, &gpu_info);

    delete[]badmm_para->iter_err;
    delete[]badmm_para->iter_obj;
    delete[]badmm_para->iter_time;
    free(badmm_para);
    if(badmm_mt->A)delete[]badmm_mt->A;
    if(badmm_mt->a)delete[]badmm_mt->a;
    if(badmm_mt->b)delete[]badmm_mt->b;
    free(badmm_mt);
}


void matInit(float* &X, unsigned int size, float value)
{
    for ( int i = 0 ; i < size ; i++ )
        X[i] = value;
}
