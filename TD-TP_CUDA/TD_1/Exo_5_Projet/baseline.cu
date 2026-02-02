#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <stdexcept>

const int VECTOR_SIZE = 1024 * 1024;
inline void checkCuda(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess)
    {
        std::cerr << msg << " : " << cudaGetErrorString(err) << '\n';
        std::exit(EXIT_FAILURE);
    }
}

__global__ void vector_sum_baseline(float *A, double *sum, int N)
{
    
}

int main(){

    try
    {
        const int N = VECTOR_SIZE;
        const size_t nBytes = N * sizeof(float);
        const size_t resBytes = sizeof(double);

        //host buffer (C++)
        std::vector<float> hA(N,1.0f);
        double sum_res = 0;


        //Device buffer (CUDA)
        float *dA = nullptr; 
        double *dsum_res = nullptr;
        checkCuda(cudaMalloc(&dA, nBytes), "cudaMalloc dA");
        checkCuda(cudaMalloc(&dsum_res, resBytes), "cudaMalloc dsum_res");

        checkCuda(cudaMemcpy(dA, hA.data(), nBytes, cudaMemcpyHostToDevice), "cpy A");
        checkCuda(cudaMemcpy(dsum_res, &sum_res,resBytes, cudaMemcpyHostToDevice), "cpy res_sum");

        dim3 block(256);
        dim3 grid((N+block.x -1)/block.x );

        vector_sum_baseline<<<grid , block>>>(dA,dsum_res,N);

        

        return 0;
    }

    catch (const std::exception &e)
    {
        std::cerr << "Exception: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}