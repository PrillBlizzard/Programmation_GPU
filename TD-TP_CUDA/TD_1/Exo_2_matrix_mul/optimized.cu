// matmul_optimized.cu
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
const int N = 1024;
const int TILE_SIZE = 32; // Taille des tiles en Shared Memory

inline void checkCuda(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess)
    {
        std::cerr << msg << " : " << cudaGetErrorString(err) << '\n';
        std::exit(EXIT_FAILURE);
    }
}

__global__ void matmul_optimized(float *A, float *B, float *C, int n)
{
    // Shared memory pour tiles
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    float sum = 0.0f;

    // Boucle sur les tiles
    for (int t = 0; t < (n + TILE_SIZE - 1) / TILE_SIZE; t++)
    {
        // Charger tile de A en Shared Memory
        if (row < n && t * TILE_SIZE + tx < n)
        {
            As[ty][tx] = A[row * n + t * TILE_SIZE + tx];
        }
        else
        {
            As[ty][tx] = 0.0f;
        }

        // Charger tile de B en Shared Memory
        if (col < n && t * TILE_SIZE + ty < n)
        {
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * n + col];
        }
        else
        {
            Bs[ty][tx] = 0.0f;
        }
        __syncthreads();
        
        // Calculer avec données en Shared Memory (très rapide !)
        for (int k = 0; k < TILE_SIZE; k++)
        {
            sum += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }
    if (row < n && col < n)
    {
        C[row * n + col] = sum;
    }
}
int main()
{

    const int N = 1024;
    const size_t sz = static_cast<size_t>(N) * N;
    const size_t nBytes = sz * sizeof(float);
    std::vector<float> hA(sz, 1.0f);
    std::vector<float> hB(sz, 2.0f);
    std::vector<float> hC(sz, 0.0f);
    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    checkCuda(cudaMalloc(&dA, nBytes), "cudaMalloc A");
    checkCuda(cudaMalloc(&dB, nBytes), "cudaMalloc B");
    checkCuda(cudaMalloc(&dC, nBytes), "cudaMalloc C");
    checkCuda(cudaMemcpy(dA, hA.data(), nBytes, cudaMemcpyHostToDevice), "cpy A");
    checkCuda(cudaMemcpy(dB, hB.data(), nBytes, cudaMemcpyHostToDevice), "cpy B");
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + block.x - 1) / block.x,
              (N + block.y - 1) / block.y);
    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start), "event start");
    checkCuda(cudaEventCreate(&stop), "event stop");

    // Naive
    checkCuda(cudaEventRecord(start), "record start naive");
    matmul_optimized<<<grid, block>>>(dA, dB, dC, N);
    checkCuda(cudaEventRecord(stop), "record stop naive");
    checkCuda(cudaEventSynchronize(stop), "sync naive");
    float msNaive = 0.0f;
    checkCuda(cudaEventElapsedTime(&msNaive, start, stop), "elapsed naive");
    double flops = 2.0 * N * static_cast<double>(N) * N;
    double gflopsNaive = (flops / 1e9) / (msNaive / 1e3);
    std::cout << "Naive MatMul: " << msNaive << " ms, " << gflopsNaive << " GFLOPs\n";

    // Optimized
    checkCuda(cudaEventRecord(start), "record start opt");
    matmul_optimized<<<grid, block>>>(dA, dB, dC, N);
    checkCuda(cudaEventRecord(stop), "record stop opt");
    checkCuda(cudaEventSynchronize(stop), "sync opt");
    float msOpt = 0.0f;

    checkCuda(cudaEventElapsedTime(&msOpt, start, stop), "elapsed opt");
    double gflopsOpt = (flops / 1e9) / (msOpt / 1e3);
    std::cout << "Optimized MatMul: " << msOpt << " ms, " << gflopsOpt << " GFLOPs\n";
    checkCuda(cudaMemcpy(hC.data(), dC, nBytes, cudaMemcpyDeviceToHost), "cpy C");

    // Vérification rudimentaire (pour A=1, B=2 → chaque C[i]=2*N)
    float ref = 2.0f * N;
    float maxErr = 0.0;
    for (int i = 0; i < 10; ++i)
    {
        maxErr = std::max(maxErr, std::abs(hC[i] - ref));
    }
    std::cout << "Max error: " << maxErr << '\n';
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}