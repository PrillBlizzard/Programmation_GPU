// vector_add_optimized.cu
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

__global__ void vector_add_optimized(float *A, float *B, float *C, int N)
{
    // Shared memory pour les vecteurs
    __shared__ float As[VECTOR_SIZE];
    __shared__ float Bs[VECTOR_SIZE];
    __shared__ float Cs[VECTOR_SIZE];

    // Accès coalescé : threads consécutifs lisent adresses consécutives
    int idx = blockIdx.x * blockDim.x + threadIdx.x;



    // Traiter plusieurs éléments par thread (meilleur ratio calcul/accès)
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < N; i += stride)
    {
        C[i] = A[i] + B[i];
    }
}
int main()
{
    try
    {
        const int N = VECTOR_SIZE;
        const size_t nBytes = N * sizeof(float);

        // Host buffers (C++)
        std::vector<float> hA(N, 1.0f);
        std::vector<float> hB(N, 2.0f);
        std::vector<float> hC(N, 0.0f);

        // Device buffers
        float *dA = nullptr, *dB = nullptr, *dC = nullptr;
        checkCuda(cudaMalloc(&dA, nBytes), "cudaMalloc dA");
        checkCuda(cudaMalloc(&dB, nBytes), "cudaMalloc dB");
        checkCuda(cudaMalloc(&dC, nBytes), "cudaMalloc dC");
        checkCuda(cudaMemcpy(dA, hA.data(), nBytes, cudaMemcpyHostToDevice), "cpy A");
        checkCuda(cudaMemcpy(dB, hB.data(), nBytes, cudaMemcpyHostToDevice), "cpy B");

        dim3 block(256);
        dim3 grid((N + block.x - 1) / block.x);
        cudaEvent_t start, stop;
        checkCuda(cudaEventCreate(&start), "event create start");
        checkCuda(cudaEventCreate(&stop), "event create stop");

        // Optimized (grid‑stride loop)
        checkCuda(cudaMemset(dC, 0, nBytes), "memset C");
        checkCuda(cudaEventRecord(start), "record start opt");
        vector_add_optimized<<<grid, block>>>(dA, dB, dC, N);

        checkCuda(cudaEventRecord(stop), "record stop opt");
        checkCuda(cudaEventSynchronize(stop), "sync stop opt");

        float msOpt = 0.0f;
        checkCuda(cudaEventElapsedTime(&msOpt, start, stop), "elapsed opt");
        std::cout << "Optimized Vector Add: " << msOpt << " ms\n";
        checkCuda(cudaMemcpy(hC.data(), dC, nBytes, cudaMemcpyDeviceToHost), "cpy C opt");

        // Vérification simple
        bool ok = true;
        for (int i = 0; i < 10; ++i)
        {
            if (hC[i] != 3.0f)
            {
                ok = false;
                break;
            }
        }

        std::cout << "Check: " << (ok ? "OK" : "FAILED") << '\n';
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);
        return ok ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}