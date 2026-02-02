// vector_add_naive.cu
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <stdexcept>

constexpr int VECTOR_SIZE = 1024 * 1024;

inline void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << msg << " : " << cudaGetErrorString(err) << '\n';
        std::exit(EXIT_FAILURE);
    }
}
__global__ void vector_add_naive(float *A, float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // Chaque thread accède directement à Global Memory
        C[idx] = A[idx] + B[idx];
    }
}
int main() {
    try {
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

        // Naive
        checkCuda(cudaEventRecord(start), "record start naive");
        vector_add_naive<<<grid, block>>>(dA, dB, dC, N);
        checkCuda(cudaEventRecord(stop), "record stop naive");
        checkCuda(cudaEventSynchronize(stop), "sync stop naive");
        float msNaive = 0.0f;
        checkCuda(cudaEventElapsedTime(&msNaive, start, stop), "elapsed naive");

        std::cout << "Naive Vector Add: " << msNaive << " ms\n";
        checkCuda(cudaMemcpy(hC.data(), dC, nBytes, cudaMemcpyDeviceToHost), "cpy C naive");
        
        // Vérification simple
        bool ok = true;
        for (int i = 0; i < 10; ++i) {
            if (hC[i] != 3.0f) { ok = false; break; }
        }
        std::cout << "Check: " << (ok ? "OK" : "FAILED") << '\n';
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);
        return ok ? EXIT_SUCCESS : EXIT_FAILURE;
    } 

    catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}