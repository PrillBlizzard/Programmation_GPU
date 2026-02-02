// bank_conflicts.cu
#include <stdio.h>
#include <cuda_runtime.h>
__global__ void transpose_with_bank_conflicts(float *in, float *out, int N)
{
    __shared__ float s_data[32][32];
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Lecture coalescée depuis Global Memory
    s_data[ty][tx] = in[y * N + x];
    __syncthreads();

    // Accès transposé : BANK CONFLICTS !
    // Threads dans un warp accèdent à s_data[tx][ty]
    // Cela crée des conflits de banques
    out[x * N + y] = s_data[tx][ty];
}

int main() {
    const int N = 1024;
    float *d_in, *d_out;
    int size = N * N * sizeof(float);
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);
    cudaMemset(d_in, 1, size);
    dim3 threads(32, 32);
    dim3 blocks((N + threads.x - 1) / threads.x, (N + threads.y - 1) / threads.y);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    transpose_with_bank_conflicts<<<blocks, threads>>>(d_in, d_out, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Transpose WITH bank conflicts: %.3f ms\n", milliseconds);
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}