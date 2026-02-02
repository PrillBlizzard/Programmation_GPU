// bank_conflicts_fixed.cu
#include <stdio.h>
#include <cuda_runtime.h>
__global__ void transpose_without_bank_conflicts(float *in, float *out, int N) {

    // Padding d'1 élément : 33 au lieu de 32
    __shared__ float s_data[32][33];
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    s_data[ty][tx] = in[y * N + x];
    __syncthreads();

    // Pas de bank conflicts grâce au padding
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
    transpose_without_bank_conflicts<<<blocks, threads>>>(d_in, d_out, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Transpose WITHOUT bank conflicts (padded): %.3f ms\n",
    milliseconds);
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}