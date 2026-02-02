#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <memory.h>

cudaEvent_t start_ker, stop_ker;
cudaStream_t stream_ker;
float milis;


__global__ void vectorAdd(const float* A, const float* B, float* C, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    } 
};


int main(int argc,char *argv[]){

    size_t N = 1;
    
    struct timespec chron_start;
    struct timespec chron_stop;

   

    float *A = (float*)malloc(N*sizeof(float));
    float *B = (float*)malloc(N*sizeof(float));
    float *C = (float*)malloc(N*sizeof(float));
    
    for(size_t i=0; i<N; i++){
            A[i] = (double)i;
            B[i] = 2*(double)i;
        } 

    int start_time = clock_gettime(0, &chron_start);
    for(size_t i=0; i<N; i++){

            C[i] = A[i] + B[i];

        } 
    int end_time = clock_gettime(0, &chron_stop);


    unsigned long addition_time_ns = chron_stop.tv_nsec - chron_start.tv_nsec;
    time_t addition_time_s = chron_stop.tv_sec - chron_start.tv_sec;

    printf("tume using the CPU : \n");
    printf("With addition time of : %lu nanoseconds \n", addition_time_ns);
    printf("With addition time of : %d seconds \n\n", addition_time_s);

// --------------------------------------------------------------------------------- //

    start_time = clock_gettime(0, &chron_start);

    
    cudaEventCreate(&start_ker);
    cudaEventCreate(&stop_ker);
    

        
    // dynamic allocation of an array A with the size N*sizeof(float)
    float* A_d =0;
    cudaMalloc((void**)&A_d, N*sizeof(float));
    float* B_d =0;
    cudaMalloc((void**)&B_d, N*sizeof(float));
    float* C_d =0;
    cudaMalloc((void**)&C_d, N*sizeof(float));
    
    // copy the array to the GPU 
    cudaMemcpy(A_d, A, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(C_d, C, N*sizeof(float), cudaMemcpyHostToDevice);

    
    cudaEventRecord(start_ker,stream_ker);
    // <<grid, bloc>
    vectorAdd<<<(N+255)/256,256>>>(A_d, B_d, C_d, N);
    cudaEventRecord(stop_ker,stream_ker);

    cudaDeviceSynchronize();
    cudaMemcpy(C_d, C, N*sizeof(float), cudaMemcpyDeviceToHost);
    
    end_time = clock_gettime(0, &chron_stop);


    addition_time_ns = chron_stop.tv_nsec - chron_start.tv_nsec;
    addition_time_s = chron_stop.tv_sec - chron_start.tv_sec;

    printf("tume using the GPU : \n");
    printf("With addition time of : %lu nanoseconds \n", addition_time_ns);
    printf("With addition time of : %d seconds \n\n", addition_time_s);

    cudaEventElapsedTime(&milis,start_ker,stop_ker);

    printf("time of kernel: ");
    printf("%f milis \n\n", milis);


}



