#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    int deviceNumber;
    cudaDeviceProp prop_bob;
    int deviceCount;

    cudaGetDevice(&deviceNumber);
    cudaGetDeviceProperties(&prop_bob,deviceNumber);
    cudaGetDeviceCount(&deviceCount);


    printf("Nombre de GPU CUDA détectés : %d\n", deviceCount);
    printf("Numéro GPU : %d \n", deviceNumber);
    printf("Nom GPU : %s \n",prop_bob.name);
    printf("Capacité Mémoire : %u \n", prop_bob.totalGlobalMem);
    printf(" \n");
    printf("Version CUDA runtime : %d\n", CUDART_VERSION);

    printf("max grid size : %u \n",prop_bob.maxGridSize);

    return 0;
}
