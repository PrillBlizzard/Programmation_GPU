#include <time.h>
#include <stdio.h>

void generate_array(unsigned char *array,int size){


    printf("array size = %d\n", size);

    for(int i =0; i< size; i++){

        array[i]= rand()%256;

    }

}

__global__ void vectorAdd(int* _data_d, int* _hist_d ,int N)
{
    int i, stride;

    for(i=0; i<N; i+stride){
        atomicAdd(&_hist_d[_data_d[i]],1);

    }

};

int main(int argc, char *argv[]){

    const int array_size = 1e5;
    unsigned char array[array_size] = {0};;
    generate_array(array, array_size);

    int hist[256]= {0};
    
    unsigned char* array_d =0;
    cudaMalloc((void**)&array_d, array_size*sizeof(unsigned char));

    int* hist_d =0;
    cudaMalloc((void**)&hist_d, 256*sizeof(int));

    cudaMemcpy(array_d, array, array_size*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemset(hist_d,0,256*sizeof(int));

    for(int i =0; i< array_size; i++){
        unsigned char array_value = array[i];

        hist[array_value]+=1;

    }

    int total_number = 0;

    for(int i=0; i<256; i++){

        total_number+=hist[i];
        printf("%d / ", hist[i]);
    }
    printf("\n sum of hist values = %d\n", total_number);

    if(total_number == array_size)
        printf("success\n");
    else
        printf("you failed bitch\n");

}