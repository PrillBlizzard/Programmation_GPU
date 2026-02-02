
#include <stdio.h>
#include <stdlib.h>
#include <time.h>



int main(int argc,int *argv[]){

    size_t N = 10e7;

    struct timespec chron_start;
    struct timespec chron_stop;

    double *A = malloc(N*sizeof(double));
    double *B = malloc(N*sizeof(double));
    double *C = malloc(N*sizeof(double));

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

    printf(" C[0] : %lf \n C[1] : %lf \n C[N-1] : %lf \n", C[0], C[1], C[N-1]);
    printf("With addition time of : %lu nanoseconds \n", addition_time_ns);
    printf("With addition time of : %d seconds \n", addition_time_s);

}