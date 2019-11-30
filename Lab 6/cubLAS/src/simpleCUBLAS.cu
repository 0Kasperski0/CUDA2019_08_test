# include <stdio.h>
# include <stdlib.h>
# include <cstdlib>
# include <cuda_runtime.h>
# include "cublas_v2.h"
#include <time.h>
#include <sys/time.h>

void matprint(float *M,int row, int col){


	printf ("\nMatrix %d by %d \n",row,col);
	for (int i=0;i<row;i ++){
		for (int j=0;j<col;j ++){
			printf ("%0.f ",M[i*col+j]);
		}
		printf ("\n");
	}
	printf ("\n");


	for(int i=0;i<col*row;i++){
		printf ("%0.f ",M[i]);
	}
	printf ("\n");
	printf ("\n");
}

void matfill(float *M,int row, int col){
	int ind =1;
	for(int j=0;j<col;j ++){
		for(int i=0;i<row;i ++){
			M[j*row+i]=ind++ ;//rand()%30;
		}
	}
}

double cpuTimer(){
	struct timeval clock;
	gettimeofday(&clock, NULL);
	return ( (double)clock.tv_sec + (double)clock.tv_usec * 1.e-6 );
}

int main ( void ){
cublasHandle_t handle ; // CUBLAS context
cudaError_t err = cudaSuccess;
srand (time(NULL));
double ti;
double duration;

int m=3200; // a - mxk
int n=1600; // b - kxn
int k=2400; // P_G and P_C - mxn


//creating matrices and allocate data		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
float * a; // mxk
float * b; // kxn
float *P_G; // mxn

err = cudaMallocManaged(&a,m*k*sizeof(float));
if (err != cudaSuccess){
        fprintf(stderr, "Failed to allocate device matrix M (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);}
err = cudaMallocManaged(&b,k*n*sizeof(float));
if (err != cudaSuccess){
        fprintf(stderr, "Failed to allocate device matrix N (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);}
err = cudaMallocManaged(&P_G,m*n*sizeof(float));
    if (err != cudaSuccess){
            fprintf(stderr, "Failed to allocate device matrix P_G (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);}

// filling with data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

matfill(a,m,k);
matfill(b,k,n);



cublasCreate (& handle ); // initialize CUBLAS context
float al =1;
float bet =0;
//matrix multiplication : c = al*a*b + bet *c
ti = cpuTimer();
cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,&al,a,m,b,k,&bet,P_G,m);
cudaDeviceSynchronize ();
duration = cpuTimer() - ti;


printf("\nCublas - Matrix Multiplication: Done,\nDuration: %.6f s",duration);

cudaFree (a); // free memory
cudaFree (b); // free memory
cudaFree (P_G); // free memory

cublasDestroy ( handle ); // destroy CUBLAS context
return 0;
}
