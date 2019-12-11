#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#define BLOCK_SIZE 32

__global__ void total(float *input, float *output, int len){
	__shared__ float partialSum[2*BLOCK_SIZE];
	unsigned int t=threadIdx.x,start=2*blockIdx.x*BLOCK_SIZE;

	if(start+t<len)	partialSum[t] = input[start+t];

	else partialSum[t]=0;
	__syncthreads();
	if(start+BLOCK_SIZE+t<len)partialSum[BLOCK_SIZE+t]=input[start+BLOCK_SIZE+t];
	else partialSum[BLOCK_SIZE+t]=0;
	__syncthreads();
	for(unsigned int stride=BLOCK_SIZE;stride>=1; stride>>=1){
		__syncthreads();
		if (t<stride) partialSum[t]+=partialSum[t+stride];
		__syncthreads();
	}
	if(t==0) output[blockIdx.x]=partialSum[0];
}

double cpuTimer(){
	struct timeval clock;
	gettimeofday(&clock, NULL);
	return ( (double)clock.tv_sec + (double)clock.tv_usec * 1.e-6 );
}



int
main(int argc, char **argv)
{
	cudaError_t err = cudaSuccess;
	double ti;
	double duration;
	unsigned long int a=pow(2,30); //table length
	double sumcpu=0;

	// Table initialization and error check________________________________________________________
	float *M;
	float *result;
	err = cudaMallocManaged(&M,a*sizeof(float));
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device matrix M (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);}
	err = cudaMallocManaged(&result,a*sizeof(float));
		if (err != cudaSuccess){
			fprintf(stderr, "Failed to allocate device matrix M (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);}
		// Table data _____________________________________________________________________________________
	for(int i=0;i<a;++i){
		M[i]=1;
	}



	// CPU CODE_____________________________________________________________________________________

	ti = cpuTimer();
	for(unsigned long int i=0;i<a;++i)sumcpu+=M[i];
	duration = cpuTimer() - ti;
	printf("Base Line - Sum reduction - CPU \nDuration: %.6f s\n", duration);
	printf("CPU result: %f",sumcpu);

	// GPU KERNEL _____________________________________________________________________________________
		int tpb= BLOCK_SIZE;
		dim3 threadsPerBlock(tpb);
		int step=2*BLOCK_SIZE;
		int u=ceil(float(a)/float(step));
		dim3 blocksPerGrid((a + (tpb - 1)) / tpb);

		printf("\n\nBase Line: CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid.x*blocksPerGrid.y, threadsPerBlock.x*threadsPerBlock.y);
		ti = cpuTimer();

		total<<<blocksPerGrid, threadsPerBlock>>>(M,result,a);
		cudaDeviceSynchronize();

		duration = cpuTimer() - ti;

		float totres=0;
		for(int i=0;i<u;i++){
			totres+=result[i];
		}


		printf("Base Line - Sum reduction\nDuration: %.6f s\n", duration);
		printf("GPU result: %f\n",totres);
	// Result check _____________________________________________________________________________________

	
	if(totres==sumcpu){
		printf("\nVerification successful!\n");
	}
	else printf("\nERROR!\n");

	cudaFree(M);
	cudaFree(result);
	printf("Task ended");
	return 0;
}
