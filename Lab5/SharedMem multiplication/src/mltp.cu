#include <stdio.h>
#include <cuda_runtime.h>
#include <assert.h>

//Shared f... memory. Attempt #ilostcount3hoursago
#define BLOCK_SIZE 16 //size of our tile !!! cna't be larger than matrices we calculate on!!!

__global__ void MatrixMulKernel(float* M, float* N, float* P, int Width) {
	//We need to iterate with tiles - starting point and end needed for tiles
	int Mstart=Width*BLOCK_SIZE*blockIdx.y;//rows of matrix M
	int Mend=Mstart+Width-1;
	int mstep=BLOCK_SIZE;
	int Nstart=BLOCK_SIZE*blockIdx.x;//cols of matrix N
	int nstep=BLOCK_SIZE*Width;
	float temp=0;

	//loop through tiles


	for(int m=Mstart,n=Nstart;m<Mend;m+=mstep,n+=nstep){
		__shared__ float Ms[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ float Ns[BLOCK_SIZE][BLOCK_SIZE];
		Ms[threadIdx.y][threadIdx.x]=M[m+Width*threadIdx.y+threadIdx.x];
		Ns[threadIdx.y][threadIdx.x]=N[n+Width*threadIdx.y+threadIdx.x];
		__syncthreads();


		for (int i = 0; i < BLOCK_SIZE; ++i) {
			temp += Ms[threadIdx.y][i] * Ns[i][threadIdx.x];
		}

		__syncthreads();

	}

	P[Width * BLOCK_SIZE * blockIdx.y + BLOCK_SIZE * blockIdx.x + Width * threadIdx.y + threadIdx.x] = temp;

}

int main(void){
cudaError_t err = cudaSuccess;

int deviceId;
cudaGetDevice(&deviceId);
printf("Sharedmem\n");

int matdim=14400;
if(BLOCK_SIZE>matdim){
    fprintf(stderr, "Tile size is bigger than initial matrices, operation cannot be performed (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
}
// MAtrices initialisation and error check %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
float *M;
float *N;
float *P_C;
float *P_G;

err = cudaMallocManaged(&M,matdim*matdim*sizeof(float));
if (err != cudaSuccess){
        fprintf(stderr, "Failed to allocate device matrix M (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);}
err = cudaMallocManaged(&N,matdim*matdim*sizeof(float));
if (err != cudaSuccess){
        fprintf(stderr, "Failed to allocate device matrix N (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);}
err = cudaMallocManaged(&P_G,matdim*matdim*sizeof(float));
    if (err != cudaSuccess){
            fprintf(stderr, "Failed to allocate device matrix P_G (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);}
err = cudaMallocManaged(&P_C,matdim*matdim*sizeof(float));
    if (err != cudaSuccess){
                fprintf(stderr, "Failed to allocate device matrix P_C (error code %s)!\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);}
for(int i=0;i<matdim*matdim;++i){
	M[i]=i;
	N[i]=i;
}
//  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//CPU code
//for(int i=0;i<matdim;i++){ //row
//	for(int j=0;j<matdim;j++){	//col
//		float temp=0;
//		for(int k=0;k<matdim;k++){
//			temp+=M[i*matdim+k]*N[matdim*k+j];
//		}
//		P_C[i*matdim+j]=temp;
//	}
//}
//Kernel call and grid layout	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



cudaMemPrefetchAsync(N, matdim*matdim*sizeof(float), deviceId);
cudaMemPrefetchAsync(M, matdim*matdim*sizeof(float), deviceId);
cudaMemPrefetchAsync(P_G, matdim*matdim*sizeof(float), deviceId);

	dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE); //1024 - full block
    dim3 blocksPerGrid((matdim + BLOCK_SIZE - 1) / BLOCK_SIZE, (matdim + BLOCK_SIZE - 1) / BLOCK_SIZE);
    //matrix has to be fully covered by threads and now one block is equal to one tile!!!!

    cudaDeviceSynchronize();

    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid.x*blocksPerGrid.y, threadsPerBlock.x*threadsPerBlock.y);
   MatrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(M, N, P_G, matdim);
   cudaDeviceSynchronize();

   cudaMemPrefetchAsync(P_G, matdim*matdim*sizeof(float), cudaCpuDeviceId);
    /*
    for(int i=0;i<matdim;++i){
    		for(int j=0;j<matdim;++j){
    			printf("%.2f \t",P_C[j+matdim*i]);
    		}
    		printf("\n");
    }
    printf("GPU macierz:\n");
    for(int i=0;i<matdim;++i){
     		for(int j=0;j<matdim;++j){
     			printf("%.2f \t",P_G[j+matdim*i]);
     		}
     		printf("\n");
     }
*/

    //Verification 		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//        for (int i = 0; i < matdim*matdim; ++i)
//           {
//               if (fabs(P_C[i]-P_G[i]) > 1e-5)
//               {
//                   fprintf(stderr, "Result verification failed at element %d!\n", i);
//                   //exit(EXIT_FAILURE);
//               }
//           }


//cleaning			%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    cudaFree(M);
    cudaFree(N);
    cudaFree(P_C);
    cudaFree(P_G);
    printf("Done, Verification successful;\n");

	return 0;
	//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



}
