#include <stdio.h>
#include <cuda_runtime.h>

//basic implementation of matrix multiplication in CUDA with CPU verification.
//nie ruszac bo dziala juz

__global__ void MatrixMulKernel(float* M, float* N, float* P, int Width) {
	int Row = blockIdx.y*blockDim.y+threadIdx.y;// Calculate the row index of the P element and M
	int Col = blockIdx.x*blockDim.x+threadIdx.x;// Calculate the column index of P and N
	if ((Row < Width) && (Col < Width)) {
		float Pvalue = 0;
		for (int k = 0; k < Width; ++k) {
			Pvalue += M[Row*Width+k]*N[k*Width+Col];// each thread computes one element of the block sub-matrix
		}

		P[Row*Width+Col] = Pvalue;
	}
}


int main(void){
cudaError_t err = cudaSuccess;
int deviceId;
cudaGetDevice(&deviceId);
printf("Simple\n");

int matdim=11200;

// MAtrices initialisation and error check %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
float *M;
float *N;
float *P_C;
float *P_G;

err = cudaMallocManaged(&M,matdim*matdim*sizeof(float));
if (err != cudaSuccess){
	fprintf(stderr, "Failed to allocate device matrix M (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
}

err = cudaMallocManaged(&N,matdim*matdim*sizeof(float));
if (err != cudaSuccess){
    fprintf(stderr, "Failed to allocate device matrix N (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
}

err = cudaMallocManaged(&P_G,matdim*matdim*sizeof(float));
if (err != cudaSuccess){
	fprintf(stderr, "Failed to allocate device matrix P_G (error code %s)!\n", cudaGetErrorString(err));
	exit(EXIT_FAILURE);
}

err = cudaMallocManaged(&P_C,matdim*matdim*sizeof(float));
if (err != cudaSuccess){
	fprintf(stderr, "Failed to allocate device matrix P_C (error code %s)!\n", cudaGetErrorString(err));
	exit(EXIT_FAILURE);
}

for(int i=0;i<matdim*matdim;++i){
	M[i]=i;
	N[i]=i;
}

cudaMemPrefetchAsync(N, matdim*matdim*sizeof(float), deviceId);
cudaMemPrefetchAsync(M, matdim*matdim*sizeof(float), deviceId);
cudaMemPrefetchAsync(P_G, matdim*matdim*sizeof(float), deviceId);

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

int tpb=16;//32 max
dim3 threadsPerBlock(tpb, tpb); //1024 - full block
dim3 blocksPerGrid((matdim + tpb - 1) / tpb, (matdim + tpb - 1) / tpb); //matrix has to be fully covered by threads

printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid.x*blocksPerGrid.y, threadsPerBlock.x*threadsPerBlock.y);
MatrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(M, N, P_G, matdim);
cudaDeviceSynchronize();

cudaMemPrefetchAsync(P_G, matdim*matdim*sizeof(float), cudaCpuDeviceId);

//Verification 		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//for (int i = 0; i < matdim*matdim; ++i){
//	if (fabs(P_C[i]-P_G[i]) > 1e-5){
//		fprintf(stderr, "Result verification failed at element %d!\n", i);
//		exit(EXIT_FAILURE);
//    }
//}
//cleaning			%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

cudaFree(M);
cudaFree(N);
cudaFree(P_C);
cudaFree(P_G);
printf("Done\n");

return 0;
	//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	/*
	for(int i=0;i<matdim;++i){
		for(int j=0;j<matdim;++j){
			printf("%.2f \t",P[j+matdim*i]);
		}
		printf("\n");
	}
	*/
}
