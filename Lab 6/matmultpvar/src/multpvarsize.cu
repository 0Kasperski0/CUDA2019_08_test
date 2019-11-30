#include <stdio.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <sys/time.h>

#define tileSize 32

__global__ void MatrixMulVarKernel(float* M, float* N, float* P, int widthAHeightB, int heightA, int widthB) {
	int Row = blockIdx.y*blockDim.y+threadIdx.y;// Calculate the row index of the P element and M
	int Col = blockIdx.x*blockDim.x+threadIdx.x;// Calculate the column index of P and N
	if ((Row < heightA) && (Col < widthB)) {
		float Pvalue = 0;
		for (int k = 0; k < widthAHeightB; ++k) {
			Pvalue += M[Row*widthAHeightB+k]*N[k*widthB+Col];// each thread computes one element of the block sub-matrix
		}

		P[Row*widthB+Col] = Pvalue;
	}
}


__global__ void MatrixMulVarSharedMemoryKernel(float* M, float* N, float* P, int widthAHeightB, int heightA, int widthB) {

	int Mstart=widthAHeightB*tileSize*blockIdx.y;
	int Mend=Mstart+ widthAHeightB - 1;
	int mstep=tileSize;
	int Nstart=tileSize*blockIdx.x;
	int nstep=tileSize*widthB;
	float temp=0;

	__shared__ float Ms[tileSize][tileSize];
	__shared__ float Ns[tileSize][tileSize];

		//area where the tiles fits without "cutting"
		if(Mstart < (heightA/tileSize)*tileSize*widthAHeightB && Nstart%widthB < (widthB/tileSize)*tileSize ){
			for(int m=Mstart,n=Nstart;m<Mend;m+=mstep,n+=nstep){
				Ms[threadIdx.y][threadIdx.x]=M[m+widthAHeightB*threadIdx.y+threadIdx.x];
				Ns[threadIdx.y][threadIdx.x]=N[n+widthB*threadIdx.y+threadIdx.x];
				__syncthreads();


				for (int i = 0; i < tileSize; ++i) {
					temp += Ms[threadIdx.y][i] * Ns[i][threadIdx.x];
				}
				__syncthreads();

			}
		} else {//the rest of the matrix
			for(int m=Mstart,n=Nstart;m<=Mend;m+=mstep,n+=nstep){

				if(m%widthAHeightB + threadIdx.x < widthAHeightB && blockIdx.y*tileSize + threadIdx.y < heightA){
					Ms[threadIdx.y][threadIdx.x]=M[m+widthAHeightB*threadIdx.y+threadIdx.x];
				}
				else{
					Ms[threadIdx.y][threadIdx.x]=0.0;
				}

				if((n/widthB) + threadIdx.y < widthAHeightB && blockIdx.x*tileSize + threadIdx.x < widthB){
					Ns[threadIdx.y][threadIdx.x]=N[n+widthB*threadIdx.y+threadIdx.x];
				}
				else{
					Ns[threadIdx.y][threadIdx.x]=0.0;
				}
				__syncthreads();


				for (int i = 0; i < tileSize; ++i) {
					temp += Ms[threadIdx.y][i] * Ns[i][threadIdx.x];
				}
				__syncthreads();

			}
		}



	if(blockIdx.y*tileSize + threadIdx.y < heightA && blockIdx.x*tileSize + threadIdx.x < widthB){
		P[widthB * tileSize * blockIdx.y + tileSize * blockIdx.x + widthB * threadIdx.y + threadIdx.x] = temp;
	}
}



double cpuTimer(){
	struct timeval clock;
	gettimeofday(&clock, NULL);
	return ( (double)clock.tv_sec + (double)clock.tv_usec * 1.e-6 );
}



int main(void){
cudaError_t err = cudaSuccess;
double ti;
double duration;

int arow=3200;
int acol=1600;
int brow=1600;
int bcol=2400;

// Matrices initialization and error check________________________________________________________
float *M;
float *N;
float *P_C;
float *P_G;

err = cudaMallocManaged(&M,arow*acol*sizeof(float));
if (err != cudaSuccess){
	fprintf(stderr, "Failed to allocate device matrix M (error code %s)!\n", cudaGetErrorString(err));
	exit(EXIT_FAILURE);}
err = cudaMallocManaged(&N,brow*bcol*sizeof(float));
if (err != cudaSuccess){
	fprintf(stderr, "Failed to allocate device matrix N (error code %s)!\n", cudaGetErrorString(err));
	exit(EXIT_FAILURE);}
err = cudaMallocManaged(&P_G,arow*bcol*sizeof(float));
if (err != cudaSuccess){
	fprintf(stderr, "Failed to allocate device matrix P_G (error code %s)!\n", cudaGetErrorString(err));
	exit(EXIT_FAILURE);}
err = cudaMallocManaged(&P_C,arow*bcol*sizeof(float));
if (err != cudaSuccess){
	fprintf(stderr, "Failed to allocate device matrix P_C (error code %s)!\n", cudaGetErrorString(err));
	exit(EXIT_FAILURE);}

for(int i=0;i<acol*arow;++i){
	M[i]=i%100;
}


for(int i=0;i<bcol*brow;++i){
	N[i]=i%100;
}

// CPU Matrix Multiplication____________________________________________________________________
ti = cpuTimer();
for(int i=0;i<arow;i++){ //row
	for(int j=0;j<bcol;j++){	//col
		float temp=0;
		for(int k=0;k<acol;k++){
			temp+=M[i*acol+k]*N[bcol*k+j];
		}
		P_C[i*bcol+j]=temp;
	}
}
duration = cpuTimer() - ti;
printf("\nCPU - Matrix Multiplication: Done,\nDuration: %.6f s",duration);


// Base Line - Matrix Multiplication________________________________________________________________
int tpb= tileSize;
dim3 threadsPerBlock(tpb, tpb);
dim3 blocksPerGrid((bcol + tpb - 1) / tpb, (arow + tpb - 1) / tpb);

printf("\n\nBase Line: CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid.x*blocksPerGrid.y, threadsPerBlock.x*threadsPerBlock.y);
ti = cpuTimer();
MatrixMulVarKernel<<<blocksPerGrid, threadsPerBlock>>>(M, N, P_G, acol, arow, bcol);
cudaDeviceSynchronize();
duration = cpuTimer() - ti;

for (int i = 0; i < arow*bcol; ++i){
	if (fabs(P_C[i]-P_G[i]) > 1e-5){
		fprintf(stderr, "Result verification failed at element %d!\n", i);
		exit(EXIT_FAILURE);
    }
}

printf("Base Line - Matrix Multiplication: Result is Correct\nDuration: %.6f s", duration);





// Shared Memory - Matrix Multiplication________________________________________________________________
printf("\n\nShared Memory: CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid.x*blocksPerGrid.y, threadsPerBlock.x*threadsPerBlock.y);
ti = cpuTimer();
MatrixMulVarSharedMemoryKernel<<<blocksPerGrid, threadsPerBlock>>>(M, N, P_G, acol, arow, bcol);
cudaDeviceSynchronize();
duration = cpuTimer() - ti;

for (int i = 0; i < arow*bcol; ++i){
	if (fabs(P_C[i]-P_G[i]) > 1e-5){
		fprintf(stderr, "Result verification failed at element %d!\n", i);
		exit(EXIT_FAILURE);
    }
}

printf("Shared Memory - Matrix Multiplication: Result is Correct\nDuration: %.6f s", duration);




//cleaning________________________________________________________________
cudaFree(M);
cudaFree(N);
cudaFree(P_C);
cudaFree(P_G);

printf("\n\nEverything cleared.\nGood night.");

return 0;

}
