#include <stdio.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <sys/time.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#define tileSize 32

//function for data initialization
void initialization( double *M,  double *N, int arow, int acol, int brow, int bcol);
//(for Debugging) prints out the input data
void printInput( double *M,  double *N, int arow, int acol, int brow,  int bcol);
//(for Debugging) prints out the output data
void printOutput( double *P_C,  double *P_G, int arow, int bcol);

//GPU kernels
__global__ void
vectorAddition(const double *A, const double *B, double *C, int numElements);
__global__ void
vectorSubtraction(const double *A, const double *B, double *C, int numElements);
__global__ void
vectorScaling(const double *A, double s, double *C, int numElements);
__global__ void
vectorSum(double *input, double *output, int len);
__global__ void
vectorDotProd (double *a,  double *b,  double *c,int N);
__global__ void
matrixMultiplication(double* M, double* N, double* P, int widthAHeightB, int heightA, int widthB);
__global__ void
matrixTransposeSqr(double *P, double* M, int width, int height);


//CPU functions needed for checking
void CPUVectorAddition(const double *A, const double *B, double *C, int numElements);
void CPUVectorSubtraction(const double *A, const double *B, double *C, int numElements);
void CPUVectorScaling(const double *A, double s, double *C, int numElements);
void CPUVectorSum(double *M, double *P_C, int len);
void CPUVectorDotProd( const  double *A, const  double *B,   double *C, int N);
void CPUMatrixMultiplication(double *M, double *N, double *P_C, double *P_G, int arow, int acol, int brow, int bcol);
void CPUMatrixTransposeSqr(double *A, double *C, const int arow, const int acol);

//checking and performance
void CPUGPUCheck(double *P_C, double *P_G, int arow, int acol, int brow, int bcol);
double cpuTimer();



int main(void){
cudaError_t err = cudaSuccess;
double ti;
double duration,durationCPU;

int deviceId;
int numberOfSMs;

cudaGetDevice(&deviceId);
cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

//SIZES OF INPUT DATA _____________________________________________________________________________
//if you perform operations on matrices it is pretty self explanatory
//if you perform operations on vectors, the length of a vector is a product of row and column.
int arow=2<<8;
int acol=2<<8;
int brow=2<<8;
int bcol=2<<8;


// Matrices initialization and error check_________________________________________________________
double *M, *N, *P_C, *P_G;

size_t sizeM = arow*acol*sizeof(double);
size_t sizeN = brow*bcol*sizeof(double);
size_t sizeP_G = arow*bcol*sizeof(double);
size_t sizeP_C = arow*bcol*sizeof(double);


err = cudaMallocManaged(&M,sizeM);
if (err != cudaSuccess){
	fprintf(stderr, "Failed to allocate device matrix M (error code %s)!\n", cudaGetErrorString(err));
	exit(EXIT_FAILURE);}
err = cudaMallocManaged(&N,sizeN);
if (err != cudaSuccess){
	fprintf(stderr, "Failed to allocate device matrix N (error code %s)!\n", cudaGetErrorString(err));
	exit(EXIT_FAILURE);}
err = cudaMallocManaged(&P_G,sizeP_G);
if (err != cudaSuccess){
	fprintf(stderr, "Failed to allocate device matrix P_G (error code %s)!\n", cudaGetErrorString(err));
	exit(EXIT_FAILURE);}
err = cudaMallocManaged(&P_C,sizeP_C);
if (err != cudaSuccess){
	fprintf(stderr, "Failed to allocate device matrix P_C (error code %s)!\n", cudaGetErrorString(err));
	exit(EXIT_FAILURE);}



initialization(M, N, arow, acol, brow, bcol);
//printInput(M, N, arow, acol, brow, bcol);


// CUDA Computation________________________________________________________________________________
int tpb= tileSize;
dim3 m_threadsPerBlock(tpb, tpb);
dim3 m_blocksPerGrid((bcol + tpb - 1) / tpb, (arow + tpb - 1) / tpb);

dim3 v_threadsPerBlock(tpb);
dim3 v_blocksPerGrid((arow*bcol + tpb -1)/tpb);


// grid info for matrices
//printf("\nCUDA kernel launch with %d blocks of %d threads\n", m_blocksPerGrid.x*m_blocksPerGrid.y, m_threadsPerBlock.x*m_threadsPerBlock.y);

//grid info for vectors
//printf("\nCUDA kernel launch with %d blocks of %d threads\n", v_blocksPerGrid.x, v_threadsPerBlock.x);


cudaMemPrefetchAsync(M, sizeM, deviceId);
cudaMemPrefetchAsync(N, sizeN, deviceId);
cudaMemPrefetchAsync(P_G, sizeP_G, deviceId);


ti = cpuTimer();

//choose the kernel you want to use _______________________________________________________________
//vectorAddition<<<v_blocksPerGrid, v_threadsPerBlock>>>(M, N, P_G, acol*arow);
//vectorSubtraction<<<v_blocksPerGrid, v_threadsPerBlock>>>(M, N, P_G, acol*arow);
//vectorScaling<<<v_blocksPerGrid, v_threadsPerBlock>>>(M, 2.45, P_G, acol*arow);
//vectorSum<<<v_blocksPerGrid, v_threadsPerBlock>>>(M, P_G, arow*acol);
//vectorDotProd<<<v_blocksPerGrid, v_threadsPerBlock>>>(M, N, P_G, acol*arow);
//matrixTransposeSqr<<<m_blocksPerGrid, m_threadsPerBlock>>>(P_G,M,acol, arow);
matrixMultiplication<<<m_blocksPerGrid, m_threadsPerBlock>>>(M, N, P_G, acol, arow, bcol);

cudaDeviceSynchronize();
duration = cpuTimer() - ti;

cudaMemPrefetchAsync(M, sizeM, cudaCpuDeviceId);
cudaMemPrefetchAsync(N, sizeN, cudaCpuDeviceId);

ti = cpuTimer();

//if you want to check the result, use the corresponding CPU function to the kernel
//CPUVectorAddition(M, N, P_C, acol*arow);
//CPUVectorSubtraction(M, N, P_C, acol*arow);
//CPUVectorScaling(M, 2.45, P_C, acol*arow);
//CPUVectorSum(M, P_C, acol*arow);
//CPUVectorDotProd(M, N, P_C, acol*arow);
//CPUMatrixTransposeSqr(M, P_C, acol,arow);
CPUMatrixMultiplication(M, N, P_C, P_G, arow, acol, brow, bcol);

durationCPU=cpuTimer() - ti;

cudaMemPrefetchAsync(P_G, sizeP_G, cudaCpuDeviceId);
//printOutput(P_C, P_G, arow, bcol);
CPUGPUCheck(P_C, P_G, arow, acol, brow, bcol);


printf("CUDA Computation Duration: %.6lf s\n", duration);
printf("CPU Computation Duration: %.6lf s\n", durationCPU);

//cleaning________________________________________________________________
cudaFree(M);
cudaFree(N);
cudaFree(P_C);
cudaFree(P_G);
printf("Everything is cleared");
return 0;

}


__global__ void
vectorAddition(const double *A, const double *B, double *C, int numElements)
{
    int gridIndex = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = gridIndex; i<numElements; i+=stride)
    {
        C[i] = A[i] + B[i];
    }
}

__global__ void
vectorSubtraction(const double *A, const double *B, double *C, int numElements)
{
    int gridIndex = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = gridIndex; i<numElements; i+=stride)
    {
        C[i] = A[i] - B[i];
    }
}
__global__ void
vectorScaling(const double *A, double s, double *C, int numElements)
{
    int gridIndex = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = gridIndex; i<numElements; i+=stride)
    {
        C[i] = A[i]*s;
    }
}
__global__ void
vectorSum(double *input, double *output, int len){
	__shared__ double partialSum[2*tileSize];
	unsigned int t=threadIdx.x,start=2*blockIdx.x*tileSize;

	if(start+t<len)	partialSum[t] = input[start+t];
	else partialSum[t]=0;
	__syncthreads();

	if(start+tileSize+t<len)partialSum[tileSize+t]=input[start+tileSize+t];
	else partialSum[tileSize+t]=0;
	__syncthreads();

	for(unsigned int stride=tileSize;stride>=1; stride>>=1){
		__syncthreads();
		if (t<stride) partialSum[t]+=partialSum[t+stride];
		__syncthreads();
	}
	if(t==0) atomicAdd(output, partialSum[0]);
}

__global__
void vectorDotProd (double *a,  double *b,  double *c,int N)
{
    __shared__  double temp[tileSize];
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    temp[threadIdx.x] = a[index] * b[index];

    __syncthreads();

    if (threadIdx.x == 0)
    {
    	double sum = 0;
        for (int i = 0; i < tileSize; i++)
        {
            sum += temp[i];
        }
        __syncthreads();
        atomicAdd(c, sum);
    }
}


__global__ void
matrixMultiplication(double* M, double* N, double* P, int widthAHeightB, int heightA, int widthB) {

	int Mstart=widthAHeightB*tileSize*blockIdx.y;
	int Mend=Mstart+ widthAHeightB - 1;
	int mstep=tileSize;
	int Nstart=tileSize*blockIdx.x;
	int nstep=tileSize*widthB;
	double temp=0;

	__shared__ double Ms[tileSize][tileSize];
	__shared__ double Ns[tileSize][tileSize];

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
__global__ void
matrixTransposeSqr(double *P, double* M, int width, int height)
{
   unsigned int xIdx = blockDim.x * blockIdx.x + threadIdx.x;
   unsigned int yIdx = blockDim.y * blockIdx.y + threadIdx.y;

   if (xIdx < width && yIdx < height)
   {
       unsigned int inIdx  = xIdx + width * yIdx;
       unsigned int outIdx= yIdx + height * xIdx;
       P[outIdx] = M[inIdx];
   }
}




void initialization( double *M,  double *N, int arow, int acol, int brow, int bcol){
	for(int i=0;i<acol*arow;++i){
		M[i]=i%100;
	}

	for(int i=0;i<bcol*brow;++i){
		N[i]=i%100;
	}
}


void printInput( double *M,  double *N, int arow, int acol, int brow,  int bcol){
	printf("M:\n");
	for(int i=0;i<arow;i++){
		for(int j=0;j<acol;j++){
			printf("%5.2lf ", M[j+acol*i]);
		}
		printf("\n");
	}

	printf("N:\n");
		for(int i=0;i<brow;i++){
			for(int j=0;j<bcol;j++){
				printf("%5.2lf ", N[j+bcol*i]);
			}
			printf("\n");
		}
}


void printOutput( double *P_C,  double *P_G, int arow, int bcol){
	printf("P_C:\n");
	for(int i=0;i<arow;i++){
		for(int j=0;j<bcol;j++){
			printf("%5.2lf ", P_C[j+bcol*i]);
		}
		printf("\n");
	}

	printf("P_G:\n");
		for(int i=0;i<arow;i++){
			for(int j=0;j<bcol;j++){
				printf("%5.2lf ", P_G[j+bcol*i]);
			}
			printf("\n");
		}
}


void CPUVectorAddition(const double *A, const double *B, double *C, int numElements){
	for(int i = 0;i<numElements;i++){
		C[i] = A[i] + B[i];
	}
}
void CPUVectorSubtraction(const double *A, const double *B, double *C, int numElements){
	for(int i = 0;i<numElements;i++){
		C[i] = A[i] - B[i];
	}
}
void CPUVectorScaling(const double *A, double s, double *C, int numElements){
	for(int i = 0;i<numElements;i++){
		C[i] = A[i]*s;
	}
}
void CPUVectorSum(double *M, double *P_C, int len){
	double temp = 0;
	for(int i = 0;i<len;i++){
		temp+=M[i];
	}
	*(P_C) = temp;
}
void CPUMatrixMultiplication(double *M, double *N, double *P_C, double *P_G, int arow, int acol, int brow, int bcol){
	for(int i=0;i<arow;i++){ //row
		for(int j=0;j<bcol;j++){	//col
			double temp=0;
			for(int k=0;k<acol;k++){
				temp+=M[i*acol+k]*N[bcol*k+j];
			}
			P_C[i*bcol+j]=temp;
		}
	}
}
void CPUVectorDotProd( const  double *A, const  double *B,   double *C, int N) {
	 double temp=0;
    for(int i = 0; i != N; ++i ) {
    	temp += A[i]*B[i];
    }
    C[0]=temp;
}
void CPUMatrixTransposeSqr(double *A, double *C, const int arow, const int acol) {
    #pragma omp parallel for
    for(int n = 0; n<arow*acol; n++) {
        int i = n/arow;
        int j = n%arow;
        C[n] = A[acol*j + i];
    }
}




void CPUGPUCheck(double *P_C, double *P_G, int arow, int acol, int brow, int bcol){
	for (int i = 0; i < arow*bcol; ++i){
			if (fabs(P_C[i]-P_G[i]) > 1e-5){
				fprintf(stderr, "Result verification failed at element %d!\n", i);
				exit(EXIT_FAILURE);
		    }
		}

		printf("The result of CUDA Computation is Correct.\n");
}

double cpuTimer(){
	struct timeval clock;
	gettimeofday(&clock, NULL);
	return ( (double)clock.tv_sec + (double)clock.tv_usec * 1.e-6 );
}
