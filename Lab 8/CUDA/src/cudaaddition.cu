#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

__global__ void
vectorAdd(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        C[i] = A[i] + B[i];
    }
   /* printf("threadIdx:(%d, %d, %d) blockIdx:(%d, %d, %d) blockDim:(%d, %d, %d) "
     "gridDim:(%d, %d, %d)\n", threadIdx.x, threadIdx.y, threadIdx.z,
     blockIdx.x, blockIdx.y, blockIdx.z, blockDim.x, blockDim.y, blockDim.z,
     gridDim.x,gridDim.y,gridDim.z);*/
}

double cpuTimer(){
	struct timeval clock;
	gettimeofday(&clock, NULL);
	return ( (double)clock.tv_sec + (double)clock.tv_usec * 1.e-6 );
}


int
main(void)
{

	printf("# of elements %.0f\n",numElements);

	double duration,ti;
	float totalmem=0;
	int deviceCount = 0;
	int dev;

	cudaGetDeviceCount(&deviceCount);
	for (dev = 0; dev < deviceCount; ++dev) {
	    cudaSetDevice(dev);
	    cudaDeviceProp deviceProp;
	    cudaGetDeviceProperties(&deviceProp, dev);
	    printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);
	    totalmem+=deviceProp.totalGlobalMem;
	}

	float elperarr=(int)(deviceCount*totalmem/(7*4));
	//printf("\n%.0f\n", elperarr*12);

    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Print the vector length to be used, and compute its size
    //float numElements = 2<<26;


    if(numElements>elperarr){
    	printf("Size of array exceeds total device memory!\nMaximal number of elements in array is %.0f",elperarr);
		printf("\nArray dimension will be changed to prevent crash\nWARNING! %.0f elements will be lost!!\n\n",numElements-elperarr);
    	numElements=elperarr;


    }


    printf("[Vector addition of %.0f elements]\n", numElements);

    // Allocate vectors
    float *A;
    float *B;
    float *C;
    float *D;
    float *E;
    float *F;
    float *G;



    err = cudaMallocManaged(&A,numElements*sizeof(float));

    if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

    err = cudaMallocManaged(&B,numElements*sizeof(float));

    if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
    err =  cudaMallocManaged(&C,numElements*sizeof(float));

    if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
    err = cudaMallocManaged(&D,numElements*sizeof(float));


        if (err != cudaSuccess)
            {
                fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }
        err = cudaMallocManaged(&E,numElements*sizeof(float));


    	if (err != cudaSuccess)
                {
                    fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
                    exit(EXIT_FAILURE);
                }


        err = cudaMallocManaged(&F,numElements*sizeof(float));


        if (err != cudaSuccess)
                    {
                        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
                        exit(EXIT_FAILURE);
                    }
        err = cudaMallocManaged(&G,numElements*sizeof(float));


        if (err != cudaSuccess)
                        {
                            fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
                            exit(EXIT_FAILURE);
        }

    /////////////////////////////////////////////////////////////////

    // Verify that allocations succeeded
    if (A == NULL || B == NULL || C == NULL || D == NULL || E == NULL || F == NULL || G == NULL)
    {
        fprintf(stderr, "Failed to allocate vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i)
    {
        A[i] = rand()/(float)RAND_MAX;
        B[i] = rand()/(float)RAND_MAX;
        E[i] = rand()/(float)RAND_MAX;
        G[i] = rand()/(float)RAND_MAX;
    }

    // Launch the Vector Add CUDA Kernel
    ti = cpuTimer();
	int threadsPerBlock = 256;
	int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;



    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, numElements);

    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    cudaDeviceSynchronize();

    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(E, C, D, numElements);

    err = cudaGetLastError();

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        cudaDeviceSynchronize();

    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(D, G, F, numElements);

    err = cudaGetLastError();

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }


        cudaDeviceSynchronize();

        duration = cpuTimer() - ti;
        printf("Total kernel addition duration: %.6f s\n", duration);




    // Verify that the result vector is correct
    for (int i = 0; i < numElements; ++i)
    {
        if (fabs(A[i] + B[i] - C[i]) > 1e-5)
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    for (int i = 0; i < numElements; ++i)
        {
            if (fabs(C[i] + E[i] - D[i]) > 1e-5)
            {
                fprintf(stderr, "Result verification failed at element %d!\n", i);
                exit(EXIT_FAILURE);
            }
        }

    for (int i = 0; i < numElements; ++i)
        {
            if (fabs(G[i] + D[i] - F[i]) > 1e-5)
            {
                fprintf(stderr, "Result verification failed at element %d!\n", i);
                exit(EXIT_FAILURE);
            }
        }
    printf("Test PASSED\n");


    // Free host memory
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    cudaFree(D);
    cudaFree(E);
    cudaFree(F);
    cudaFree(G);

    printf("Done\n");
    //return 0;



}
