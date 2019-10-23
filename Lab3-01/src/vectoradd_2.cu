/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/**
 * Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */

#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>


/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
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

/**
 * Host main routine
 */
int
main(void)
{

	//int blocksPerGrid =2;

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

	float memperarr=(int)(deviceCount*totalmem/12);
	printf("\n%f\n", memperarr*12);
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Print the vector length to be used, and compute its size
    float numElements = 10000000;


    if(numElements>memperarr){
    	printf("Size of array exceeds total device memory!\nMaximal number of elements in array is %f",memperarr);
		printf("\nArray dimension will be changed to prevent crash\nWARNING! %f elements will be lost!!\n\n",numElements-memperarr);
    	numElements=memperarr;


    }

    //size_t size = numElements * sizeof(float);

    printf("[Vector addition of %f elements]\n", numElements);

    // Allocate the host input vector A
    float *A; //= (float *)malloc(size);

    // Allocate the host input vector B
    float *B; //= (float *)malloc(size);

    // Allocate the host output vector C
    float *C;//= (float *)malloc(size);



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
    // Verify that allocations succeeded
    if (A == NULL || B == NULL || C == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i)
    {
        A[i] = rand()/(float)RAND_MAX;
        B[i] = rand()/(float)RAND_MAX;
    }

    // Launch the Vector Add CUDA Kernel

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

    // Copy the device result vector in device memory to the host result vector
    // in host memory.

    cudaDeviceSynchronize();

    // Verify that the result vector is correct
    for (int i = 0; i < numElements; ++i)
    {
        if (fabs(A[i] + B[i] - C[i]) > 1e-5)
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

    printf("Done\n");
    return 0;
}

