################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/simpleCUBLAS.cpp 

OBJS += \
./src/simpleCUBLAS.o 

CPP_DEPS += \
./src/simpleCUBLAS.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-10.0/bin/nvcc -I"/usr/local/cuda-10.0/samples/7_CUDALibraries" -I"/usr/local/cuda-10.0/samples/common/inc" -I"/home/cuda-s08/cudawdir/LAB6/libmult" -G -g -O0 -gencode arch=compute_61,code=sm_61  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-10.0/bin/nvcc -I"/usr/local/cuda-10.0/samples/7_CUDALibraries" -I"/usr/local/cuda-10.0/samples/common/inc" -I"/home/cuda-s08/cudawdir/LAB6/libmult" -G -g -O0 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


