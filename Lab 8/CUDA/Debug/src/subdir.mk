################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/cudaaddition.cu 

CPP_SRCS += \
../src/template_cpu.cpp 

OBJS += \
./src/cudaaddition.o \
./src/template_cpu.o 

CU_DEPS += \
./src/cudaaddition.d 

CPP_DEPS += \
./src/template_cpu.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-10.1/bin/nvcc -I"/usr/local/cuda-10.1/samples/0_Simple" -I"/usr/local/cuda-10.1/samples/common/inc" -I"/home/kasperski/cuda-workspace/LAB8cudaaddition" -G -g -O0 -gencode arch=compute_52,code=sm_52  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-10.1/bin/nvcc -I"/usr/local/cuda-10.1/samples/0_Simple" -I"/usr/local/cuda-10.1/samples/common/inc" -I"/home/kasperski/cuda-workspace/LAB8cudaaddition" -G -g -O0 --compile --relocatable-device-code=false -gencode arch=compute_52,code=compute_52 -gencode arch=compute_52,code=sm_52  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-10.1/bin/nvcc -I"/usr/local/cuda-10.1/samples/0_Simple" -I"/usr/local/cuda-10.1/samples/common/inc" -I"/home/kasperski/cuda-workspace/LAB8cudaaddition" -G -g -O0 -gencode arch=compute_52,code=sm_52  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-10.1/bin/nvcc -I"/usr/local/cuda-10.1/samples/0_Simple" -I"/usr/local/cuda-10.1/samples/common/inc" -I"/home/kasperski/cuda-workspace/LAB8cudaaddition" -G -g -O0 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


