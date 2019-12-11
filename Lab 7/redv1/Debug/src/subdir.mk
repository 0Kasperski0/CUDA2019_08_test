################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/template.cu 

CPP_SRCS += \
../src/template_cpu.cpp 

OBJS += \
./src/template.o \
./src/template_cpu.o 

CU_DEPS += \
./src/template.d 

CPP_DEPS += \
./src/template_cpu.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-10.0/bin/nvcc -I"/usr/local/cuda-10.0/samples/0_Simple" -I"/usr/local/cuda-10.0/samples/common/inc" -I"/home/cuda-s08/cudawdir/LAB7/redv1" -G -g -O0 -gencode arch=compute_61,code=sm_61  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-10.0/bin/nvcc -I"/usr/local/cuda-10.0/samples/0_Simple" -I"/usr/local/cuda-10.0/samples/common/inc" -I"/home/cuda-s08/cudawdir/LAB7/redv1" -G -g -O0 --compile --relocatable-device-code=false -gencode arch=compute_61,code=compute_61 -gencode arch=compute_61,code=sm_61  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-10.0/bin/nvcc -I"/usr/local/cuda-10.0/samples/0_Simple" -I"/usr/local/cuda-10.0/samples/common/inc" -I"/home/cuda-s08/cudawdir/LAB7/redv1" -G -g -O0 -gencode arch=compute_61,code=sm_61  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-10.0/bin/nvcc -I"/usr/local/cuda-10.0/samples/0_Simple" -I"/usr/local/cuda-10.0/samples/common/inc" -I"/home/cuda-s08/cudawdir/LAB7/redv1" -G -g -O0 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


