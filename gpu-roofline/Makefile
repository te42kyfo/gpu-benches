NVCC := nvcc

TEMP_NVCC := $(shell which nvcc)
CUDA_HOME :=  $(shell echo $(TEMP_NVCC) | rev |  cut -d'/' -f3- | rev)

# internal flags
NVCCFLAGS   := -std=c++11 -O3 -arch=sm_70 --compiler-options="-O2 -pipe -Wall -fopenmp -g" -Xcompiler -rdynamic --generate-line-info  -Xcompiler \"-Wl,-rpath,$(CUDA_HOME)/extras/CUPTI/lib64/\"

CCFLAGS     := 
LDFLAGS     := -L/opt/cuda/lib64 -L$(CUDA_HOME)/extras/CUPTI/lib64 -lcupti -lcuda   -lnvidia-ml
NAME 		:= roof
PREFIX		:= .
N 			:= 100

$(PREFIX)/cu-$(NAME)$N: main.cu Makefile series.sh
	$(NVCC) -DPARN=$N $(NVCCFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS)


main.hip: main.cu
	hipify-perl main.cu > main.hip

$(PREFIX)/hip-$(NAME)$N: main.hip Makefile ../rocm-metrics/rocm-metrics.hpp
	echo $(HIP_HOME)
	hipcc -DPARN=$N -std=c++20 -I$(HIP_HOME)/include/rocprofiler/ -I$(HIP_HOME)/hsa/include/hsa -L$(HIP_HOME)/rocprofiler/lib -lrocprofiler64 -lhsa-runtime64 -lrocm_smi64 -ldl -fopenmp -o $@ $<


clean:
	rm -f ./$(NAME)

