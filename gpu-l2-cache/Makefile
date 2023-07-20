NVCC := nvcc

TEMP_NVCC := $(shell which nvcc)
CUDA_HOME :=  $(shell echo $(TEMP_NVCC) | rev |  cut -d'/' -f3- | rev)

HIP_HOME :=  /opt/rocm


# internal flags
NVCCFLAGS   := -std=c++17 -O3 -arch=sm_70 --compiler-options="-O2 -pipe -Wall -fopenmp -g" -Xcompiler -rdynamic --generate-line-info  -Xcompiler \"-Wl,-rpath,$(CUDA_HOME)/extras/CUPTI/lib64/\"
CCFLAGS     := 
LDFLAGS     := -L$(CUDA_HOME)/lib64 -L$(CUDA_HOME)/extras/CUPTI/lib64 -lcupti -lcuda   -lnvidia-ml -lnvperf_host -lnvperf_target
NAME 		:= l2-cache
PREFIX		:= .
INCLUDES 	:=  -I$(CUDA_HOME)/extras/CUPTI/include -I$(CUDA_HOME)/include

$(PREFIX)/cuda-$(NAME): main.cu Makefile
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS)


main.hip: main.cu
	hipify-perl main.cu > main.hip

$(PREFIX)/hip-$(NAME): main.hip Makefile ../rocm-metrics/rocm-metrics.hpp
	hipcc -std=c++20 -I$(HIP_HOME)/include/rocprofiler/ -I$(HIP_HOME)/hsa/include/hsa -L$(HIP_HOME)/rocprofiler/lib -lrocprofiler64 -lhsa-runtime64 -ldl -o $@ $<

clean:
	rm -f ./cuda-$(NAME)
	rm -f main.hip
	rm -f ./hip-$(NAME)
