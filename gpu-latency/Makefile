NVCC := nvcc

# internal flags
NVCCFLAGS   := -std=c++11 -O3 -arch=sm_70 --compiler-options="-O2 -pipe -march=native -Wall -fopenmp" -Xcompiler -rdynamic --generate-line-info
CCFLAGS     := 
LDFLAGS     := -L/opt/cuda/lib64 -lcublas -lnvidia-ml
NAME 		:= latency
PREFIX		:= .
N 			:= 1


$(PREFIX)/cuda-$(NAME): main.cu Makefile
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS)

main.hip: main.cu
	hipify-perl main.cu > main.hip

$(PREFIX)/hip-$(NAME): main.hip Makefile ../rocm-metrics/rocm-metrics.hpp
	echo $(HIP_HOME)
	hipcc -std=c++20 -I$(HIP_HOME)/include/rocprofiler/ -I$(HIP_HOME)/hsa/include/hsa -L$(HIP_HOME)/rocprofiler/lib -lrocprofiler64 -lhsa-runtime64 -lrocm_smi64 -ldl -o $@ $<

clean:
	rm -f cuda-$(NAME) hip-$(NAME)
