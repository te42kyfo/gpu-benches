NVCC := nvcc


TEMP_NVCC := $(shell which nvcc)
CUDA_HOME :=  $(shell echo $(TEMP_NVCC) | rev |  cut -d'/' -f3- | rev)


# internal flags
NVCCFLAGS   := -std=c++11 -O3 -arch=sm_70 --compiler-options="-O2 -pipe -Wall -fopenmp -g" -Xcompiler -rdynamic --generate-line-info  -Xcompiler \"-Wl,-rpath,$(CUDA_HOME)/extras/CUPTI/lib64/\" 
CCFLAGS     := 
LDFLAGS     := -L/opt/cuda/lib64 -L$(CUDA_HOME)/extras/CUPTI/lib64 -lcupti -lcuda   -lnvidia-ml
NAME 		:= cuda-incore
PREFIX		:= .
INCLUDES 	:=  -I$(CUDA_HOME)/extras/CUPTI/include

$(PREFIX)/$(NAME): main.cu Makefile
	echo $(CUDA_HOME)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS)


clean:
	rm -f ./$(NAME)
