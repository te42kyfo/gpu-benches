NVCC := nvcc

# internal flags
NVCCFLAGS   := -std=c++11 -O3 -arch=sm_35 --compiler-options="-O2 -pipe -march=native -Wall -fopenmp" -Xcompiler -rdynamic --generate-line-info
CCFLAGS     :=
LDFLAGS     := -L/opt/cuda/lib64
NAME 		:= cuda-memcpy
PREFIX		:= .


$(PREFIX)/$(NAME): main.cu Makefile
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS)

clean:
	rm -f ./$(NAME)
