NVCC := nvcc

# internal flags
NVCCFLAGS   := -std=c++11 -O3 -arch=sm_35 --compiler-options="-O2 -pipe -march=native -Wall -fopenmp" -Xcompiler -rdynamic --generate-line-info
CCFLAGS     := 
LDFLAGS     := -L/opt/cuda/lib64 -lcublas
NAME 		:= cu-roof
PREFIX		:= .
N 			:= 1

$(PREFIX)/$(NAME)$N: main.cu Makefile series.sh
	$(NVCC) -DPARN=$N $(NVCCFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS)


clean:
	rm -f ./$(NAME)

