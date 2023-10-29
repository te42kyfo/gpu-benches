NVCC := nvcc

# internal flags
NVCCFLAGS   := -std=c++11 -O3 -arch=sm_80 --compiler-options="-O2 -pipe -march=native -Wall -fopenmp" -Xcompiler -rdynamic --generate-line-info
CCFLAGS     := 
LDFLAGS     := -L/opt/cuda/lib64 -lcublas
NAME 		:= um-stream
PREFIX		:= .


$(PREFIX)/$(NAME): main.cu Makefile ../dtime.hpp ../MeasurementSeries.hpp ../gpu-error.h
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS)

clean:
	rm -f ./$(NAME)

