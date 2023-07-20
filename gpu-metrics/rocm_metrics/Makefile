HIP_HOME := /opt/rocm

HIPFLAGS := -ldl -O3 -std=c++2a -I/opt/rocm/include/hip $(shell python3-config --includes) -I$(HIP_HOME)/include/rocprofiler/ -I$(HIP_HOME)/hsa/include/hsa -L$(HIP_HOME)/rocprofiler/lib -lrocprofiler64 -lhsa-runtime64



test_rocm_metrics: test_rocm_metrics.hip
	hipcc $< -o $@ $(HIPFLAGS)

# end
