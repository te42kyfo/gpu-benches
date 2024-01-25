# Repeated Small Kernel Performance

Queue 10000 calls of a streaming SCALE kernel of varying size. Use commandline option "-graph" to use the cudaGraph/hipGraph API. 

![latency plot](gpu-small-kernels/repeated-stream.svg)


Each device gets a fit of \$a,b\$ for the function

$$T = \frac{V}{a + V/b}$$

which models the performance with a startup overhead \$a\$ and a bandwidth \$b\$ depending on the data volume \$V\$.
