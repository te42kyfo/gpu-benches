# GPU benchmarks
This is a collection of GPU micro benchmarks. Each test is designed to test a particular scenario or hardware mechanism. Some of the benchmarks have been used to produce data for these papers:

["Analytical performance estimation during code generation on modern GPUs" ](https://doi.org/10.1016/j.jpdc.2022.11.003)

["Performance engineering for real and complex tall & skinny matrix multiplication kernels on GPUs"](http://dx.doi.org/10.1177/1094342020965661)



Benchmarks that are called ```gpu-<benchmarkname>``` are hipifyable! Whereas the default Makefile target builds the CUDA executable ```cuda-<benchmarkname>```, the target ```make hip-<benchmarkname>``` uses the hipify-perl tool to create a file ```main.hip``` from the ```main.cu``` file, and builds it using the hip compiler. The CUDA main files are written so that the hipify tool works without further intervention. 



## gpu-stream

Measures the bandwidth of streaming kernels for varying occupancy. A shared memory allocation serves as a spoiler, so that only two thread blocks can run per SM. Scanning the thread block size from 32 to 1024 scans the occupancy from 3% to 100%.


Kernel | Formula |   |
-------|----------|---|
init  | A[i] = c  |   1 store stream
read | sum = A[i] |   1 load stream
scale | A[i] = B[i] * c |   1 load stream, 1 store stream
triad | A[i] = B[i] + c * C[i] |  2 load streams, 1 store stream
3pt | A[i] = B[i-1] + B[i] + B[i+1] |  1 load streams, 1 store stream
5pt | A[i] = B[i-2] + B[i-1] + B[i] + B[i+1] + B[i+2] |  1 load streams, 1 store stream


Results from a NVIDIA-H100-PCIe  / CUDA 11.7
``` console
blockSize   threads       %occ  |                init       read       scale     triad       3pt        5pt
       32        3648      3 %  |  GB/s:         228         96        183        254        168        164
       64        7296    6.2 %  |  GB/s:         452        189        341        459        316        310
       96       10944    9.4 %  |  GB/s:         676        277        472        635        443        436
      128       14592   12.5 %  |  GB/s:         888        368        607        821        567        558
      160       18240   15.6 %  |  GB/s:        1093        449        704        966        680        670
      192       21888   18.8 %  |  GB/s:        1301        533        817       1121        794        781
      224       25536   21.9 %  |  GB/s:        1495        612        925       1264        903        889
      256       29184   25.0 %  |  GB/s:        1686        702       1037       1399       1005        989
      288       32832   28.1 %  |  GB/s:        1832        764       1124       1487       1100       1082
      320       36480   31.2 %  |  GB/s:        2015        841       1213       1564       1188       1169
      352       40128   34.4 %  |  GB/s:        2016        908       1295       1615       1269       1250
      384       43776   37.5 %  |  GB/s:        2016        985       1378       1644       1348       1326
      416       47424   40.6 %  |  GB/s:        2016       1045       1439       1641       1415       1395
      448       51072   43.8 %  |  GB/s:        2016       1116       1497       1649       1472       1453
      480       54720   46.9 %  |  GB/s:        2016       1179       1544       1655       1521       1505
      512       58368   50.0 %  |  GB/s:        2017       1261       1583       1675       1556       1545
      544       62016   53.1 %  |  GB/s:        2016       1300       1591       1669       1572       1563
      576       65664   56.2 %  |  GB/s:        2016       1362       1607       1678       1587       1579
      608       69312   59.4 %  |  GB/s:        2018       1416       1619       1689       1598       1592
      640       72960   62.5 %  |  GB/s:        2016       1473       1639       1712       1613       1607
      672       76608   65.6 %  |  GB/s:        2016       1527       1638       1714       1618       1613
      704       80256   68.8 %  |  GB/s:        2015       1578       1644       1725       1625       1619
      736       83904   71.9 %  |  GB/s:        2016       1624       1651       1738       1632       1628
      768       87552   75.0 %  |  GB/s:        2016       1680       1666       1755       1642       1638
      800       91200   78.1 %  |  GB/s:        2015       1714       1663       1758       1645       1642
      832       94848   81.2 %  |  GB/s:        2016       1759       1668       1770       1649       1647
      864       98496   84.4 %  |  GB/s:        2016       1795       1673       1779       1654       1651
      896      102144   87.5 %  |  GB/s:        2016       1837       1686       1796       1663       1662
      928      105792   90.6 %  |  GB/s:        2018       1871       1684       1800       1666       1664
      960      109440   93.8 %  |  GB/s:        2016       1897       1688       1808       1672       1670
      992      113088   96.9 %  |  GB/s:        2016       1919       1693       1818       1678       1675
     1024      116736  100.0 %  |  GB/s:        2016       1942       1704       1832       1686       1683
```
The results for the SCALE kernel and a selection of GPUs:

![stream plot](gpu-stream/cuda-stream.svg)

Note that the H100 results are for the PCIe version, which has lower DRAM bandwidth than the SXM version!

## gpu-latency

Pointer chasing benchmark for latency measurement. A single warp fully traverses a buffer in random order. A partitioning scheme is used to ensure that all cache lines are hit exactly once before they are accessed again. Latency in clock cycles is computed with the current clock rate.

![latency plot](gpu-latency/latency_plot.svg)

Sharp L1 cache transitions at 128/192/256 kB for NVIDIAS V100/A100/H100 and at 16kB for AMD's MI210. V100 and MI210 both have a 6MB L2 cache. The A100's and H100 have a segmented L2 cache at 2x20MB and 2x25MB, which results as a small intermediate plateau when data is fetched from the far L2 section. 


## gpu-cache

Measures bandwidths of the higher cache levels. Launches one thread block per SM. Each thread block repeatedly reads the contents of the same buffer. Varying buffer sizes changes the targeted cache level.

![cache plot](gpu-cache/cuda-cache.svg)

The 16kB (MI100/MI210), 128kB (V100), 192kB (A100) and 256 kB (H100) L1 cache capacities are very pronounced and sharp. The three NVIDIA architectures both transfer close to 128B/cycle/SM, the maximum measured value on AMD's MI100 and MI210 depends on the data type. For double precision, the maximum is 32B/cycle/CU. For single precision and 16B data types (either float4 or double2) the bandwidth is up to 64B. 

Even for data set sizes larger than the L2 cache, there is no clear performance transition drop. Because all thread blocks read the same data, there is a lot of reuse potential inside the L2 cache before the data is evicted. A100 and H100 drop slightly at 20/25MB, when the capacity of a single cache section is exceeded. Beyond this point, data cannot be replicated in both L2 cache sections and the maximum bandwidth drops, as data has also to be fetched from the other section.

## gpu-l2-cache

Measures bandwidths of shared cache levels. This benchmark explicitly does not target the L1 caches.

![cache plot](gpu-l2-cache/cuda-cache.svg)

All three GPUs have a similar L2 cache bandwidths of about 5.x TB/s, though with different capactities. The stand out here is the RX6900XT, which has a second shared cache level, the 128MB Infinity Cache. At almost 1.92 TB/s, it is as fast as the A100's DRAM.
At the very beginning, the RX6900XT semi-shared L1 cache can be seen, where for some block placements the 4 L1 caches have a small effect. 


## gpu-strides

Read only, L1 cache benchmark that accesses memory with strides 1 to 128. The bandwidth is converted to Bytes per cycle and SM. The strides from 1 to 128 are formatted in a 16x8 tableau, because that highlights the recurring patterns of multiples of 2/4/8/16. 

![image](https://user-images.githubusercontent.com/3269202/214321378-9969f484-1067-47cd-8e11-1bab56c26534.png)

These multiples are important for NVIDIA's architecture, which clearly have their L1 cache structured in a 16 banks of 8B. For strides that are a multiple of 16, every single thread accesses data from the same cache bank. The rate of address translation is reduced when addresses do not fall into the same 128B cache line anymore.

AMD's MI210 appears to have even more banks, with especially stark slowdowns to less than 4B/cycle for multiples of 32. 

Testing the stencil-like, 2D structured grid access with different thread block shapes reveals differences in the L1 cache throughput:

![image](https://user-images.githubusercontent.com/3269202/214323402-16debc41-72c4-4824-aa2a-bc34772f361d.png)

(see the generated machine code of MI210 and A100 here: https://godbolt.org/z/1PvWqs9Kf)

AMD's MI210 is fine (at its much lower level), as long as contiguous blocks of at least 4 threads are accessed. NVIDIA's only reach their maximum throughput for 16 wide thread blocks. 

Along with the L1 cache size increass, both Ampere and Hopper also slightly improve the rate of L1 cache address lookups. 


## cuda-roofline

This program scans a range of Computational Intensities, by varying the amount of inner loop trips.  It is suitable both to study the transition from memory- to compute bound codes as well as power consumption, clock frequencies and temperatures when using multiple GPUs. The shell script series.sh builds an executable for each value, and executes them one afer another after finishing building.

The Code runs simultaneously on all available devices. Example output on four Tesla V100 PCIe 16GB:

```console
1 640 blocks     0 its      0.125 Fl/B        869 GB/s       109 GF/s   1380 Mhz   138 W   60°C
2 640 blocks     0 its      0.125 Fl/B        869 GB/s       109 GF/s   1380 Mhz   137 W   59°C
3 640 blocks     0 its      0.125 Fl/B        869 GB/s       109 GF/s   1380 Mhz   124 W   56°C
0 640 blocks     0 its      0.125 Fl/B        869 GB/s       109 GF/s   1380 Mhz   124 W   54°C

1 640 blocks     8 its      1.125 Fl/B        861 GB/s       968 GF/s   1380 Mhz   159 W   63°C
0 640 blocks     8 its      1.125 Fl/B        861 GB/s       968 GF/s   1380 Mhz   142 W   56°C
2 640 blocks     8 its      1.125 Fl/B        861 GB/s       968 GF/s   1380 Mhz   157 W   62°C
3 640 blocks     8 its      1.125 Fl/B        861 GB/s       968 GF/s   1380 Mhz   144 W   59°C
[...]
0 640 blocks    64 its      8.125 Fl/B        811 GB/s      6587 GF/s   1380 Mhz   223 W   63°C
3 640 blocks    64 its      8.125 Fl/B        813 GB/s      6604 GF/s   1380 Mhz   230 W   66°C
1 640 blocks    64 its      8.125 Fl/B        812 GB/s      6595 GF/s   1380 Mhz   241 W   71°C
2 640 blocks    64 its      8.125 Fl/B        813 GB/s      6603 GF/s   1380 Mhz   243 W   69°C
```


## cuda-memcpy

Measures the host-to-device transfer rate of the cudaMemcpy function over a range of transfer sizes

Example output for a Tesla V100 PCIe 16GB
``` console
         1kB     0.03ms    0.03GB/s   0.68%
         2kB     0.03ms    0.06GB/s   5.69%
         4kB     0.03ms    0.12GB/s   8.97%
         8kB     0.03ms    0.24GB/s   6.25%
        16kB     0.04ms    0.44GB/s   5.16%
        32kB     0.04ms    0.93GB/s   2.70%
        64kB     0.04ms    1.77GB/s   5.16%
       128kB     0.04ms    3.46GB/s   7.55%
       256kB     0.05ms    5.27GB/s   1.92%
       512kB     0.07ms    7.53GB/s   1.03%
      1024kB     0.11ms    9.25GB/s   2.52%
      2048kB     0.20ms   10.50GB/s   1.07%
      4096kB     0.37ms   11.41GB/s   0.58%
      8192kB     0.71ms   11.86GB/s   0.44%
     16384kB     1.38ms   12.11GB/s   0.14%
     32768kB     2.74ms   12.23GB/s   0.03%
     65536kB     5.46ms   12.29GB/s   0.08%
    131072kB    10.89ms   12.32GB/s   0.02%
    262144kB    21.75ms   12.34GB/s   0.00%
    524288kB    43.47ms   12.35GB/s   0.00%
   1048576kB    86.91ms   12.35GB/s   0.00%
```

## um-stream

Measures CUDA Unified Memory transfer rate using a STREAM triad kernel. A range of data set sizes is used, both smaller and larger than the device memory. Example output on a Tesla V100 PCIe 16GB:

```console
 buffer size      time   spread   bandwidth
       24 MB     0.1ms     3.2%   426.2GB/s
       48 MB     0.1ms    24.2%   511.6GB/s
       96 MB     0.1ms     0.8%   688.0GB/s
      192 MB     0.3ms     1.8%   700.0GB/s
      384 MB     0.5ms     0.5%   764.6GB/s
      768 MB     1.0ms     0.2%   801.8GB/s
     1536 MB     2.0ms     0.0%   816.9GB/s
     3072 MB     3.9ms     0.1%   822.9GB/s
     6144 MB     7.8ms     0.2%   823.8GB/s
    12288 MB    15.7ms     0.1%   822.1GB/s
    24576 MB  5108.3ms     0.5%     5.0GB/s
    49152 MB 10284.7ms     0.8%     5.0GB/s
```





## cuda-incore

Measures the latency and throughput of FMA, DIV and SQRT operation. It scans combinations of ILP=1..8, by generating 1..8 independent dependency chains, and TLP, by varying the warp count on a SM from 1 to 32. The final output is a ILP/TLP table, with the reciprocal throughputs (cycles per operation):

Example output on a Tesla V100 PCIe 16GB:

``` console
DFMA
  8.67   4.63   4.57   4.66   4.63   4.72   4.79   4.97
  4.29   2.32   2.29   2.33   2.32   2.36   2.39   2.48
  2.14   1.16   1.14   1.17   1.16   1.18   1.20   1.24
  1.08   1.05   1.05   1.08   1.08   1.10   1.12   1.14
  1.03   1.04   1.04   1.08   1.07   1.10   1.11   1.14
  1.03   1.04   1.04   1.08   1.07   1.10   1.10   1.14

DDIV
111.55 111.53 111.53 111.53 111.53 668.46 779.75 891.05
 55.76  55.77  55.76  55.76  55.76 334.26 389.86 445.51
 27.88  27.88  27.88  27.88  27.88 167.12 194.96 222.82
 14.11  14.11  14.11  14.11  14.11  84.77  98.89 113.00
  8.48   8.48   8.48   8.48   8.48  50.89  59.36  67.84
  7.51   7.51   7.51   7.51   7.51  44.98  52.48  59.97

DSQRT
101.26 101.26 101.26 101.26 101.26 612.76 714.79 816.83
 50.63  50.62  50.63  50.63  50.62 306.36 357.38 408.40
 25.31  25.31  25.31  25.31  25.31 153.18 178.68 204.19
 13.56  13.56  13.56  13.56  13.56  82.75  96.83 110.29
  9.80   9.80   9.80   9.80   9.80  60.47  70.54  80.62
  9.61   9.61   9.61   9.61   9.61  58.91  68.72  78.53
```

Some Features can be extracted from the plot.

Latencies:
 - DFMA: 8 cycles
 - DDIV: 112 cycles
 - DSQRT: 101 cycles
 
Throughput of one warp (runs on one SM quadrant), no dependencies:
 - DFMA: 1/4 per cycle (ILP 2, to ops overlap)
 - DDIV: 1/112 per cycle (no ILP/overlap)
 - DSQRT: 1/101 per cycle (no ILP/overlap)
  
Throughput of multiple warps (all SM quadrants), dependencies irrelevant:
 - DFMA: 1 per cycle 
 - DDIV: 1/7.5 cycles
 - DSQRT: 1/9.6 cycles
 



