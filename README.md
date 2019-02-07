# cuda-roofline

Short CUDA code that scans a range of Computational Intensities, by varying the amount of inner loop trips. The shell script series.sh builds an executable for each value, and executes them one afer another after finishing building.

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

3 640 blocks    56 its      7.125 Fl/B        841 GB/s      5990 GF/s   1380 Mhz   227 W   66°C
1 640 blocks    56 its      7.125 Fl/B        841 GB/s      5990 GF/s   1372 Mhz   249 W   71°C
2 640 blocks    56 its      7.125 Fl/B        841 GB/s      5990 GF/s   1380 Mhz   235 W   69°C
0 640 blocks    56 its      7.125 Fl/B        841 GB/s      5990 GF/s   1380 Mhz   220 W   62°C

0 640 blocks    64 its      8.125 Fl/B        811 GB/s      6587 GF/s   1380 Mhz   223 W   63°C
3 640 blocks    64 its      8.125 Fl/B        813 GB/s      6604 GF/s   1380 Mhz   230 W   66°C
1 640 blocks    64 its      8.125 Fl/B        812 GB/s      6595 GF/s   1380 Mhz   241 W   71°C
2 640 blocks    64 its      8.125 Fl/B        813 GB/s      6603 GF/s   1380 Mhz   243 W   69°C
```


# cuda-memcpy

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

# um-stream

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


# cuda-cache

Measures bandwidths of different cache levels. Launches one thread block per SM. Each thread block reads the contents of the same buffer. Varying buffer sizes changes the targeted cache level. Example output on a Tesla V100 PCIe 16GB:
```console
     data set   exec time     spread        Eff. bw      DRAM read     DRAM write        L2 Read       L2 Write      Tex Read
         4 kB         1ms       0.2%    5559.0 GB/s       0.0 GB/s       0.0 GB/s       0.8 GB/s       0.0 GB/s   11217.9 GB/s
         8 kB         1ms       3.9%   10983.3 GB/s       0.0 GB/s       0.0 GB/s       1.1 GB/s       0.0 GB/s   11272.8 GB/s
        16 kB         1ms       0.1%   12374.0 GB/s       0.0 GB/s       0.0 GB/s       1.3 GB/s       0.0 GB/s   12501.7 GB/s
        32 kB         2ms       0.1%   13219.3 GB/s       0.0 GB/s       0.0 GB/s       1.3 GB/s       0.0 GB/s   13297.7 GB/s
        64 kB         4ms       0.1%   13667.4 GB/s       0.0 GB/s       0.0 GB/s       1.4 GB/s       0.0 GB/s   13708.2 GB/s
       128 kB         8ms       4.8%   13271.6 GB/s       0.0 GB/s       0.0 GB/s    1380.6 GB/s       0.0 GB/s   13706.6 GB/s
       256 kB        91ms       4.7%    2310.2 GB/s       0.0 GB/s       0.0 GB/s    2520.3 GB/s       0.0 GB/s    2521.9 GB/s
       512 kB       176ms       6.8%    2383.2 GB/s       0.0 GB/s       0.0 GB/s    2544.9 GB/s       0.0 GB/s    2544.4 GB/s
      1024 kB       338ms       4.3%    2482.0 GB/s       0.0 GB/s       0.0 GB/s    2544.2 GB/s       0.0 GB/s    2544.0 GB/s
```

# cuda-stream

Measures the bandwidth of several streaming kernels for varying occupancy.


Kernel | Formula |  | |
-------|----------|--|--|
init  | A[i] = c |  | 1 store stream
sumN | sum += A[i] | N times unrolling, no global reduction | N load streams
dot | sum += A[i] * B[i] | no global reduction | 2 load streams
tdot | sum += A[i] * B[i] * C[i] | no global reduction | 3 load streams
scale | A[i] = B[i] * c |  | 1 load stream, 1 store stream
triad | A[i] = B[i] + c * C[i] | | 2 load streams, 1 store stream
sch_triad | A[i] = B[i] + C[i] * D[i] | | 3 load streams, 1 store stream


Example Results for a Tesla V100-PCIe-16GB:
``` console
    blocks     threads     %occ  |               init       sum1       sum2       sum4       sum8      sum16        dot       tdot      scale      triad  sch_triad
        1          128     0.08  |  GB/s:        37.4        3.0        5.8       10.1       15.0       16.3        5.6        8.3        5.7        8.1       10.2
        2          256      0.2  |  GB/s:        74.7        6.0       11.3       19.8       29.6       32.4       11.1       15.9       11.5       15.9       20.0
        4          512      0.3  |  GB/s:       149.4       11.9       23.2       39.5       60.2       64.9       22.0       31.4       22.4       31.3       39.4
        8         1024      0.6  |  GB/s:       298.5       23.8       45.7       84.8      118.3      128.0       43.8       62.4       44.1       60.9       77.0
       16         2048      1.2  |  GB/s:       596.1       46.8       89.1      158.6      233.1      250.8       87.5      126.8       84.1      115.1      144.8
       32         4096      2.5  |  GB/s:       841.7       93.3      171.0      295.5      453.1      479.6      170.8      243.3      158.5      212.7      265.0
       64         8192      5.0  |  GB/s:       831.6      182.5      333.3      546.9      774.2      797.4      325.1      451.4      286.1      386.5      450.1
       80        10240      6.2  |  GB/s:       875.8      227.6      404.7      629.1      800.1      814.5      402.6      543.7      339.1      436.8      518.8
      160        20480     12.5  |  GB/s:       855.9      418.6      663.8      823.7      874.8      876.8      665.5      775.8      550.8      682.0      714.9
      240        30720     18.8  |  GB/s:       839.0      571.0      788.1      859.5      881.9      880.3      788.8      831.7      681.3      760.3      773.7
      320        40960     25.0  |  GB/s:       834.8      680.6      834.8      871.9      885.9      884.1      837.1      856.2      719.2      783.4      791.2
      400        51200     31.2  |  GB/s:       814.0      749.7      855.5      875.4      887.4      887.4      857.4      863.1      737.4      794.5      804.0
      480        61440     37.5  |  GB/s:       829.8      791.8      868.8      880.3      888.0      886.9      869.5      870.0      751.3      804.2      815.0
      560        71680     43.8  |  GB/s:       816.5      817.6      876.0      883.2      888.9      888.1      874.8      874.3      758.2      810.9      822.1
      640        81920     50.0  |  GB/s:       823.3      833.5      879.5      883.9      890.2      887.3      879.7      879.1      765.6      817.5      827.7
      720        92160     56.2  |  GB/s:       832.4      846.1      883.2      887.1      890.2      888.8      883.0      883.3      771.7      822.8      832.4
      800       102400     62.5  |  GB/s:       833.5      856.0      883.8      887.0      890.9      890.8      884.3      887.0      780.1      829.1      835.7
      880       112640     68.8  |  GB/s:       826.7      861.7      885.1      888.3      890.7      886.0      886.0      889.4      783.3      832.3      839.4
      960       122880     75.0  |  GB/s:       841.2      868.3      886.8      889.6      888.5      882.8      888.0      891.4      785.8      832.7      841.9
     1040       133120     81.2  |  GB/s:       829.8      873.7      887.3      890.2      887.8      887.3      888.5      892.5      788.3      830.7      843.2
     1120       143360     87.5  |  GB/s:       817.1      875.4      886.8      890.1      886.9      882.5      889.5      893.2      789.7      830.9      843.8
     1200       153600     93.8  |  GB/s:       825.2      877.8      889.3      890.7      883.8      884.0      890.8      893.3      788.9      829.4      844.2
     1280       163840    100.0  |  GB/s:       834.1      878.9      888.4      891.0      884.5      877.1      891.0      894.5      787.8      829.1      844.0
```

# cuda-latency

Pointer chasing benchmark for latency measurement. A single warp fully traverses a buffer in random order. A partitioning scheme is used to ensure that all cache lines are hit exactly once before they are accessed again. Latency in clock cycles is computed with the current clock rate.

Example results for a Tesla-V100-PCIe-16GB
``` console
  MHz       kB       ms   cycles
 1380        0      2.9    30.0
 1380        0      2.8    30.0
 1380        1      2.8    30.0
 1380        2      2.8    30.0
 1380        4      2.9    30.0
 1380        8      2.9    30.1
 1380       16      2.9    30.3
 1380       32      2.9    30.7
 1380       64      3.0    31.4
 1380      128      7.4    78.3
 1380      256     20.4   214.4
 1380      512     20.4   214.4
 1380     1024     20.4   214.4
 1380     2048     20.4   214.4
 1380     4096     20.4   214.4
 1380     8192     40.6   428.0
 1380    16384     81.3   427.9
 1380    32768    162.6   428.0
 1380    65536    328.5   432.3
 1380   131072    660.2   434.4
 1380   262144   1323.5   435.5
 1380   524288   2650.5   436.0
```


Both the L1 cache (128kB) and the L2 cache(6MB) are clearly visible
