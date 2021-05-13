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
        1           32      0.2  |  GB/s:           9          1          1          3          3          2          1          2          1          2          3
        1           64      0.2  |  GB/s:          19          1          3          5          6          3          3          4          3          4          5
        1          128      0.2  |  GB/s:          37          3          6         11         11          6          6          8          6          8         10
        1          256      0.2  |  GB/s:          44          6         11         20         21         11         10         14         10         14         18
        1          256      0.2  |  GB/s:          44          6         11         20         21         11         10         14         10         14         18
        2          512      0.3  |  GB/s:          89         11         21         39         41         21         20         28         20         28         34
        4         1024      0.6  |  GB/s:         177         21         41         75         78         41         38         54         38         52         65
        8         2048      1.2  |  GB/s:         353         42         79        137        146         81         75        106         73         98        122
       16         4096      2.5  |  GB/s:         702         81        146        248        276        152        145        202        135        179        221
       32         8192      5.0  |  GB/s:         807        162        286        464        515        297        285        389        245        315        379
       64        16384     10.0  |  GB/s:         813        312        538        713        730        537        507        627        420        516        577
       80        20480     12.5  |  GB/s:         829        383        585        756        782        617        597        713        511        602        637
      160        40960     25.0  |  GB/s:         795        606        764        842        848        772        771        815        675        742        730
      240        61440     37.5  |  GB/s:         800        719        819        863        869        821        820        849        713        777        758
      320        81920     50.0  |  GB/s:         809        763        844        871        876        846        843        862        721        791        774
      400       102400     62.5  |  GB/s:         816        791        859        875        879        860        858        869        726        791        784
      480       122880     75.0  |  GB/s:         819        808        867        878        879        866        867        872        734        795        790
      560       143360     87.5  |  GB/s:         810        822        874        879        878        871        872        876        739        794        795
      640       163840    100.0  |  GB/s:         780        833        877        879        877        874        875        878        741        791        799
```


Results from a NVIDIA A100-SXM4-40GB / CUDA 1.3
``` console
    blocks     threads     %occ  |               init       sum1       sum2       sum4       sum8      sum16        dot       tdot      scale      triad  sch_triad
        1           32      0.1  |  GB/s:           7          2          3          6          6          3          1          2          2          2          2
        1           64      0.1  |  GB/s:          18          2          3          6          6          3          2          3          2          3          4
        1          128      0.1  |  GB/s:          37          2          5          9          9          4          4          6          4          6          8
        1          256      0.1  |  GB/s:          45          4          8         16         17          9          8         12          8         12         15
        1          512      0.1  |  GB/s:          45          8         16         29         31         17         15         22         16         22         28
        2          512      0.2  |  GB/s:          90          9         16         30         33         17         16         23         17         23         30
        4         1024      0.5  |  GB/s:         180         18         31         60         64         33         32         47         34         46         59
        8         2048      0.9  |  GB/s:         360         33         63        121        130         66         63         92         62         88        113
       16         4096      1.9  |  GB/s:         703         66        124        240        253        132        123        179        121        169        216
       32         8192      3.7  |  GB/s:        1348        133        245        457        497        263        243        350        235        322        407
       64        16384      7.4  |  GB/s:        1460        267        466        827        877        494        475        669        438        591        735
      108        27648     12.5  |  GB/s:        1485        440        759       1171       1228        805        764       1013        681        908       1062
      216        55296     25.0  |  GB/s:        1481        786       1186       1405       1424       1213       1188       1343       1079       1232       1301
      324        82944     37.5  |  GB/s:        1453       1039       1339       1411       1427       1351       1338       1422       1207       1302       1326
      432       110592     50.0  |  GB/s:        1419       1176       1399       1429       1443       1410       1397       1431       1259       1323       1339
      540       138240     62.5  |  GB/s:        1413       1260       1412       1452       1460       1418       1418       1437       1264       1331       1353
      648       165888     75.0  |  GB/s:        1422       1310       1421       1467       1473       1424       1424       1449       1302       1350       1371
      756       193536     87.5  |  GB/s:        1441       1347       1430       1476       1480       1435       1430       1458       1316       1359       1380
      864       221184    100.0  |  GB/s:        1436       1372       1441       1484       1482       1445       1437       1466       1328       1363       1384

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

# cuda-incore

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
 



