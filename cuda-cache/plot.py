#!/usr/bin/env python3

import os
import csv
import matplotlib.pyplot as plt

fig,ax = plt.subplots(figsize=(7,4))
for filename in os.listdir("."):
    if not filename.endswith(".txt"):
        continue
    with open(filename, newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=" ", skipinitialspace=True)
        sizes = []
        bw = []
        for row in csvreader:
            if row[0] == "data":
                continue
            sizes.append(float(row[0]))
            bw.append(float(row[4]))

        print(sizes)
        print(bw)
        ax.plot(sizes, bw, "-x", label=filename)

ax.set_xlabel("dataset per thread block, kB")
ax.set_ylabel("GB/s")
ax.set_xscale('log')
ax.legend()
ax.set_ylim([0, ax.get_ylim()[1]])
fig.savefig("cache_plot.png")
