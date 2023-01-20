#!/usr/bin/env python3

import os
import csv
import matplotlib.pyplot as plt
import matplotlib

fig,ax = plt.subplots(figsize=(9,4))
for filename in os.listdir("."):
    if not filename.endswith(".txt"):
        continue
    with open(filename, newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=" ", skipinitialspace=True)
        sizes = []
        bw = []
        for row in csvreader:
            if len(row) == 0 or row[0] == "clock:":
                continue
            sizes.append(float(row[2]))
            bw.append(float(row[4]))
        print(filename)
        print(sizes)
        print(bw)
        ax.plot(sizes, bw, "-o", label=filename, linewidth=0.4, markeredgewidth=0, alpha=0.66, markersize=3)

ax.set_xlabel("chain data volume, kB")
ax.set_ylabel("latency, cycles")
ax.set_xscale("log", base=2)

ax.set_xticks(
    [64, 128, 192, 256, 1024, 6*1024, 20 * 1024, 40 * 1024]
)

ax.axvline(16)
ax.axvline(4*1024)

formatter = matplotlib.ticker.FuncFormatter(lambda x, pos: "{0:g}".format(x))
ax.get_xaxis().set_major_formatter(formatter)
ax.get_yaxis().set_major_formatter(formatter)

ax.legend()
ax.set_ylim([0, ax.get_ylim()[1]])

fig.savefig("cache_plot.svg")

plt.show()
