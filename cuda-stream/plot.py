#!/usr/bin/env python3

import os
import csv
import matplotlib.pyplot as plt


fig,ax = plt.subplots(figsize=(9,4))

color = 0

for filename in os.listdir("."):
    if not filename.endswith(".txt"):
        continue
    with open(filename, newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=" ", skipinitialspace=True)
        threads = []
        locs = []
        init = []
        read = []
        scale = []
        triad = []

        for row in csvreader:
            if row[0] == "blockSize":
                continue
            threads.append(int(row[1]))
            init.append(float(row[6]))
            read.append(float(row[7]))
            scale.append(float(row[8]))
            triad.append(float(row[9]))
            locs.append(float(row[2]))

        #locs = threads#[15 + l / 6 if l > 15 else l for l in locs]
        print(locs)
        print(threads)
        #ax.plot(locs, init,  "-v", label=filename, color="C" + str(color))
        ax.plot(threads, scale, "-o", label=filename[:-4].upper(), color="C" + str(color))
        #ax.plot(threads, triad, "-<", label=filename, color="C" + str(color))
        #ax.plot(threads, read, "-^", label=filename, color="C" + str(color))

    color += 1

ax.set_xticks(threads[::5])
#ax.set_xticklabels(threads, rotation="vertical")
ax.set_xlabel("threads")
ax.set_ylabel("GB/s")

#ax.axhline(1400, linestyle="--", color="C1")
#ax.axhline(800, linestyle="--", color="C0")

ax.grid()
fig.tight_layout()
ax.legend()
ax.set_ylim([0, ax.get_ylim()[1]])
plt.show()
fig.savefig(filename + ".svg")
