#!/usr/bin/env python3

import os
import csv
import matplotlib.pyplot as plt
import numpy as np

import sys

sys.path.append("..")
from device_order import *


fig, ax = plt.subplots(figsize=(8, 4))
# fig2, ax2 = plt.subplots(figsize=(8, 4))
# fig3, ax3 = plt.subplots(figsize=(8, 4))


maxbars = {}
minbars = {}


peakBW = [897, 1555, 2039, 2039, 1229, 1638, 690, 690]


for filename in sorted(os.listdir("."), key=lambda f1: getOrderNumber(f1)):
    if not filename.endswith(".txt"):
        continue
    with open(filename, newline="") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=" ", skipinitialspace=True)
        threads = []
        locs = []
        init = []
        read = []
        scale = []
        triad = []
        stencil3pt = []
        stencil5pt = []

        for row in csvreader:
            if row[0] == "blockSize":
                continue
            threads.append(int(row[1]))
            init.append(float(row[6]))
            read.append(float(row[7]))
            scale.append(float(row[8]))
            triad.append(float(row[9]))
            locs.append(float(row[2]))
            stencil3pt.append(float(row[10]))
            stencil5pt.append(float(row[11]))

        # locs = threads#[15 + l / 6 if l > 15 else l for l in locs]
        # print(locs)
        # print(threads)
        # ax.plot(locs, init,  "-v", label=filename, color="C" + str(color))
        ax.plot(
            np.array(threads) * 2,
            scale,
            label=filename[:-4].upper(),
            color="C" + str(getOrderNumber(filename)),
            **lineStyle
        )
        print(filename, getOrderNumber(filename))
        # ax.plot(threads, triad, "-<", label=filename, color="C" + str(color))
        # ax.plot(threads, read, "-^", label=filename, color="C" + str(color))

        maxbars[filename] = [
            init[-1],
            read[-1],
            scale[-1],
            triad[-1],
            stencil3pt[-1],
            stencil5pt[-1],
        ]
        minbars[filename] = [
            init[0],
            read[0],
            scale[0],
            triad[0],
            stencil3pt[0],
            stencil5pt[0],
        ]

########ax.set_xticks(threads[::5])
# ax.set_xticklabels(threads, rotation="vertical")
ax.set_xlabel("threads")
ax.set_ylabel("GB/s")

# ax.axhline(1400, linestyle="--", color="C1")
# ax.axhline(800, linestyle="--", color="C0")

# ax.grid()
ax.legend()
ax.set_ylim([0, ax.get_ylim()[1]])
ax.set_xlim([0, ax.get_xlim()[1]])

fig.tight_layout()
fig.savefig("cuda-stream.svg", dpi=300)

plt.show()

print(maxbars)

fig2, ax2 = plt.subplots(figsize=(8, 4))

valueCount = len(list(maxbars.values())[0])
for m in range(valueCount):
    ax2.bar(
        np.arange(len(maxbars)) + 0.8 * m / valueCount - 0.4,
        [i[m] for i in maxbars.values()],
        width=0.8 / valueCount,
    )

ax2.text(-0.4, 51, "init", rotation=90, color="w")
ax2.text(-0.28, 51, "read", rotation=90, color="w")
ax2.text(-0.16, 51, "scale", rotation=90, color="w")
ax2.text(-0.04, 51, "triad", rotation=90, color="w")
ax2.text(0.08, 51, "1D3PT", rotation=90, color="w")
ax2.text(0.22, 51, "1D5pt", rotation=90, color="w")

print(list(maxbars.keys()))
ax2.set_xticks(range(len(list(maxbars.keys()))))
ax2.set_xticklabels(list(maxbars.keys()))

plt.show()
