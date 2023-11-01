#!/usr/bin/env python3

import os
import csv
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import math
from random import *

plt.style.use("ggplot")

fig, ax = plt.subplots(figsize=(10, 4))

color = 0

maxbars = {}
minbars = {}

order = ["mi210.txt", "v100.txt"]
peakBW = [897, 1555, 2039, 2039, 1229, 1638]


def getOrderNumber(f):
    for o in range(len(order)):
        if f.startswith(order[o]):
            return o
    return len(order) + 1


def getData(filename):
    with open(filename, newline="") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=" ", skipinitialspace=True)
        dims = []
        bw = []

        for row in csvreader:
            if row[0] == "blockSize":
                continue
            dims.append(float(row[0]))
            values = []
            for r in row[1:]:
                if len(r) == 0:
                    continue
                values.append(float(r))
            bw.append(values)

        return dims, bw


blockSizes = [
    (xblock, 1024 // xblock) for xblock in [4, 8, 16, 32, 64, 128, 256, 512, 1024]
]


def getColor(b):
    return tuple(min(1.0, math.log2(c) / math.log2(128) * 1.4) for c in b)


def getModelData(sizes, bandwidth, blockSize, spawnRate, clock, smCount):
    return sizes / (16000 / clock + (sizes * 16 / bandwidth))


color = 0
markers = ["-", "-", "-", "-", "-", "-"]
for filename in reversed(list(os.listdir("."))):
    if filename in order:
        dims, bw = getData(filename)
        dims = np.array(dims)
        sizes = dims * dims * 8
        if 1 == 2:
            for b in range(1, len(bw[0])):
                ax.plot(
                    sizes / 1024,
                    [v[b] if b < len(v) else 0 for v in bw],
                    "-",
                    label=str(blockSizes[b]) + " " + filename[:-4].upper(),
                    color=getColor((*blockSizes[b], 1)),
                    alpha=1.0,
                    linewidth=2,
                    zorder=-2 - randint(0, 5),
                )
        else:
            b = 2
            ax.plot(
                sizes / 1024,
                [
                    max([v[b] if b < len(v) else 0 for b in range(len(bw[0]))])
                    for v in bw
                ],
                markers[color],
                label=filename[:-4].upper() + " max",
                color=(0.1 + color * 0.25, 0.8 - color * 0.25, 0.7),
                linewidth=3,
            )
            color += 1


# ax.plot(
#    sizes / 1024,
#    getModelData(sizes / 8, 1.4e12, 128, 1, 1.48e9, 108) / 1e9,
#    "--",
#    label="model",
# )


def fitValues(xdata, ydata):
    print(xdata)
    print(ydata)

    from scipy.optimize import curve_fit

    # def func(x, a, b, c):
    #    return a * np.exp(-b * np.exp(-c * x))

    def func(x, a, b):
        return x / (a / 1.48e9 + (x * 16 / b))

    popt, pcov = curve_fit(
        func,
        xdata,
        ydata,
        bounds=([0, 0], [60000, 10000e9]),
    )
    print(popt)
    print(pcov)

    # xdata = np.array([*list(xdata), *[i / 25 for i in range(1, 25)]])
    # xdata.sort()

    plt.plot(
        xdata * 8 / 1024,
        func(xdata, *popt) / 1e9 * 16,
        "r-",
        label="fit: a=%5.3f, \\n     b=%5.3f GB/s," % (popt[0], popt[1] / 1e9),
    )


def fitCurve(splitA, splitB):
    fitValues(
        sizes[splitA:splitB] / 8,
        np.array(
            [
                max([v[b] / 16 if b < len(v) else 0 for b in range(len(bw[0]))])
                for v in bw
            ][splitA:splitB]
        )
        * 1e9,
    )


# rx6900
# fitCurve(0, 80)
# fitCurve(84, 110)
# fitCurve(110, 139)
# fitCurve(146, 240)

# mi210
# fitCurve(0, 76)
# fitCurve(98, 160)

# A100
# fitCurve(0, 102)
# fitCurve(117, 220)

# L100
# fitCurve(0, 128)
# fitCurve(146, 195)

# v100
fitCurve(0, 85)
# fitCurve(102, 250)

ax.set_xlabel("grid size, kB")
ax.set_ylabel("GLup/s")
ax.set_xscale("log")

ax.set_xscale("log")
ax.set_xticks([128, 256, 512, 1024, 2048, 8192, 20 * 1024, 64 * 1024])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

# ax.axhline(1400, linestyle="--", color="C1")
# ax.axhline(800, linestyle="--", color="C0")

# ax.grid()
ax.legend()
ax.set_ylim([0, ax.get_ylim()[1]])
# ax.set_xlim([32, 256 * 1024])

fig.tight_layout()
fig.savefig("repeated-stencil.svg", dpi=300)


plt.show()
