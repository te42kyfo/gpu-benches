#!/usr/bin/env python3

import os
import csv
import numpy as np
import math
from random import *


import sys

sys.path.append("..")
from device_order import *

fig, ax = plt.subplots(figsize=(10, 6))


maxbars = {}
minbars = {}

peakBW = [897, 1555, 2039, 2039, 1229, 1638]


filesToInclude = ["L40", "A100", "RX6900XT", "MI210"]


def fitValues(xdata, ydata, color=None):
    from scipy.optimize import curve_fit

    # def func(x, a, b, c):
    #    return a * np.exp(-b * np.exp(-c * x))

    def func(x, a, b):
        return x / (a / 1e9 + (x * 16 / 1e9 / b))

    best = 0
    lim = 50
    bestLim = lim
    perr = -1

    while lim < len(xdata):
        lim += 1
        popt, pcov, infodict, mesg, ier = curve_fit(
            func,
            xdata[:lim],
            ydata[:lim],
            bounds=([0, 0], [np.inf, np.inf]),
            full_output=True,
        )
        # print(popt)
        # print(pcov)
        # print(mesg)
        perr = np.sqrt(np.diag(pcov))[1] + np.sqrt(np.diag(pcov))[0]
        if perr / lim / lim < best or best == 0:
            best = perr / lim / lim
            bestLim = lim

        print("fit: a=%5.0f ns,   b=%5.0f GB/s," % (popt[0], popt[1]))
    print()
    # print(perr)

    lim = bestLim
    popt, pcov, infodict, mesg, ier = curve_fit(
        func,
        xdata[:lim],
        ydata[:lim],
        bounds=([0, 0], [np.inf, np.inf]),
        full_output=True,
    )
    print(lim, best)

    # xdata = np.array([*list(xdata), *[i / 25 for i in range(1, 25)]])
    # xdata.sort()

    plt.plot(
        xdata[:lim] * 8 / 1024,
        func(xdata[:lim], *popt) / 1e9 * 16,
        "-",
        color="black",  # icolor,
        label="fit: a=%5.0f ns, \n     b=%5.0f GB/s," % (popt[0], popt[1]),
        zorder=-1,
    )
    return perr


def fitCurve(splitA, splitB, color=None):
    fitValues(
        sizes[splitA:splitB] / 8,
        np.array(
            [
                max([v[b] / 16 if b < len(v) else 0 for b in range(len(bw[0]))])
                for v in bw
            ][splitA:splitB]
        )
        * 1e9,
        color,
    )


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


for filename in sorted(os.listdir("."), key=lambda f1: getOrderNumber(f1)):
    if any([filename.upper().startswith(f) for f in filesToInclude]):
        dims, bw = getData(filename)
        dims = np.array(dims)
        sizes = dims * dims * 16
        lineStyle["marker"] = None  # "|" if "graph" in filename.lower() else "_"
        lineStyle["linestyle"] = "--" if "graph" in filename.lower() else "-"
        lineStyle["alpha"] = 1

        b = 2
        ax.plot(
            sizes / 1024,
            [max([v[b] if b < len(v) else 0 for b in range(len(bw[0]))]) for v in bw],
            label=filename[:-4].upper(),
            color="C" + str(getOrderNumber(filename)),
            **lineStyle,
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
        fitCurve(2, 120)

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
