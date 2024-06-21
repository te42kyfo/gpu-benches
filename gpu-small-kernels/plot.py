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


filesToInclude = ["L40", "A100", "RX6900XT", "MI210", "H200"]

# filesToInclude = ["L40", "RX6900XT"]


def getIncludeNumber(filename):
    for i in range(len(filesToInclude)):
        if filename.upper().startswith(filesToInclude[i]):
            return i
    return len(filesToInclude) + 1


def fitValues(xdata, ydata, color=None):
    ydata[2:-2] = (
        ydata[0:-4] + ydata[1:-3] + ydata[2:-2] + ydata[3:-1] + ydata[4:]
    ) / 5

    from scipy.optimize import curve_fit

    # def func(x, a, b, c):
    #    return a * np.exp(-b * np.exp(-c * x))

    def func(x, a, b):
        return x / (a / 1e9 + (x / 1e9 / b))

    best = 0
    lim = 1
    bestLim = lim
    perr = -1

    while lim + 1 < len(xdata):
        lim += 1
        if xdata[lim] < 3 * 1024 * 1024 or xdata[lim] > 100 * 1024 * 1024:
            continue

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
        perr = np.diag(pcov)[0] * np.diag(pcov)[1]
        if perr < best or best == 0:
            best = perr
            bestLim = lim

        print("%d fit: a=%5.0f ns,   b=%5.0f GB/s," % (lim, popt[0], popt[1]))
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
        xdata[:lim] / 1024,
        func(xdata[:lim], *popt) / 1e9,
        "-",
        color="black",  # icolor,
        label="fit: a=%5.0f ns, b=%5.0f GB/s," % (popt[0], popt[1]),
        zorder=-1,
        linewidth=2,
        alpha=1.0,
    )
    return perr


def fitCurve(splitA, splitB, color=None):
    fitValues(
        sizes[splitA:splitB],
        np.array(
            [max([v[b] if b < len(v) else 0 for b in range(len(bw[0]))]) for v in bw][
                splitA:splitB
            ]
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
            for r in row[2:]:
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


for filename in sorted(sorted(os.listdir(".")), key=lambda f1: getOrderNumber(f1)):
    if (
        any([filename.upper().startswith(f) for f in filesToInclude])
        and not "linear" in filename
        and not "graph" in filename
        and not "pt" in filename
        and not "gsync" in filename
    ):
        dims, bw = getData(filename)
        if len(bw) < 3:
            continue

        dims = np.array(dims)
        sizes = dims * 16

        lineStyle["marker"] = None  # "|" if "graph" in filename.lower() else "_"
        lineStyle["linewidth"] = 2
        lineStyle["linestyle"] = (
            "-."
            if "gsync" in filename.lower()
            else (
                ":"
                if "pt" in filename.lower()
                else "--" if "graph" in filename.lower() else "-"
            )
        )
        b = 2
        ax.plot(
            sizes / 1024,
            [max([v[b] if b < len(v) else 0 for b in range(len(bw[0]))]) for v in bw],
            label=filename[:-4].upper(),
            color="C" + str(getOrderNumber(filename)),
            **lineStyle,
            zorder=0
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


def func(x, a, b):
    return x / (a / 1e9 + (x * 16 / 1e9 / b))


# values = np.arange(256, 32 * 1024, 256)
# ax.plot(
#    values,
#    func(values * 1024 / 8, 3000, 2100) * 1e-9 * 16,
#    color="red",
#    linewidth=3,
#    label="MI300X, \n fit: a = 15000 GB/s, \n      b = 3000 ns",
# )

# values = np.arange(32 * 1024, 1024 * 1024, 256)
# ax.plot(
#    values, func(values * 1024 / 8, 3000, 500) * 1e-9 * 16, color="red", linewidth=3
# )


ax.set_xlabel("grid size, kB")
ax.set_ylabel("GB/s")
ax.set_xscale("log")

ax.set_xscale("log")
ax.set_xticks([128, 256, 512, 1024, 2048, 8192, 20 * 1024, 64 * 1024])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

# ax.axhline(1400, linestyle="--", color="C1")
# ax.axhline(800, linestyle="--", color="C0")

# ax.grid()
ax.legend()
ax.set_ylim([0, ax.get_ylim()[1]])
ax.set_xlim([64, 512 * 1024])

fig.tight_layout()
fig.savefig("repeated-stream.svg", dpi=300)


plt.show()
