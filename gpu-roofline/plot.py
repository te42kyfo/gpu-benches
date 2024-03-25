#!/usr/bin/env python3


import os
import csv
import matplotlib.pyplot as plt
import numpy as np

import sys

sys.path.append("..")
from device_order import *


fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7, 7))
# fig2, ax2 = plt.subplots(figsize=(8, 4))
# fig3, ax3 = plt.subplots(figsize=(8, 4))


filename = "genoa_normal.txt"

colors = ["#349999", "#CC1343", "#649903", "#c7aa3e"]

with open(filename, newline="") as csvfile:
    csvreader = csv.reader(csvfile, delimiter=" ", skipinitialspace=True)

    datapoints = [[]]

    for row in csvreader:
        print(row)
        if len(row) == 0:
            datapoints.append([])

        elif len(row) == 16:
            datapoints[-1].append(
                [float(row[5]), float(row[9]), float(row[13]), float(row[11])]
            )

    print(datapoints)
    print()

    for i in range(len(datapoints[1])):
        print([d[i][1] for d in datapoints if len(d) > 0])
        ax1.plot(
            [d[i][0] for d in datapoints if len(d) > 0],
            [d[i][1] / 1000 for d in datapoints if len(d) > 0],
            "-",
            color=colors[i],
            label=list(["L40", "L40S"])[i],
        )

        ax2.plot(
            [d[i][0] for d in datapoints if len(d) > 0],
            [d[i][2] for d in datapoints if len(d) > 0],
            "--",
            color=colors[i],
        )

        ax3.plot(
            [d[i][0] for d in datapoints if len(d) > 0],
            [d[i][3] / 1000 for d in datapoints if len(d) > 0],
            "-.",
            color=colors[i],
        )


ax1.legend()

ax3.set_xlabel("Arithmetic Intensity, Flop/B")
ax1.set_ylabel("FP32, TFlop/s")
ax2.set_ylabel("Power, W")
ax3.set_ylabel("Clock, GHz")


ax1.set_ylim([0, ax1.get_ylim()[1]])
ax1.set_xlim([0, ax1.get_xlim()[1]])

ax2.set_ylim([0, ax2.get_ylim()[1]])
ax2.set_xlim([0, ax2.get_xlim()[1]])

ax3.set_ylim([0, ax3.get_ylim()[1]])
ax3.set_xlim([0, ax3.get_xlim()[1]])

# ax.set_xscale("log")
# ax2.set_xscale("log")

# ax.set_yscale("log")
# ax2.set_yscale("log")


fig.tight_layout()

plt.savefig("L40_plot.pdf", dpi=4000)
plt.show()
