#!/usr/bin/env python3

import os
import csv

import sys

sys.path.append(".")
sys.path.append("..")
from device_order import *


fig, ax = plt.subplots(figsize=(8, 4))
fig2, ax2 = plt.subplots(figsize=(8, 4))


for filename in sorted(os.listdir("."), key=lambda f1: getOrderNumber(f1)):
    if not filename.endswith(".txt"):
        continue

    with open(filename, newline="") as csvfile:

        csvreader = csv.reader(csvfile, delimiter=" ", skipinitialspace=True)
        sizes = []
        bw = []
        L2bw = []
        for row in csvreader:
            if row[0] == "data" or not row[0].isnumeric():
                continue
            sizes.append(float(row[2]))
            bw.append(float(row[6]))
            L2bw.append(float(row[12]))

        # print(sizes)
        # print(bw)
        ax.plot(
            sizes,
            bw,
            label=filename[:-4].upper(),
            color="C" + str(getOrderNumber(filename)),
            **lineStyle
        )
        ax2.plot(
            sizes,
            L2bw,
            label=filename[:-4].upper(),
            color="C" + str(getOrderNumber(filename)),
            **lineStyle
        )
        print(filename, getOrderNumber(filename))


ax.set_xlabel("total data volume, MB")
ax.set_ylabel("GB/s")
ax.set_xscale("log", base=2)


ax2.set_xlabel("total data volume, kB")
ax2.set_ylabel("GB/s")
ax2.set_xscale("log", base=2)

formatter = matplotlib.ticker.FuncFormatter(lambda x, pos: "{0:g}".format(x / 1024))
ax.get_xaxis().set_major_formatter(formatter)
# ax.get_yaxis().set_major_formatter(formatter)

ax2.get_xaxis().set_major_formatter(formatter)
ax2.get_yaxis().set_major_formatter(formatter)

ax.set_xticks([1024, 4 * 1024, 8 * 1024, 20 * 1024, 40 * 1024, 128 * 1024, 1024 * 1024])

ax2.set_xticks([1024, 6 * 1024, 20 * 1024, 40 * 1024, 128 * 1024])

fig.autofmt_xdate()
ax.set_ylim([0, ax.get_ylim()[1]])
ax.set_xlim([1024 * 1.5, 1024 * 1024])

fig2.autofmt_xdate()
ax2.set_xlim([1024 * 1.5, 1024 * 1024])

ax.set_xlim([1024, ax.get_xlim()[1]])
ax.legend()
fig.tight_layout()

ax2.legend()
fig2.tight_layout()

plt.show()
fig.savefig("cuda-cache.svg")
# fig2.savefig("cuda-cache-l2.svg")
