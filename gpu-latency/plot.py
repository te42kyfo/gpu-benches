#!/usr/bin/env python3

import os
import csv
import matplotlib.pyplot as plt
import matplotlib

plt.style.use("ggplot")

order = ["v100", "a100_80", "h100_pcie", "mi100", "mi210", "rx6900xt", "l40"]


def getOrderNumber(f):
    for o in range(len(order)):
        if f.startswith(order[o]):
            return o
    return len(order) + 1


fig, ax = plt.subplots(figsize=(8, 4))
for filename in sorted(os.listdir("."), key=lambda f1: getOrderNumber(f1)):
    if not filename.endswith(".txt") or filename[:-4] not in order:
        continue
    with open(filename, newline="") as csvfile:
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
        ax.plot(
            sizes,
            bw,
            "P-",
            label=filename[:-4].upper(),
            linewidth=3,
            markeredgewidth=0,
            alpha=0.7,
            markersize=5,
            color="C" + str(getOrderNumber(filename)),
        )

ax.set_xlabel("chain data volume, kB")
ax.set_ylabel("latency, cycles")
ax.set_xscale("log", base=2)


# ax.axvline(16)
# ax.axvline(4*1024)

formatter = matplotlib.ticker.FuncFormatter(
    lambda x, pos: "{0:g} kB".format(x) if x < 1024 else "{0:g} MB".format(x // 1024)
)
ax.get_xaxis().set_major_formatter(formatter)
# ax.get_yaxis().set_major_formatter(formatter)

ax.set_xticks([16, 128, 256, 6 * 1024, 20 * 1024, 40 * 1024, 128 * 1024])

ax.set_ylim([0, 700])

fig.autofmt_xdate()
ax.legend()
ax.set_ylim([0, ax.get_ylim()[1]])
fig.tight_layout()
fig.savefig("latency_plot.svg")

plt.show()
