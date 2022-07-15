#!/usr/bin/env python3

import os
import csv
import matplotlib.pyplot as plt
import matplotlib

fig,ax = plt.subplots(figsize=(6,3.5))

for filename in sorted( os.listdir(".") ):
    if not filename.endswith(".txt"):
        continue

    with open(filename, newline='') as csvfile:
        if filename[0] == 'a':
            color = "C1"
        else:
            color = "C0"
        if filename[5] == 's':
            style = "-o"
        else:
            style = "-^"

        csvreader = csv.reader(csvfile, delimiter=" ", skipinitialspace=True)
        sizes = []
        bw = []
        for row in csvreader:
            if row[0] == "data":
                continue
            sizes.append(float(row[2]))
            bw.append(float(row[6]))

        print(sizes)
        print(bw)
        ax.plot(sizes, bw, style, color=color, label=filename[:-4].upper())

ax.set_xlabel("data volume, MB")
ax.set_ylabel("GB/s")

ax.set_xscale("log")


#ax.axvline(20*1024, color="C1")
#ax.axvline(40*1024, color="C1")
#ax.axvline(6*1024, color="C0")

ax.grid()
ax.legend()
ax.set_xticks([6*1024, 20*1024, 40*1024])
ax.set_xticklabels([6, 20, 40])
ax.set_yticks([800, 2500, 1400, 5000])
ax.set_ylim([0, ax.get_ylim()[1]*1.1])

fig.tight_layout()
plt.show()
fig.savefig("cuda-cache.pdf")
