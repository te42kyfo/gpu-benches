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
        style = ".-"

        csvreader = csv.reader(csvfile, delimiter=" ", skipinitialspace=True)
        sizes = []
        bw = []
        for row in csvreader:
            if row[0] == "data":
                continue
            sizes.append(float(row[0]))
            bw.append(float(row[4]))

        print(sizes)
        print(bw)
        ax.plot(sizes, bw, style, label=filename[:-4].upper())

ax.set_xlabel("data volume per SM/CU, kB")
ax.set_ylabel("GB/s")
ax.set_xscale("log")

ax.set_xticks([16,  64, 256, 1024, 4096, 16384])
ax.set_xticklabels([16, 64, 256, 1024, 4096, 16384])

ax.axvline(16, color="black", alpha=0.7)
ax.axvline(192, color="black", alpha=0.7)


ax.grid()
ax.legend()
#ax.set_xticks([16, 64, 256])
#ax.set_yticks([5000, 20000])
ax.set_ylim([0, ax.get_ylim()[1]*1.1])

fig.tight_layout()
plt.show()
fig.savefig("cuda-cache.pdf")
