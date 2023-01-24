#!/usr/bin/env python3

import os
import csv
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('ggplot')

fig,ax = plt.subplots(figsize=(8,4))
fig2, ax2 =   plt.subplots(figsize=(8,4))
fig3, ax3 =   plt.subplots(figsize=(8,4))


color = 0

maxbars = {}
minbars = {}

order = ["v100", "a100_40", "a100_80", "h100", "mi100", "mi210"]
def getOrderNumber(f):
    for o in range(len(order)):
        if f.startswith(order[o]):
            return o
    return len(order)+1
for filename in sorted( os.listdir("."), key= lambda f1 : getOrderNumber(f1) ):
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

        #locs = threads#[15 + l / 6 if l > 15 else l for l in locs]
        print(locs)
        print(threads)
        #ax.plot(locs, init,  "-v", label=filename, color="C" + str(color))
        ax.plot(np.array(threads)*2, scale, ".-", label=filename[:-4].upper(), color="C" + str(color), alpha=0.8, linewidth=3)
        #ax.plot(threads, triad, "-<", label=filename, color="C" + str(color))
        #ax.plot(threads, read, "-^", label=filename, color="C" + str(color))

        maxbars[filename] =  [init[-1], read[-1], scale[-1], triad[-1], stencil3pt[-1], stencil5pt[-1]]
        minbars[filename] =  [init[0], read[0], scale[0], triad[0], stencil3pt[0], stencil5pt[0]]
    color += 1

########ax.set_xticks(threads[::5])
#ax.set_xticklabels(threads, rotation="vertical")
ax.set_xlabel("threads")
ax.set_ylabel("GB/s")

#ax.axhline(1400, linestyle="--", color="C1")
#ax.axhline(800, linestyle="--", color="C0")

#ax.grid()
ax.legend()
ax.set_ylim([0, ax.get_ylim()[1]])
ax.set_xlim([0, ax.get_xlim()[1]])

fig.tight_layout()
fig.savefig("cuda-stream.svg", dpi=300)


def plotBars(ax, bars):
    ax.bar(np.arange(0,6)-0.30, [bars[d][0] for d in bars], width=0.12, label="init")
    ax.bar(np.arange(0,6)-0.18, [bars[d][1] for d in bars], width=0.12, label="read")
    ax.bar(np.arange(0,6)-0.06, [bars[d][2] for d in bars], width=0.12, label="scale")
    ax.bar(np.arange(0,6)+0.06, [bars[d][3] for d in bars], width=0.12, label="triad")
    ax.bar(np.arange(0,6)+0.18, [bars[d][4] for d in bars], width=0.12, label="3pt")
    ax.bar(np.arange(0,6)+0.30, [bars[d][5] for d in bars], width=0.12, label="5pt")
    ax.legend()
    ax.set_xticks(np.arange(0,6), [d[:-4].upper() for d in maxbars])
    ax.set_ylabel("GB/s")

plotBars(ax2, maxbars)
fig2.tight_layout()
fig2.savefig("cuda-stream-maxbars.svg", dpi=300)

plotBars(ax3, minbars)
fig3.tight_layout()
fig3.savefig("cuda-stream-minbars.svg", dpi=300)

plt.show()
