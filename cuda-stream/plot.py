#!/usr/bin/env python3

import os
import csv
import matplotlib.pyplot as plt

for filename in os.listdir("."):
    if not filename.endswith(".txt"):
        continue
    with open(filename, newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=" ", skipinitialspace=True)
        threads = []
        locs = []
        init = []
        sum1 = []
        sum4 = []
        scale = []
        triad = []

        for row in csvreader:
            if row[0] == "blocks":
                continue
            threads.append(row[2])
            init.append(float(row[5]))
            sum1.append(float(row[6]))
            sum4.append(float(row[8]))
            scale.append(float(row[13]))
            triad.append(float(row[14]))
            locs.append(float(row[2]))

        locs = [15 + l / 6 if l > 15 else l for l in locs]
        print(locs)
        print(threads)
        fig,ax = plt.subplots(figsize=(9,4))
        ax.plot(locs, init, "-x", label="init")
        ax.plot(locs, sum1, "-x", label="sum1")
        ax.plot(locs, sum4, "-x", label="sum4")
        ax.plot(locs, scale, "-x", label="scale")
        ax.plot(locs, triad, "-x", label="triad")
        ax.set_xticks(locs)
        ax.set_xticklabels(threads, rotation="vertical")
        ax.set_xlabel("Occupancy, %")
        ax.set_ylabel("GB/s")



        ax.legend()
        ax.set_ylim([0, ax.get_ylim()[1]])
        fig.savefig(filename + ".svg")
