#!/usr/bin/env python3

import matplotlib
import matplotlib.pyplot as plt

plt.style.use("bmh")

order = [
    "v100",
    "a100_40",
    "a100_80",
    "h100_pcie",
    "mi100",
    "mi210",
    "rx6900xt",
    "a40",
    "l40",
]


def getOrderNumber(f):
    for o in range(len(order)):
        if f.startswith(order[o]):
            return o
    return len(order) + 1


lineStyle = {"linewidth": 2, "alpha": 0.7, "markersize": 4, "marker": "P"}
