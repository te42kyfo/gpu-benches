#!/usr/bin/env python3

import matplotlib
import matplotlib.pyplot as plt

plt.style.use("bmh")

order = [
    "a40",
    "l40",
    "v100",
    "a100_40",
    "a100",
    "h100_pcie",
    "h200",
    "mi100",
    "mi210",
    "rx6900xt",
]


def getOrderNumber(f):
    for o in range(len(order)):
        if f.startswith(order[o]):
            return o
    return len(order) + 1


lineStyle = {"linewidth": 1.5, "alpha": 1, "markersize": 4, "marker": ""}
