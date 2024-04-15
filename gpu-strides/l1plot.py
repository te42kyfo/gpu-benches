#!/usr/bin/env python3

import os
import csv
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import SymLogNorm, LogNorm

from matplotlib.font_manager import FontProperties
from matplotlib.ticker import LogFormatter

import sys

sys.path.append("..")
from device_order import *


def heatmap(
    data, row_labels, col_labels, ax=None, cbar_kw=None, cbarlabel="", **kwargs
):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(np.maximum(data, 1.0), **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels, style="italic")

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="center", rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=4, clip_on=False)
    ax.tick_params(which="minor", bottom=False, left=False, top=False)

    # ax.set_yticks(np.arange(3, data.shape[0], 3) - 0.5, minor=False)
    # ax.set_xticks([], minor=False)
    # ax.grid(which="major", color="w", linestyle="-", linewidth=0)
    ax.tick_params(which="major", bottom=False, left=False, top=False)

    return im, cbar


def annotate_heatmap(
    im,
    data=None,
    valfmt="{x:.2f}",
    textcolors=("black", "white"),
    threshold=None,
    preds=None,
    **textkw
):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

        # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.0

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[1 if im.norm(data[i, j]) > threshold else 0])
            text = im.axes.text(
                j,
                i - 0.17,
                valfmt(data[i, j], None) if data[i, j] > 0.0 else " ",
                **kw,
                fontweight="bold",
            )

            texts.append(text)

            im.axes.text(
                j,
                i + 0.23,
                valfmt(preds[i, j], None) if data[i, j] > 0.0 else " ",
                **kw,
                fontsize=7,
            )
    return texts


filesToInclude = ["A40", "A100", "MI210", "RDNA"]
filenames = []

for filename in sorted(sorted(os.listdir(".")), key=lambda f1: getOrderNumber(f1)):
    if not filename.endswith(".txt") or not any(
        [filename.upper().startswith(f) for f in filesToInclude]
    ):
        continue
    with open(filename, newline="") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=" ", skipinitialspace=True)

        for row in csvreader:
            if len(row) < 4:
                continue
            if not row[0] in ["double", "float"] or not row[1] in ["stride", "block"]:
                continue
            filenames.append(filename)
            break

file_count = len(filenames)


file_id = 0


gpuData = dict()
for filename in filenames + [
    "pred_" + f for f in filenames if "pred_" + f in os.listdir(".")
]:

    data = {
        ("float", "stride"): dict(),
        ("double", "stride"): dict(),
        ("float", "block"): dict(),
        ("double", "block"): dict(),
    }

    with open(filename, newline="") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=" ", skipinitialspace=True)

        for row in csvreader:
            if len(row) < 4:
                continue
            if not row[0] in ["double", "float"] or not row[1] in ["stride", "block"]:
                continue
            data[(row[0], row[1])][int(row[2])] = row[3:]

    if len(data[("float", "block")]) == 0:
        continue

    gpuData[filename] = data


def getData(data):
    cycles = [float(d[1]) for d in data]
    data_cycles = [float(d[3]) for d in data]
    tag_cycles = [float(d[4]) for d in data]
    return cycles, data_cycles, tag_cycles


#####  -----------
#####  Tableau of the block throughputs with all GPUs


imdata = None
impred = None
names = []
rowLabels = []
rowCounts = []

strideList = [1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17]


from operator import itemgetter

datatype = "double"
for filename in filenames:
    cycles, data_cycles, tag_cycles = getData(
        [gpuData[filename][(datatype, "stride")][0]]
        + list(gpuData[filename][(datatype, "block")].values())
        + list(
            itemgetter(*strideList)(
                list(gpuData[filename][(datatype, "stride")].values())
            )
        )
    )

    if "pred_" + filename in gpuData:
        print(filename)
        predname = "pred_" + filename

        preds_cycles, preds_data, preds_tag = getData(
            [gpuData[predname][(datatype, "stride")][0]]
            + list(gpuData[predname][(datatype, "block")].values())
            + list(
                itemgetter(*strideList)(
                    list(gpuData[predname][(datatype, "stride")].values())
                )
            )
        )
    else:
        preds_cycles, preds_data, preds_tag = (cycles, data_cycles, tag_cycles)

    print(preds_cycles)

    if imdata is None:
        imdata = np.array(cycles)
    else:
        imdata = np.vstack([imdata, cycles])

    if impred is None:
        impred = np.array(preds_cycles)
    else:
        impred = np.vstack([impred, preds_cycles])

    rowLabels.append("total")
    rowCounts.append(1)

    if data_cycles[0] != 0:
        imdata = np.vstack([imdata, data_cycles])
        impred = np.vstack([impred, preds_data])
        rowLabels.append("data")
        rowCounts[-1] += 1

    if tag_cycles[0] != 0:
        imdata = np.vstack([imdata, tag_cycles])
        impred = np.vstack([impred, preds_tag])
        rowLabels.append("tag")
        rowCounts[-1] += 1

    imdata = np.vstack([imdata, [0] * len(cycles)])
    impred = np.vstack([impred, [0] * len(cycles)])
    rowLabels.append(" ")
    rowCounts[-1] += 1

    names.extend([filename[:-4].upper()])
    file_id += 1

imdata = imdata[:-1, :]
rowLabels.pop()

fig, ax = plt.subplots(figsize=(8.3, 5))

font = FontProperties()
font.set_family("sans")
font.set_name("calibri")
font.set_style("normal")
font.set_weight("bold")
pos = 0
for n, c in zip(names, rowCounts):
    ax.text(
        -2.1,
        pos + (c - 1) / 2 - 0.6,
        n,
        rotation=90,
        fontsize=16,
        verticalalignment="center",
        horizontalalignment="center",
        fontproperties=font,
    )
    pos += c

pos = 0
for c in rowCounts[:-1]:
    pos += c
    ax.axhline(pos - 1, -0.1, 1.0, color="w", linewidth=30, clip_on=False)
    ax.axhline(pos - 1, -0.07, 1.0, color="gray", linewidth=1, clip_on=False)


ax.grid(None)
im, cbar = heatmap(
    imdata,
    rowLabels,
    ["U", "B1", "B2", "B4", "B8", "B16", "B32", "B64"]
    + ["S" + str(s) for s in strideList],
    ax=ax,
    cmap="Blues",
    cbarlabel="cycles / 32 threads",
    aspect=1.0,
    norm=LogNorm(vmin=1.0, vmax=32),
    cbar_kw={
        "ticks": [1, 2, 4, 8, 16],
        "format": LogFormatter(2, labelOnlyBase=False),
    },
)

cbar.remove()
ax.set_xlabel("cycles / 32 threads")
ax.text(-2.5, -0.8, datatype, fontsize=16, fontweight="bold")


print(impred)

texts = annotate_heatmap(im, imdata, valfmt="{x:.2g}", preds=impred)


def MAPE(data1, data2):
    return sum([abs(d1 - d2) / d2 for d1, d2 in zip(data1, data2)]) / len(data1)


fig.tight_layout()
fig.savefig("L1plot_" + datatype + ".pdf", dpi=300)


plt.show()


##########################
#####  Tableau of strides for each GPU
########------------------


for filename in []:

    cycles, data_cycles, tag_cycles = getData(
        list(gpuData[filename][("float", "stride")].values())
    )
    cycles = cycles[1:]

    print(cycles)
    print(np.array(cycles))
    imdata = np.array(cycles)
    imdata.resize((2, 32))

    print(imdata)

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.grid(None)
    im, cbar = heatmap(
        imdata,
        [""] * imdata.shape[0],
        [""] * imdata.shape[1],
        ax=ax,
        cmap="Blues",
        cbarlabel="cycles / 32 threads",
        aspect=1.0,
        norm=LogNorm(vmin=1.0, vmax=20),
        cbar_kw={
            "ticks": [1, 2, 4, 8, 16],
            "format": LogFormatter(2, labelOnlyBase=False),
        },
    )

    cbar.remove()

    texts = annotate_heatmap(im, imdata, valfmt="{x:.2g}")

    fig.tight_layout()
    fig.savefig("strides_" + filename + ".pdf", dpi=300)


def MAPE(data1, data2):
    return sum([abs(d1 - d2) / d2 for d1, d2 in zip(data1, data2)]) / len(data1)


plt.show()
