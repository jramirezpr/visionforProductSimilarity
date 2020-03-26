# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 17:22:03 2019

@author: cenic
"""

import matplotlib.pyplot as plt

segments = {1: [(0, 2.50),
                (2.9, 5.4),
                (6.5, 10)],
            2: [(0, 1.2),
                (2.3, 4.3),
                (4.6, 10)],
            3: [(0, 1.4),
                (2.9, 4),
                (4.4, 6.4),
                (6.8, 10)]}

colors = {1: 'b', 2: 'g'}
fig, (ax1, ax2) = plt.subplots(2, 1)  # a figure with a 2x1 grid of Axes
redvertcolor = [0, 1, 3, 4, 5, 7, 8, 9, 10]
for y in segments:
    col = colors.get(y, 'k')
    for seg in segments[y]:
        ax1.plot(seg, [y, y], color=col)
        ax1.axes.xaxis.set_ticklabels(["10:00",
                                       "10:30",
                                       "11:00",
                                       "11:30",
                                       "12:00" ,
                                       "12:30",
                                       "13:00"])
        ax1.axes.yaxis.set_ticks([1, 2, 3])
for xval in range(0, 11):
    if xval in redvertcolor:
        ax1.axvline(x=xval, color='r', linestyle='--')
    else:
        ax1.axvline(x=xval, linestyle='--')
ax1.set_title("cajas muy utilizadas")
ax1.set_xlabel("hora")
ax1.set_ylabel("número de caja")

segments = {1: [
                (5, 8)],
            2: [
                (1.0, 2.3),
                (3.8, 6.2),
                ],
            3: [(0, 1.4),
                (2.0, 3.0),
                (4.4, 5.4),
                (6.8, 10)]}

colors = {1: 'b', 2: 'g'}
redvertcolor = [5]
for y in segments:
    col = colors.get(y, 'k')
    for seg in segments[y]:
        ax2.plot(seg, [y, y], color=col)
        ax2.axes.xaxis.set_ticklabels(["10:00",
                                       "10:30",
                                       "11:00",
                                       "11:30",
                                       "12:00" ,
                                       "12:30",
                                       "13:00"])
        ax2.axes.yaxis.set_ticks([1, 2, 3])
for xval in range(0, 11):
    if xval in redvertcolor:
        ax2.axvline(x=xval, color='r', linestyle='--')
    else:
        ax2.axvline(x=xval, linestyle='--')
ax2.set_title("cajas poco utilizadas")
ax2.set_xlabel("hora")
ax2.set_ylabel("número de caja")
fig.tight_layout()

