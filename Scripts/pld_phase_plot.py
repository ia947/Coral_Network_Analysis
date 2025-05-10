# -*- coding: utf-8 -*-
"""
Created on Fri May 9 23:09:05 2025

@author: isaac
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Define developmental phases with typical ranges, uncertainty, and colours
phases = [
    {"name": "Embryo",         "start": 0,   "end": 0.5,  "min": 0,   "max": 1,   "color": "#88ccee"},
    {"name": "Larvae formation",        "start": 0.5, "end": 2,    "min": 0.3, "max": 3,   "color": "#44aa99"},
    {"name": "Pre‑competency", "start": 2,   "end": 5,    "min": 1.5, "max": 7,   "color": "#117733"},
    {"name": "Competency",     "start": 5,   "end": 20,   "min": 4,   "max": 30,  "color": "#ddcc77"},
    {"name": "Settlement",     "start": 20,  "end": 30,   "min": 15,  "max": 45,  "color": "#cc6677"}
]

# Create figure
fig, ax = plt.subplots(figsize=(10, 4), dpi=300)

# Plot each phase on its own horizontal band
band_height = 0.8
spacing = 1.5
label_offset = 0.5

for i, ph in enumerate(phases):
    y = i * spacing

    # Add shaded uncertainty band
    ax.add_patch(Rectangle(
        (ph["min"], y),
        ph["max"] - ph["min"],
        band_height,
        facecolor="lightgray",
        edgecolor=None,
        alpha=0.4
    ))

    # Typical duration band with phase-specific colour
    ax.add_patch(Rectangle(
        (ph["start"], y),
        ph["end"] - ph["start"],
        band_height,
        facecolor=ph["color"],
        edgecolor="black",
        linewidth=0.5,
        alpha=0.8
    ))

    # Adjust label positioning
    if ph["name"] in ["Embryo", "Larvae formation", "Pre‑competency"]:
        # place label just to the right of the uncertainty band
        text_x = ph["max"] + label_offset
        ha = "left"
    else:
        # center label over the typical band
        text_x = (ph["start"] + ph["end"]) / 2
        ha = "center"

    ax.text(
        text_x,
        y + band_height / 2,
        ph["name"],
        ha=ha, va="center",
        fontsize=10,
        fontweight="bold"
    )

# Configure axes
ax.set_xlim(0, 50)
ax.set_ylim(-0.5, spacing * len(phases))
ax.set_xlabel("Time since fertilisation (days)", fontsize=12)
ax.set_yticks([])
ax.set_xticks(range(0, 51, 5))
ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.7)


plt.tight_layout()
plt.show()
