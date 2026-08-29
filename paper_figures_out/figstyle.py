"""Shared colour palette for RTAR gender-bias paper figures."""

from matplotlib.colors import LinearSegmentedColormap

PALETTE = ["#8f0707", "#be6c65", "#d7c2c1", "#de99a1", "#de6e8c"]
DARK, MID, NEUT, LIGHT, PINK = PALETTE

DIVERGING = LinearSegmentedColormap.from_list("rtar_div", PALETTE, N=256)
