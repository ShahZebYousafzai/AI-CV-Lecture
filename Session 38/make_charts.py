import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyArrowPatch

plt.rcParams.update({
    "font.family": "Carlito",
    "axes.edgecolor": "#C9DBD8",
    "text.color": "#12343B",
    "axes.labelcolor": "#5A7A80",
    "xtick.color": "#5A7A80",
    "ytick.color": "#5A7A80",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

INK   = "#12343B"
MUTED = "#5A7A80"
GRID  = "#E3EFED"
FP32C = "#034E5B"
FP16C = "#028090"
INT8C = "#02C39A"
COLS  = [FP32C, FP16C, INT8C]
LABELS = ["FP32", "FP16", "INT8"]

# ---------------------------------------------------------------- chart 1
# Ultralytics YOLO26 TensorRT benchmarks (COCO detection)
lat = {"NVIDIA A100": [0.52, 0.34, 0.28], "RTX 3080 12GB": [1.06, 0.62, 0.52]}
maps = [0.37, 0.37, 0.33]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 3.05),
                               gridspec_kw={"width_ratios": [1.55, 1.0], "wspace": 0.30})

x = np.arange(len(lat)); w = 0.24
for i, (lbl, c) in enumerate(zip(LABELS, COLS)):
    vals = [lat[g][i] for g in lat]
    b = ax1.bar(x + (i - 1) * (w + 0.02), vals, w, color=c, label=lbl, zorder=3)
    for r, v in zip(b, vals):
        ax1.text(r.get_x() + r.get_width() / 2, v + 0.028, f"{v:.2f}",
                 ha="center", va="bottom", fontsize=9.5, color=INK)
ax1.set_xticks(x); ax1.set_xticklabels(list(lat), fontsize=10.5, color=INK)
ax1.set_ylabel("milliseconds per image", fontsize=9.5)
ax1.set_title("Inference latency falls with precision", fontsize=11.5,
              color=INK, loc="left", pad=26, fontweight="bold")
ax1.set_ylim(0, 1.25)
ax1.yaxis.grid(True, color=GRID, lw=1, zorder=0); ax1.set_axisbelow(True)
ax1.legend(frameon=False, fontsize=9.5, ncol=3, loc="lower left",
           bbox_to_anchor=(0.0, 1.005), labelcolor=MUTED, handlelength=1.1,
           columnspacing=1.4, borderpad=0)
for s in ("top", "right", "left"): ax1.spines[s].set_visible(False)

b2 = ax2.bar(LABELS, maps, 0.5, color=COLS, zorder=3)
for r, v in zip(b2, maps):
    ax2.text(r.get_x() + r.get_width() / 2, v + 0.006, f"{v:.2f}",
             ha="center", va="bottom", fontsize=9.5, color=INK)
ax2.set_ylim(0, 0.405)
ax2.set_ylabel("mAP 50-95 on COCO", fontsize=9.5)
ax2.set_title("Accuracy holds at FP16, dips at INT8", fontsize=11.5,
              color=INK, loc="left", pad=26, fontweight="bold")
ax2.yaxis.grid(True, color=GRID, lw=1, zorder=0); ax2.set_axisbelow(True)
ax2.tick_params(axis="x", labelsize=10.5, labelcolor=INK)
for s in ("top", "right", "left"): ax2.spines[s].set_visible(False)

fig.savefig("/home/claude/s38/figures/fig_precision_benchmark.png", dpi=220,
            bbox_inches="tight", pad_inches=0.10)
plt.close(fig)

# ---------------------------------------------------------------- chart 2
fig, (axA, axB) = plt.subplots(2, 1, figsize=(7.9, 3.55),
                               gridspec_kw={"height_ratios": [1.0, 0.86], "hspace": 0.78})

rng = np.random.default_rng(7)
wts = np.concatenate([rng.normal(0, 0.055, 40000), rng.normal(0, 0.16, 6000)])
wts = wts[(wts > -0.40) & (wts < 0.40)]
axA.hist(wts, bins=220, color="#BFE0DA", edgecolor="none")
axA.set_xlim(-0.47, 0.47); axA.set_yticks([])
top = axA.get_ylim()[1]
axA.set_ylim(0, top * 1.20)
axA.set_title("FP32: a smooth distribution, ~4 billion representable values",
              fontsize=10.5, color=INK, loc="left", pad=9, fontweight="bold")
axA.tick_params(axis="x", labelsize=9)
for s_ in ("top", "right", "left"): axA.spines[s_].set_visible(False)
axA.axvline(-0.40, color="#C15B4E", lw=1.6, ls=(0, (4, 3)))
axA.axvline(0.40, color="#C15B4E", lw=1.6, ls=(0, (4, 3)))
axA.text(-0.385, top * 1.10, "min  -0.40", fontsize=9.5, color="#C15B4E",
         ha="left", va="center")
axA.text(0.385, top * 1.10, "max  +0.40", fontsize=9.5, color="#C15B4E",
         ha="right", va="center")

axB.set_xlim(-0.47, 0.47); axB.set_ylim(0, 1); axB.set_yticks([])
for s_ in ("top", "right", "left"): axB.spines[s_].set_visible(False)
for lv in np.linspace(-0.40, 0.40, 51):
    axB.plot([lv, lv], [0.34, 0.60], color=FP16C, lw=1.0, alpha=0.7)
axB.plot([-0.44, 0.44], [0.34, 0.34], color="#9FC4C0", lw=1.2)
axB.set_title("INT8: the same range, chopped into 256 legal levels",
              fontsize=10.5, color=INK, loc="left", pad=9, fontweight="bold")
axB.set_xticks([]); axB.spines["bottom"].set_visible(False)
for xv, lab in [(-0.40, "-128"), (0.0, "0"), (0.40, "+127")]:
    axB.plot([xv], [0.34], marker="o", ms=6, color=INK, zorder=5)
    axB.text(xv, 0.19, lab, ha="center", va="top", fontsize=9.5, color=INK)
axB.annotate("", xy=(0.1134, 0.64), xytext=(0.1134, 0.90),
             arrowprops=dict(arrowstyle="-|>", color="#C15B4E", lw=1.6))
axB.text(0.1134, 0.93, "0.1131 and 0.1145 both land on level 36  =  0.1134",
         fontsize=9.5, color="#C15B4E", ha="center", va="bottom")
axB.text(-0.465, -0.10, "scale = 0.40 / 127 = 0.00315        weight $\\approx$ scale $\\times$ (integer level)"
         "        one weight: 32 bits $\\rightarrow$ 8 bits",
         fontsize=10, color=MUTED, ha="left", va="top", transform=axB.transData)

fig.savefig("/home/claude/s38/figures/fig_quantization_mapping.png", dpi=220,
            bbox_inches="tight", pad_inches=0.14)
plt.close(fig)
print("charts written")
