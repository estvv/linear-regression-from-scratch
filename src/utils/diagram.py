"""
python diagram.py  ->  diagram.png
Technical diagram of linear regression / gradient descent training cycle.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

BG, AX_BG = "#0f0f1a", "#0f0f1a"

fig, ax = plt.subplots(figsize=(14, 7))
fig.patch.set_facecolor(BG)
ax.set_facecolor(AX_BG)
ax.set_xlim(0, 14)
ax.set_ylim(0, 7)
ax.axis("off")

fig.suptitle("Linear Regression — training cycle", color="white", fontsize=14)

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def box(ax, x, y, w, h, color, label, sublabel=None):
    rect = mpatches.FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.08",
        facecolor=color, edgecolor="white", linewidth=1.2, zorder=3
    )
    ax.add_patch(rect)
    ax.text(x, y + (0.18 if sublabel else 0), label,
            ha="center", va="center", color="white",
            fontsize=9, fontweight="bold", zorder=4)
    if sublabel:
        ax.text(x, y - 0.28, sublabel,
                ha="center", va="center", color="#cccccc",
                fontsize=7.5, zorder=4)

def arrow(ax, x1, y1, x2, y2, color="white", label=None):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=1.6, mutation_scale=14),
                zorder=2)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx + 0.1, my + 0.18, label,
                color=color, fontsize=7.5, ha="center", zorder=5)

def code(ax, x, y, txt, color="#aaaaff"):
    ax.text(x, y, txt, color=color, fontsize=8,
            ha="center", va="center",
            fontfamily="monospace", zorder=5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a2e",
                      edgecolor="#333366", linewidth=1))

# ─────────────────────────────────────────────
# Step 1 — Data
# ─────────────────────────────────────────────
box(ax, 1.5, 3.5, 2.2, 1.0, "#1d3557", "DATA", "(xs, ys)")
code(ax, 1.5, 2.5, "xs = [x0, x1, ... xn]\nys = [y0, y1, ... yn]")

# ─────────────────────────────────────────────
# Step 2 — Prediction
# ─────────────────────────────────────────────
box(ax, 4.5, 3.5, 2.2, 1.0, "#2d6a4f", "PREDICTION", "y_hat = w*x + b")
code(ax, 4.5, 2.5, "y_hat = w * x + b")
ax.text(4.5, 1.85, "w=0, b=0 at start", color="#888888",
        fontsize=7, ha="center")

# ─────────────────────────────────────────────
# Step 3 — Loss
# ─────────────────────────────────────────────
box(ax, 7.5, 3.5, 2.2, 1.0, "#7b2d00", "LOSS (MSE)", "measures the error")
code(ax, 7.5, 2.5, "MSE = S(y_hat - y)^2 / n")

# ─────────────────────────────────────────────
# Step 4 — Gradients
# ─────────────────────────────────────────────
box(ax, 10.5, 3.5, 2.2, 1.0, "#4a1942", "GRADIENTS", "correction direction")
code(ax, 10.5, 2.5, "dw = S(y_hat-y)*x * 2/n\ndb = S(y_hat-y)   * 2/n")

# ─────────────────────────────────────────────
# Step 5 — Update
# ─────────────────────────────────────────────
box(ax, 13.0, 3.5, 1.6, 1.0, "#1a3a4a", "UPDATE", "w, b <- new values")
code(ax, 13.0, 2.5, "w -= lr * dw\nb -= lr * db")

# ─────────────────────────────────────────────
# Main arrows (left -> right)
# ─────────────────────────────────────────────
arrow(ax, 2.6, 3.5, 3.4, 3.5, "#7ec8e3")
arrow(ax, 5.6, 3.5, 6.4, 3.5, "#7ec8e3")
arrow(ax, 8.6, 3.5, 9.4, 3.5, "#7ec8e3")
arrow(ax, 11.6, 3.5, 12.2, 3.5, "#7ec8e3")

# ─────────────────────────────────────────────
# Training loop (UPDATE -> PREDICTION)
# ─────────────────────────────────────────────
ax.annotate("", xy=(4.5, 4.0), xytext=(13.0, 4.0),
            arrowprops=dict(
                arrowstyle="-|>", color="#ffd166",
                lw=2, mutation_scale=14,
                connectionstyle="arc3,rad=-0.35"
            ), zorder=2)
ax.text(8.75, 5.6, "<- training loop (x epochs)",
        color="#ffd166", fontsize=8.5, ha="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1500",
                  edgecolor="#ffd166", linewidth=1))

# ─────────────────────────────────────────────
# Loss goal label
# ─────────────────────────────────────────────
ax.annotate("", xy=(7.5, 3.0), xytext=(7.5, 3.5),
            arrowprops=dict(arrowstyle="-|>", color="#ef476f",
                            lw=1.4, mutation_scale=12), zorder=2)
ax.text(7.5, 1.65, "goal: minimize", color="#ef476f",
        fontsize=7.5, ha="center")

# ─────────────────────────────────────────────
# Step numbers
# ─────────────────────────────────────────────
for i, (x, lbl) in enumerate([(1.5,"1."), (4.5,"2."), (7.5,"3."), (10.5,"4."), (13.0,"5.")], 1):
    ax.text(x, 4.2, lbl, color="#aaaaaa", fontsize=11, ha="center", zorder=5)

# ─────────────────────────────────────────────
# Final result label
# ─────────────────────────────────────────────
ax.text(7.0, 0.55,
        "After N epochs:  w ~= true value,  b ~= true value  ->  the line fits the data",
        color="#00ff99", fontsize=8.5, ha="center",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#001a0d",
                  edgecolor="#00ff99", linewidth=1))

fig.savefig("src/utils/diagram.png", dpi=150, bbox_inches="tight", facecolor=BG)
print("Saved -> src/utils/diagram.png")
