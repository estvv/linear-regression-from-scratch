import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors

# ─────────────────────────────────────────────
# DONNÉES
# ─────────────────────────────────────────────
random.seed(42)
TRUE_W, TRUE_B = 2.0, 1.0

xs = [i * 0.1 for i in range(50)]
ys = [TRUE_W * x + TRUE_B + random.gauss(0, 0.5) for x in xs]

BG       = "#0f0f1a"
AX_BG    = "#13132b"
GRID_COL = "#1e1e3a"

def _style(ax, title):
    ax.set_facecolor(AX_BG)
    ax.set_title(title, color="white", fontsize=10, pad=8)
    ax.tick_params(colors="#888888", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COL)
    ax.grid(color=GRID_COL, linewidth=0.5)

def _mse(w, b):
    n = len(xs)
    return sum((w * x + b - y) ** 2 for x, y in zip(xs, ys)) / n

# ⚡ PLACEHOLDER — remplace par tes fonctions quand tu les as codées
def _run_gradient_descent(lr=0.01, epochs=300):
    w, b = 0.0, 0.0
    history = []
    n = len(xs)
    for _ in range(epochs):
        y_preds = [w * x + b for x in xs]
        loss_val = sum((yp - yt) ** 2 for yp, yt in zip(y_preds, ys)) / n
        grad_w = sum(2 * (yp - yt) * x for yp, yt, x in zip(y_preds, ys, xs)) / n
        grad_b = sum(2 * (yp - yt) for yp, yt in zip(y_preds, ys)) / n
        w -= lr * grad_w
        b -= lr * grad_b
        history.append((w, b, loss_val))
    return history


# ─────────────────────────────────────────────
# FIGURE PRINCIPALE
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(17, 10))
fig.patch.set_facecolor(BG)
fig.suptitle("Gradient Descent — de la problématique à la convergence",
             color="white", fontsize=14, y=0.98)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)


# ── Panel 1 (haut-gauche) : Les données brutes ───────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
_style(ax1, "Étape 1 — Les données\nOn cherche w et b sans les connaître")

ax1.scatter(xs, ys, color="#7ec8e3", s=22, alpha=0.8, zorder=3,
            label="données bruitées")
x_ends = [xs[0], xs[-1]]
ax1.plot(x_ends, [TRUE_W * x + TRUE_B for x in x_ends],
         color="#ff6b6b", lw=2, linestyle="--", label=f"vérité : y={TRUE_W}x+{TRUE_B}")
ax1.plot(x_ends, [0, 0],
         color="#ffd166", lw=1.8, linestyle=":", label="départ algo : y=0")
ax1.set_xlabel("x", color="#888888", fontsize=8)
ax1.set_ylabel("y", color="#888888", fontsize=8)
ax1.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=7, loc="upper left")


# ── Panel 2 (haut-centre) : Heatmap loss landscape ───────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
_style(ax2, "Étape 2 — Surface de perte\nL'algo cherche le creux (★)")

w_vals = [w * 0.05 for w in range(-20, 101)]
b_vals = [b * 0.05 for b in range(-20, 81)]
Z = [[_mse(w, b) for w in w_vals] for b in b_vals]

im = ax2.imshow(Z, origin="lower", aspect="auto",
                extent=[w_vals[0], w_vals[-1], b_vals[0], b_vals[-1]],
                cmap="magma", vmax=30)
plt.colorbar(im, ax=ax2, label="MSE")
ax2.scatter([TRUE_W], [TRUE_B], color="#00ff99", s=120, zorder=5,
            marker="*", label=f"min (w={TRUE_W}, b={TRUE_B})")
ax2.scatter([0], [0], color="#ffd166", s=60, zorder=5,
            marker="o", label="départ (0, 0)")
ax2.set_xlabel("w", color="#888888", fontsize=8)
ax2.set_ylabel("b", color="#888888", fontsize=8)
ax2.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=7)


# ── Panel 3 (haut-droite) : Coupe loss(w) avec b fixé ────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
_style(ax3, f"Étape 2 — Coupe loss(w)  [b={TRUE_B}]\nLa descente cherche le fond du U")

loss_w = [_mse(w, TRUE_B) for w in w_vals]
ax3.plot(w_vals, loss_w, color="#7ec8e3", lw=2)
ax3.axvline(TRUE_W, color="#00ff99", linestyle="--", lw=1.5,
            label=f"w optimal = {TRUE_W}")
ax3.axvline(0, color="#ffd166", linestyle=":", lw=1.5, label="départ w=0")
ax3.set_xlabel("w", color="#888888", fontsize=8)
ax3.set_ylabel("MSE", color="#888888", fontsize=8)
ax3.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=7)


# ── Panel 4 (bas-gauche) : Le gradient ───────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
_style(ax4, "Étape 3 — Le gradient\nFlèche = direction du prochain pas (−grad)")

ax4.plot(w_vals, loss_w, color="#7ec8e3", lw=2)
ax4.axvline(TRUE_W, color="#00ff99", linestyle="--", lw=1, alpha=0.5)

for w0, col in zip([0.0, 1.0, 3.5], ["#ffd166", "#06d6a0", "#ef476f"]):
    l0 = _mse(w0, TRUE_B)
    h  = 0.001
    grad = (_mse(w0 + h, TRUE_B) - _mse(w0 - h, TRUE_B)) / (2 * h)
    span = 0.55
    x_tan = [w0 - span, w0 + span]
    y_tan = [l0 + grad * (x - w0) for x in x_tan]
    ax4.plot(w0, l0, "o", color=col, markersize=8, zorder=5)
    ax4.plot(x_tan, y_tan, color=col, lw=1.6, alpha=0.8)
    ax4.annotate("", xy=(w0 - grad * 0.16, l0), xytext=(w0, l0),
                 arrowprops=dict(arrowstyle="->", color=col, lw=1.8))
    ax4.text(w0 + 0.08, l0 + 0.5,
             f"w={w0}\ng={grad:.0f}", color=col, fontsize=7)

ax4.set_xlabel("w", color="#888888", fontsize=8)
ax4.set_ylabel("MSE", color="#888888", fontsize=8)


# ── Panel 5 (bas-centre) : Convergence de la droite (snapshots) ──────────────
ax5 = fig.add_subplot(gs[1, 1])
_style(ax5, "Étape 4 — Convergence\nLa droite se cale sur les données")

history = _run_gradient_descent()
snap_epochs = [0, 10, 30, 80, 299]
snap_colors = ["#ffd166", "#f4a261", "#e76f51", "#c1121f", "#00ff99"]
snap_labels = ["epoch 1", "epoch 10", "epoch 30", "epoch 80", "epoch 300 (final)"]

ax5.scatter(xs, ys, color="#7ec8e3", s=18, alpha=0.6, zorder=3)
for ep, col, lbl in zip(snap_epochs, snap_colors, snap_labels):
    w, b, _ = history[ep]
    ax5.plot(x_ends, [w * x + b for x in x_ends],
             color=col, lw=1.8, alpha=0.9, label=lbl)

ax5.set_xlabel("x", color="#888888", fontsize=8)
ax5.set_ylabel("y", color="#888888", fontsize=8)
ax5.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=7)


# ── Panel 6 (bas-droite) : Courbe de loss ────────────────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
_style(ax6, "Étape 4 — Loss au fil des epochs\nConverge vers le minimum")

losses = [h[2] for h in history]
ax6.plot(range(len(losses)), losses, color="#7ec8e3", lw=2)

for ep, col in zip(snap_epochs, snap_colors):
    ax6.axvline(ep, color=col, linestyle=":", lw=1, alpha=0.7)
    ax6.plot(ep, losses[ep], "o", color=col, markersize=6, zorder=5)

final_w, final_b, _ = history[-1]
ax6.text(len(losses) * 0.55, losses[0] * 0.7,
         f"final :\nw = {final_w:.3f}\nb = {final_b:.3f}",
         color="white", fontsize=8,
         bbox=dict(boxstyle="round,pad=0.4", facecolor="#1a1a2e", edgecolor="#333355"))

ax6.set_xlabel("epoch", color="#888888", fontsize=8)
ax6.set_ylabel("MSE", color="#888888", fontsize=8)


# ─────────────────────────────────────────────
# EXPORT
# ─────────────────────────────────────────────
out = "gradient_descent.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"Sauvegardé → {out}")
