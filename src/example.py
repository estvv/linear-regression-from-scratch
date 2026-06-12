"""
Exemple réel : prédire le prix d'une pizza selon son diamètre.
Données collectées dans une vraie pizzeria (inventées mais réalistes).

python example.py  →  produit example.png
"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from main import loss, gradients, train

BG, AX_BG, GRID = "#0f0f1a", "#13132b", "#1e1e3a"

def _style(ax, title):
    ax.set_facecolor(AX_BG)
    ax.set_title(title, color="white", fontsize=10, pad=8)
    ax.tick_params(colors="#888888", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_xlabel("Diamètre (cm)", color="#888888", fontsize=8)

# ─────────────────────────────────────────────
# DONNÉES COLLECTÉES (relevé dans une pizzeria)
# diamètre (cm) → prix (€)
# ─────────────────────────────────────────────
data = [
    (20, 8.5),
    (22, 9.0),
    (24, 10.5),
    (24, 11.0),
    (26, 11.5),
    (28, 12.0),
    (28, 13.0),
    (30, 13.5),
    (30, 14.0),
    (32, 14.5),
    (32, 15.5),
    (34, 16.0),
    (36, 17.0),
    (38, 18.5),
    (40, 19.0),
]

xs = [d[0] for d in data]
ys = [d[1] for d in data]

history = train(xs, ys, lr=0.0001, epochs=500)
final_w, final_b, _ = history[-1]

# ─────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(12, 5))
fig.patch.set_facecolor(BG)
fig.suptitle("Prédire le prix d'une pizza selon son diamètre",
             color="white", fontsize=13)
gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

# — Gauche : données + droite trouvée
ax1 = fig.add_subplot(gs[0])
_style(ax1, "Données collectées + droite apprise")
ax1.set_ylabel("Prix (€)", color="#888888", fontsize=8)

ax1.scatter(xs, ys, color="#7ec8e3", s=60, zorder=4, label="pizzas relevées")

x_range = [min(xs) - 1, max(xs) + 1]
ax1.plot(x_range, [final_w * x + final_b for x in x_range],
         color="#ff6b6b", lw=2, label=f"droite apprise\ny = {final_w:.2f}x + ({final_b:.2f})")

# Prédictions pour quelques tailles non vues
for diam, col in [(25, "#ffd166"), (33, "#06d6a0"), (42, "#c77dff")]:
    prix_predit = final_w * diam + final_b
    ax1.plot([diam], [prix_predit], "^", color=col, markersize=10, zorder=5)
    ax1.annotate(f"⌀{diam}cm → {prix_predit:.1f}€",
                 xy=(diam, prix_predit), xytext=(diam - 7, prix_predit + 0.8),
                 color=col, fontsize=8,
                 arrowprops=dict(arrowstyle="->", color=col, lw=1.2))

ax1.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)

# — Droite : courbe de loss
ax2 = fig.add_subplot(gs[1])
_style(ax2, "Loss au fil de l'entraînement")
ax2.set_ylabel("MSE", color="#888888", fontsize=8)

losses = [h[2] for h in history]
ax2.plot(range(len(losses)), losses, color="#7ec8e3", lw=2)
ax2.text(len(losses) * 0.5, losses[0] * 0.6,
         f"w final = {final_w:.3f}\nb final = {final_b:.2f}\n\n"
         f"→ chaque +1cm de diamètre\n   coûte +{final_w:.2f}€",
         color="white", fontsize=8.5,
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#1a1a2e", edgecolor="#333355"))

fig.savefig("example.png", dpi=150, bbox_inches="tight", facecolor=BG)
print(f"Sauvegardé → example.png")
print(f"Droite apprise : y = {final_w:.3f}x + ({final_b:.2f})")
print(f"→ chaque +1cm de diamètre coûte +{final_w:.2f}€")
