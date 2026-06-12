

def loss(w: float, b: float, xs: list[float], ys: list[float]) -> float:
    """
    w: Coefficient de ta droite
    b: Ordonnée à l'origine de ta droite
    xs: Les x de tes données
    ys: Les y de tes données

    Mesure à quel point ta droite est mauvaise. Retourne un seul nombre. Plus il est petit, mieux c'est.
    Mean Squared Error

    Fonction affine: y = w * x + b

    Donc, on additionne

    Formule = MSE = (1/n) * Σ((w * x + b) - y) ** 2
     - n: nombre de données
     - Σ: somme pour tous les points de données
     - w * x + b: la prédiction de ta droite pour un x donné
     - y: la valeur réelle pour ce x
     - (w * x + b) - y: l'erreur de ta prédiction
     - ((w * x + b) - y) ** 2: l'erreur au carré (pour éviter les problèmes de signe et pour pénaliser les grandes erreurs)
     - (1/n) * Σ(...): la moyenne de toutes les erreurs au carré, ce qui donne une mesure globale de la qualité de ta droite par rapport à tes données.
    """

    MSE = 0

    for x, y in zip(xs, ys):
        MSE += ((w * x + b) - y)  ** 2

    return MSE / len(xs)

def gradients(w: float, b: float, xs: list[float], ys: list[float]) -> tuple[float, float]:
    """
    w: Coefficient de ta droite
    b: Ordonnée à l'origine de ta droite
    xs: Les x de tes données
    ys: Les y de tes données

    Retourne les gradients par rapport à w et b. C'est à dire, dans quelle direction il faut aller pour améliorer ta droite.

    dw: Gradient par rapport à w. Si dw est positif, cela signifie que pour améliorer ta droite, tu devrais diminuer w. Si dw est négatif, tu devrais augmenter w.
    db: Gradient par rapport à b. Si db est positif, cela signifie que pour améliorer ta droite, tu devrais diminuer b. Si db est négatif, tu devrais augmenter b.

    Formules:
     - dw = (2/n) * Σ(x * ((w * x + b) - y))
     - db = (2/n) * Σ((w * x + b) - y)
    """

    dw = 0
    db = 0

    for x, y in zip(xs, ys):
        dw += 2 * x * ((w * x + b) - y)
        db += 2 * ((w * x + b) - y)

    return dw / len(xs), db / len(xs)

def train(xs: list[float], ys: list[float], lr: float = 0.01, epochs: int = 300) -> list[tuple[float, float, float]]:
    """
    xs: Les x de tes données
    ys: Les y de tes données
    lr: Learning rate. C'est à dire, à quelle vitesse tu veux que ton modèle apprenne. Un learning rate trop élevé peut faire diverger ton modèle, tandis qu'un learning rate trop bas peut rendre l'entraînement très lent.
    epochs: Le nombre de fois que tu veux faire passer tes données à travers ton modèle pour l'entraîner.

    Retourne l'historique de l'entraînement, c'est à dire une liste de tuples (w, b, loss) pour chaque étape d'entraînement. Cela te permettra de visualiser comment ton modèle a évolué au fil du temps.
     - w: Le coefficient de ta droite à cette étape d'entraînement
     - b: L'ordonnée à l'origine de ta droite à cette étape d'entraînement
     - loss: La valeur de la fonction de perte (MSE) à cette étape d'entraînement
    """

    w, b = 0.0, 0.0
    history = []

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        current_loss = loss(w, b, xs, ys)

        history.append((w, b, current_loss))

        dw, db = gradients(w, b, xs, ys)

        w -= lr * dw
        b -= lr * db

    return history
