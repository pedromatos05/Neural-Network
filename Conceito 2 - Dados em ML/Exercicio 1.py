import numpy as np
np.random.seed(42)

# Dataset: 8 alunos, 2 notas cada
X = np.array([
    [8, 7], [5, 6], [9, 10], [3, 4],
    [7, 8], [6, 5], [4,  3], [8,  9]
], dtype=float)

y = np.array([7.5, 5.5, 9.5, 3.5, 7.5, 5.5, 3.5, 8.5])

# PASSO 1 — Baralhar
indices = np.arange(len(X))
np.random.shuffle(indices)
X, y = X[indices], y[indices]

# PASSO 2 — Standardizar
X_media = X.mean(axis=0)
X_std   = X.std(axis=0)
X_stand = (X - X_media) / X_std

# PASSO 3 — Dividir
n          = len(X)
n_treino   = int(n * 0.70)

X_treino   = X_stand[:n_treino]
X_teste    = X_stand[n_treino:]
y_treino   = y[:n_treino]
y_teste    = y[n_treino:]

print("X treino:\n",  X_treino)
print("X teste:\n",   X_teste)
print("y treino:",    y_treino)
print("y teste:",     y_teste)

# Objetivo fazer este ex porem com normalização 