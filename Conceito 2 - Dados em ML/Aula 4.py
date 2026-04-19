import numpy as np
# Standardização — média 0, desvio padrão 1
X = np.array([
    [8, 7],
    [5, 6],
    [9, 10],
    [3, 4],
    [7, 8],
], dtype=float)

# Fórmula: (valor - média) / desvio padrão
X_media = X.mean(axis=0)  # média de cada coluna Basicamente a média de cada disciplina 
X_std   = X.std(axis=0)   # desvio padrão de cada coluna

'''
sem axis  →  opera em tudo       →  devolve 1 número
axis=0    →  opera por coluna    →  devolve 1 valor por coluna  ← o mais usado
axis=1    →  opera por linha     →  devolve 1 valor por linha

'''

X_stand = (X - X_media) / X_std

print("Original:\n", X)
print("\nStandardizado:\n", X_stand)
# Valores centrados em 0, maioria entre -3 e 3