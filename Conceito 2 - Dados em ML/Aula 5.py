import numpy as np
#Normalização — escala entre 0 e 1
X = np.array([
    [8, 7],
    [5, 6],
    [9, 10],
    [3, 4],
    [7, 8],
], dtype=float)

# Fórmula: (valor - mínimo) / (máximo - mínimo)
X_min = X.min(axis=0)   # mínimo de cada coluna
X_max = X.max(axis=0)   # máximo de cada coluna

X_norm = (X - X_min) / (X_max - X_min)

print("Original:\n", X)
print("\nNormalizado:\n", X_norm)
# Todos os valores ficam entre 0 e 1


#Quando é que se usa cada um ?

'''
Normalização    → quando sabes que os dados têm
                  um mínimo e máximo fixo
                  ex: notas (0 a 20), pixels (0 a 255)

Standardização  → quando não sabes os limites
                  ou os dados têm outliers
                  ex: salários, preços de casas
                  ← mais usada em neural networks
'''