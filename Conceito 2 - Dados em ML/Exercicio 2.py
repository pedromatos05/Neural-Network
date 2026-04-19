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

# PASSO 2 — Normalizar   
X_max = X.max(axis=0)
X_min = X.min(axis=0)
X_norm = (X - X_min) / (X_max - X_min)

# PASSO 3 — Dividir
n          = len(X)
n_treino   = int(n * 0.70)

X_treino   = X_norm[:n_treino]
X_teste    = X_norm[n_treino:]
y_treino   = y[:n_treino]
y_teste    = y[n_treino:]

print("X treino:\n",  X_treino)
print("X teste:\n",   X_teste)
print("y treino:",    y_treino)
print("y teste:",     y_teste)


#Diferenca Normalização valores entre 0 e 1

#Standardização valores centrados em 0, maioria entre -3 e 3


# o que nos concluimos com isto , 

'''
Normalização    →  "onde está cada aluno
                    numa escala de 0 a 1?"
                    útil para perceber posições relativas

Standardização  →  "o quanto cada aluno
                    se afasta da média?"
                    útil para detetar quem é muito bom
                    ou muito fraco
'''
