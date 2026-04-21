import numpy as np
np.random.seed(42)

# ─── DADOS ────────────────────────────────────────────
X = np.array([
    [8, 7], [5, 6], [9, 10],
    [3, 4], [2, 3], [7,  8],
], dtype=float)

y = np.array([1, 0, 1, 0, 0, 1], dtype=float)

# ─── NORMALIZAR ───────────────────────────────────────
X_min  = X.min(axis=0)
X_max  = X.max(axis=0)
X_norm = (X - X_min) / (X_max - X_min)

# ─── FUNÇÕES ──────────────────────────────────────────
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def forward(X, w, b):
    return sigmoid(np.dot(X, w) + b)

def loss(y_pred, y):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

# ─── PESOS INICIAIS ───────────────────────────────────
w  = np.random.randn(2)
b  = np.random.randn(1)
lr = 0.5

print(f"{'Epoca':<8} {'Loss':<12} {'Precisao'}")
print("-" * 32)

# ─── TREINO ───────────────────────────────────────────
for epoca in range(50):

    # Forward pass
    y_pred = forward(X_norm, w, b)

    # Loss atual
    l = loss(y_pred, y)

    # Gradientes
    erro       = y_pred - y
    dw         = np.dot(X_norm.T, erro) / len(X)
    db         = np.mean(erro)

    # Atualizar pesos
    w = w - lr * dw
    b = b - lr * db

    # Precisao
    previsoes = (y_pred >= 0.5).astype(int)
    precisao  = np.mean(previsoes == y) * 100

    print(f"{epoca:<8} {l:<12.4f} {precisao:.0f}%")

# ─── RESULTADO FINAL ──────────────────────────────────
print("\nPesos finais: ", w)
print("Bias final:   ", b)
print("\nAluno        Real    Previsto    Decisao")
print("-" * 45)

y_final = forward(X_norm, w, b)
for i in range(len(X)):
    decisao = "passou" if y_final[i] >= 0.5 else "chumbou"
    print(f"Aluno {i+1}      {int(y[i])}       {y_final[i]:.4f}      {decisao}")