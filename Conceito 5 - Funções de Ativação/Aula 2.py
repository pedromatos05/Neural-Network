import numpy as np
np.random.seed(42)

X = np.array([
    [8, 7], [5, 6], [9, 10],
    [3, 4], [2, 3], [7,  8],
], dtype=float)

y = np.array([1, 0, 1, 0, 0, 1], dtype=float)

X_min  = X.min(axis=0)
X_max  = X.max(axis=0)
X_norm = (X - X_min) / (X_max - X_min)

def sigmoid(z): return 1 / (1 + np.exp(-z))
def relu(z):    return np.maximum(0, z)
def tanh(z):    return np.tanh(z)

def treinar(ativacao, nome):
    w  = np.random.randn(2)
    b  = np.random.randn(1)
    lr = 0.5

    for epoca in range(100):
        z      = np.dot(X_norm, w) + b
        y_pred = ativacao(z)
        erro   = y_pred - y
        dw     = np.dot(X_norm.T, erro) / len(X)
        db     = np.mean(erro)
        w      = w - lr * dw
        b      = b - lr * db

    y_final   = ativacao(np.dot(X_norm, w) + b)
    previsoes = (y_final >= 0.5).astype(int)
    precisao  = np.mean(previsoes == y) * 100
    print(f"{nome:<12} precisao={precisao:.0f}%")

np.random.seed(42)
treinar(sigmoid, "Sigmoid")
np.random.seed(42)
treinar(tanh,    "Tanh")