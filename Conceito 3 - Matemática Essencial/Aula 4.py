import numpy as np
np.random.seed(42)

# Dados simples, 1 feature, 1 label
X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])  # y = 2x

# Peso inicial aleatório
w = np.random.randn(1)[0]
lr = 0.01

print(f"Peso inicial: {w:.4f}\n")
print(f"{'Época':<8} {'Peso':<12} {'Loss'}")
print("-" * 32)

for epoca in range(20):

    # Forward pass -> previsão atual
    y_pred = X * w

    # Loss -> Mean Squared Error (erro médio ao quadrado)
    loss = np.mean((y_pred - y) ** 2)

    # Gradiente da loss em relação a w
    gradiente = np.mean(2 * X * (y_pred - y))

    # Atualizar peso
    w = w - lr * gradiente

    print(f"{epoca:<8} {w:<12.4f} {loss:.4f}")

print(f"\nPeso final: {w:.4f}")
print(f"Esperado:   2.0000  (porque y = 2x)")

#se alterar o lr para 0.1 ou 0.001, o resultado final afasta-se do 2 (y = 2x) ou seja converge mais devagar 

