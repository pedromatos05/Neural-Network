'''
O Perceptron é o bloco mais básico de uma neural network. é um único neurônio artificial.

BIOLÓGICO                        ARTIFICIAL
─────────────────────────────    ─────────────────────────────
Dendrites recebem sinais    →    Entradas (X) recebem dados
Corpo processa o sinal      →    Soma ponderada (X * W + b)
Axônio dispara ou não       →    Função de ativação (0 ou 1)
'''

'''
Estrutura do Perceptron
Entradas     Pesos        Soma          Ativação    Saída
────────────────────────────────────────────────────────
x1 ────► × w1 ──┐
                 ├──► (x1w1 + x2w2 + b) ──► f(x) ──► y
x2 ────► × w2 ──┘
              ↑
             bias
'''

'''
Tem 3 partes: 

1. Soma ponderada  →  z = (X * W) + b
2. Função ativação →  y = f(z)
3. Aprendizagem    →  ajustar W e b para minimizar o erro
'''

#Vamos Passar para o Exemplo Prático do Perceptron do Zero

import numpy as np 

np.random.seed(42) 

#Dados de exemplo 
# 6 alunos → mat, port
# label → 1 = passou, 0 = chumbou
X = np.array([
    [8, 7],
    [5, 6],
    [9, 10],
    [3, 4],
    [2, 3],
    [7, 8],
], dtype=float)

y = np.array([1, 0, 1, 0, 0, 1]) # quer dizer se passou ou reprovou

#Vanos usar uma Normalização  (Pois esta dentro um intervalo)
X_min  = X.min(axis=0)
X_max  = X.max(axis=0)
X_norm = (X - X_min) / (X_max - X_min)

w = np.random.randn(2)   # 1 peso por feature
b = np.random.randn(1)   # 1 bias

print("Pesos iniciais:", w)
print("Bias inicial:  ", b)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Exemplos
print(sigmoid(-3))   # 0.047 → muito perto de 0 → chumbou
print(sigmoid(0))    # 0.500 → incerto
print(sigmoid(3))    # 0.952 → muito perto de 1 → passou


def forward(X, w, b):

    # PASSO 1 — soma ponderada
    z = np.dot(X, w) + b

    # PASSO 2 — função de ativação
    y_pred = sigmoid(z)

    return y_pred

# Testar com os dados
y_pred = forward(X_norm, w, b)
print("\nPrevisões iniciais:", y_pred)
print("Labels reais:      ", y)


def loss(y_pred, y):
    # evitar log(0) que daria infinito
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

