import numpy as np 

# Este Serve para fazer a divizão dos dados em treino , validação e teste 

X = np.array([
    [8, 7], [5, 6], [9, 10], [3, 4], [7, 8],
    [6, 5], [4, 3], [8, 9],  [2, 5], [7, 6]
])

y = np.array([7.5, 5.5, 9.5, 3.5, 7.5, 5.5, 3.5, 8.5, 3.5, 6.5])

n = len(X) #tamanho

n_treino = int(n * 0.70) #70% dos dados para treino , este srve para a rede aprender com estes dados os pesos são ajudatados aqui
n_validação = int(n * 0.15) #15% dos dados para validação usas durante o treino para ver se a rede está generalizar ou a fazer overfitting
n_teste = int (n* 0.15) #15% dos dados para teste só se usa no final , é avalização final  , e a rede nunca viu estes dados  

X_treino = X[:n_treino] # Basicamente do inicio ate o n_treino
X_validação = X[n_treino:n_treino + n_validação] # basicamente começa no numero de treino mais n de validacao
X_teste = X[n_treino + n_validação:] # o numero de treino mais o numero de validacao até o final


y_treino = y[:n_treino]
y_validação = y[n_treino:n_treino + n_validação]
y_teste = y[n_treino + n_validação:]

print("Treino:   ", X_treino.shape)     # (7, 2)
print("Validação:", X_validação.shape)  # (1, 2)
print("Teste:    ", X_teste.shape)      # (2, 2)

