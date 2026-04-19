import numpy as np
# Nesta Aula vamos fazer o Shuffling 

#Basicamente antes de dividir os dados , é uma boa prática baralhar os dados. 

np.random.seed(42)

X = np.array([
    [8, 7], [5, 6], [9, 10], [3, 4], [7, 8],
    [6, 5], [4, 3], [8, 9],  [2, 5], [7, 6]
])

y = np.array([7.5, 5.5, 9.5, 3.5, 7.5, 5.5, 3.5, 8.5, 3.5, 6.5])

indices = np.arange(len(X)) #cria um array de indices com o mesmo tamanho de X
np.random.shuffle(indices)

X = X[indices] #reorganiza X de acordo com os indices baralhados
y = y[indices] #reorganiza y de acordo com os indices baralhados

# IMPORTANTE: Tens de baralhar X e y juntos com os mesmos índices , se baralhares separado, as features deixam de corresponder às labels corretas. 

'''
# Imagina que tens estas features
Normalização vs Standardização
Este é um passo crítico que muita gente ignora. Redes neurais funcionam muito melhor quando os dados estão numa escala semelhante.
X = np.array([
    [8,    7,   25000],   # nota_mat, nota_port, salário_pai
    [5,    6,   80000],
    [9,    10,  45000],
])

# A feature "salário" tem valores na casa dos milhares
# As notas têm valores entre 0 e 10
# A rede vai dar muito mais "atenção" ao salário
# só por ter números maiores — mesmo que não seja mais importante
'''