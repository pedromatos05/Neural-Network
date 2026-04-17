import numpy as np 
np.random.seed(0)

X = np.array([  # este vai ser basicamente 3 alunos com duas caracteristicas , que sao nota de mat e de port
    [8, 7],
    [5, 6],
    [9, 10]
])

W = np.random.randn(2, 1) # Aqui vai ser basicamente 2 entradas com 1 neurônio
b = np.random.randn(1) # este basicamente vai fucniocar como o + b de uma equção de reta. Dá à rede um ajuste extyra independente das entradas

saida = np.dot(X, W) + b #é basicamente o coração do neurônio na parte que se faz dot que é multiplicacao de matrizes e depois adicionamos a bia que é sempre igual 

print("Pesos:", W)
print("Bias:", b)
print("Saída:", saida)
print("Shape da saída:", saida.shape)

# isto é exatamente uma camada de uma neural network , sendo função de ativação ainda. Depois vamos introduzir um comportamento não linear 

# O resultado não vai dizer nada , pois precisamos de mais camadas para concluir algo , com isto vamos precisar de mais dados 