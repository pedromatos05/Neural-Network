import numpy as np
# Multiplicação de Matrizes , isto é muito importante para redes neurais , pois as redes neurais são basicamente uma série de multiplicações de matrizes, em cada  camada de uma rede neural tem isto
X = np.array([
    [1, 2, 3], 
    [3, 4, 6], 
])

W = np.array([
    [0.2, 0.4],
    [0.1, 0.3],
    [0.3, 0.9]
])

resultado = np.dot(X, W) # Isto vai multiplicar a matriz X pela matriz W usando a função dot do numpy

print(resultado) # Isto vai imprimir o resultado da multiplicação das matrizes X e W
print(resultado.shape) # Isto vai imprimir o formato do resultado da multiplicação das matrizes X e W
