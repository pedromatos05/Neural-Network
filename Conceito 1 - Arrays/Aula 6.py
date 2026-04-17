import numpy as np

np.random.seed(42)

a = np.random.rand(3, 4)
                          # A Diferença entre os dois é que o rand é entre 0 e 1 e o randn é valores a volta do zero , valores extremos são dificies de aparecer 
b = np.random.randn(3 ,4)

print(a)
print(b)