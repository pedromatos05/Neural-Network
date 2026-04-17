import numpy as np

a = np.array([1, 2, 3, 4, 5, 6])
print(a.shape) 

b = a.reshape(2, 3) # basicamente vai organizar os dados e vai meter 2 linhas com 3 caracteristicas 
print(b) 

c = a.reshape(3, 2) # 3 linhas 2 caracteristicas
print(c)

print(b.T) #Transpose , basicamete vira a matriz
print(b.T.shape) # diz o shape da Transposta

