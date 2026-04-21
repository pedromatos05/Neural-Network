'''
O Probelam Central
Um neural network tem um objetivo muito simples:
Encontrar os pesos que fazem a rede errar o mesmo possível  

Para Isso respondemos a duas perguntas : 
 1. O quanto estou a errar?  <- loss function
 2. Para que lado devo ajustar? <-Gradiente e Derivadas

'''

# O que é uma função ?
# Uma função recebe um valor, faz algo com ele e depois devolve um valor

import numpy as np 

def funcao(x):
    return x**2

print(funcao(2)) # 4
print(funcao(3)) # 9 
print(funcao(-2)) # 4

# Numa rede Neural Network, a função ais importante é a loss function , recebe os pesos e devolve o erro atual da rede 
# O objetivo é encontrar os pesos que tornam esse erro o mais pequeno possível 

'''
*
       / \
      /   \         *
     /     \       / \
    /       \     /   \
   /         \   /     \
  /           \ /       \
 /             *         \
                          \
                           * ← queremos chegar aqui (mínimo)

'''

# O Que é uma derivada , A derivada mede a inclinação de uma função

def derivada(x):
    return 2 * x

print(derivada(3))  # 6 função está a subir, inclinação positiva
print(derivada(-3)) # -6 função está a descer , inclinação negativa 
print(derivada(0))  # 0 função é plana, inclinação zero


def f1(x): return x ** 2
def d1(x): return 2 * x
print(d1(0))   # 0 → mínimo nos queremos chegar aqui 