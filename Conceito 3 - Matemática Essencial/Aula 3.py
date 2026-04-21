import numpy as np

w_inicial = 5.0

def gradient_descent(w, lr, passos):
    for i in range(passos):
        gradiente = 2 * w
        w = w - lr * gradiente
    return w

resultado = gradient_descent(5.0, lr=0.01, passos=100)
print(f"lr=0.01 -> peso final: {resultado:.4f}")

# Learning rate ideal → converge bem
resultado = gradient_descent(5.0, lr=0.1, passos=100)
print(f"lr=0.1  -> peso final: {resultado:.4f}")

# Learning rate grande → passos grandes → pode não convergir
resultado = gradient_descent(5.0, lr=1.1, passos=20)
print(f"lr=1.1  -> peso final: {resultado:.4f}")

'''
O qu é O Learning Rate , o Que é o peso e o que é o gradiente ?

LR -> é basicamente um valor que defines antes de treinar 
Loss a diminuir devagar  ->  aumenta o lr
Loss a saltar para cima  ->  diminui o lr
Loss a diminuir bem      ->  mantém o lr

w(peso) , o Peso começa sempre aleaório. A rede não sabe nada no início, começa aleatório que vai ser melhorado ao logo do treino

O gradiente é o único valor que tem cálculo real. vem sempre dp errp emtre a previsão da rede e o valor real.

'''

'''
lr=0.01 -> peso final: 0.6631
lr=0.1  -> peso final: 0.0000
lr=1.1  -> peso final: 191.6880
'''