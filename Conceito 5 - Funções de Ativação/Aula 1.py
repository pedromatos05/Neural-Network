import numpy as np 

# Para que serve a função de ativação? 
# Sem funcções de ativação , uma neural network seria matematicamente igual a uma rede com 1 camada 
# As funões de ativação introduzem não linearidade , o que permite à rede aprender padrões complexos.

# Sem ativação, a rede aprende linhas retas 
# Com ativação, a rede pode aprender curvas e padrões mais complexos

# Sigmoid , transforma qualquer valor num valor entre 0 e 1 

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

valores = [-10, -3, -1, 0, 1, 3, 10]
for v in valores:
    print(f"sigmoid({v:>4}) = {sigmoid(v):.4f}")


# O ponto de decisão seria o 0.5, ou seja, se a saída for maior que 0.5, podemos considerar como classe 1, caso contrário, classe 0.

'''
✅  Última camada de classificação binária
✅  Quando precisas de uma probabilidade entre 0 e 1
❌  Camadas intermédias (causa vanishing gradient)
❌  Regressão
'''

# Vanishing Gradient Problem: O problema do gradiente que desaparece ocorre quando os gradientes se tornam muito pequenos durante a retropropagação, o que pode impedir que a rede aprenda efetivamente. A função sigmoid é particularmente suscetível a esse problema, especialmente em camadas intermediárias, porque seus gradientes podem se tornar muito pequenos para valores de entrada grandes ou pequenos.


# O gradiente da sigmoid é máximo em z=0 e fica muito
# pequeno para valores grandes ou pequenos de z

def sigmoid_derivada(z):
    s = sigmoid(z)
    return s * (1 - s)

print(sigmoid_derivada(0))    # 0.25  → máximo
print(sigmoid_derivada(5))    # 0.006 → muito pequeno
print(sigmoid_derivada(10))   # 0.000 → quase zero

# Em redes profundas, gradientes pequenos multiplicados
# por muitas camadas ficam praticamente zero
# → a rede deixa de aprender nas primeiras camadas


'''
Qual é o nosso Problema? - Vanishing Gradient 
Como é que vamos Resolver? - Usar outras funções de ativação (ReLU, Leaky ReLU, etc.) que não sofrem tanto com o problema do gradiente que desaparece.
'''

def relu(z):
    return np.maximum(0, z)

# Testar
valores = [-10, -3, -1, 0, 1, 3, 10]
for v in valores:
    print(f"relu({v:>4}) = {relu(v):.4f}")

#Se for valor for Negativo devolve 0 , se o valor for positivo devolve o Próprio Valor  

''' Quando Usar ReLU?
✅  Camadas intermédias (hidden layers) — sempre
✅  Redes profundas
✅  Processamento de imagens
❌  Última camada (não dá probabilidades)
❌  Dados com valores negativos importantes
'''

# Dying ReLU Problem: O problema do ReLU morto ocorre quando um neurônio ReLU recebe um valor de entrada negativo durante o treinamento, o que faz com que a saída seja zero. Se isso acontecer repetidamente, o neurônio pode "morrer", ou seja, ele nunca mais ativa e não contribui para o aprendizado da rede. Isso pode ser problemático, especialmente em redes profundas, onde muitos neurônios podem acabar mortos, reduzindo a capacidade da rede de aprender padrões complexos.

# Se o Z for negativo , o ReLU devolve 0 e o neurônio não ativa
# Logo não aprende 

# E resolvemos com o Leaky ReLU, que tem um pequeno gradiente para valores negativos

def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)

# Em vez de devolver 0 para negativos
# devolve um valor muito pequeno (0.01 * z)
print(leaky_relu(-5))   # -0.05  → não morre completamente
print(leaky_relu(5))    #  5.0

'''Quando Usar Leaky ReLU?
✅  Camadas intermédias (alternativa à ReLU)
✅  Redes recorrentes (RNNs) — muito usada
✅  Quando os dados têm valores negativos
❌  Redes muito profundas (também tem vanishing gradient)
'''

# Agora temos o tahn, que é uma função de ativação que transforma os valores em um intervalo entre -1 e 1, o que pode ser útil para dados que têm valores negativos importantes.

def tanh(z):
    return np.tanh(z)   # já existe no numpy

# Testar
valores = [-10, -3, -1, 0, 1, 3, 10]
for v in valores:
    print(f"tanh({v:>4}) = {tanh(v):.4f}")

# tanh(-10) = -1.0000
# tanh( -3) = -0.9951
# tanh( -1) = -0.7616
# tanh(  0) =  0.0000  ← centrada no zero
# tanh(  1) =  0.7616
# tanh(  3) =  0.9951
# tanh( 10) =  1.0000

'''Quand o Usar Tanh?
✅  Camadas intermédias (alternativa à ReLU)
✅  Redes recorrentes (RNNs) — muito usada
✅  Quando os dados têm valores negativos
❌  Redes muito profundas (também tem vanishing gradient)
'''

#SOFTMAX: Transforma um vetor de valores em probabilidades, que somam 1.

def softmax(z):
    exp_z = np.exp(z - np.max(z))   # estabilidade numérica
    return exp_z / exp_z.sum()

# Exemplo — classificar 3 animais
z = np.array([3.0, 1.0, 0.5])      # scores para [cão, gato, pássaro]
prob = softmax(z)

print(prob)
# [0.836  0.114  0.069]  ← somam 1.0 (100%)
#    ↑       ↑      ↑
#  84%     11%     7%
# cão     gato  pássaro

print(prob.sum())   # 1.0 ← sempre

''' Quando Usar Softmax?
✅  Última camada de classificação múltipla
✅  Quando tens 3 ou mais classes
✅  Quando precisas de probabilidades que somam 1
❌  Classificação binária (usa sigmoid)
❌  Regressão
❌  Camadas intermédias
'''


'''
z        sigmoid      relu         tanh
─────────────────────────────────────────────
-3.0     0.0474       0.0000       -0.9951
-1.0     0.2689       0.0000       -0.7616
 0.0     0.5000       0.0000        0.0000
 1.0     0.7311       1.0000        0.7616
 3.0     0.9526       3.0000        0.9951
'''

#Sigmoid : basicamente são valores que vão de 0 a 1, e o ponto de decisão é 0.5
#ReLU : é 0 para valores negativos e o próprio valor para valores positivos, ou seja, não tem um ponto de decisão específico, mas é útil para camadas intermediárias.
#Tanh : é semelhante à sigmoid, mas os valores vão de -1 a 1, e o ponto de decisão é 0.

