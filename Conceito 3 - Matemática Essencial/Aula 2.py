import numpy as np 
# O gradiente , Derivada em várias Dimensões 
# Uma neural network não tem apenas um peso , tem milhares ou até milhões. O gradiente é simplemente a derivada calculada para cada peso ao mesmo tempo 

gradiente = np.array([0.5, -0.3, 0.8])

# interpretação:
# w1 -> 0.5 : se aumentar o w1 aumenta o erro -> temode de diminuir o w1
# w2 -> -0.3 → aumentar w2 diminui o erro -> temos de aumentar w2
# w3 -> 0.8  → aumentar w3 aumenta o erro -> temos de diminuir w3

# O gradiente aponta sempre para a direção de maior subida. Por isso fazemos o contário, subtrímos para descer.

'''
Gradient Descent 
Agora que sabemos o que é o gradiente, o gradient Descent é simplesmente :

novo_epso = peso_atual - learning_rate * gradiente

'''

#peso atual 
w= 5.0 
lr = 0.1 # learning rate

print(f"{'Passo':<8} {'Peso':<12} {'Loss (w²)':<12} {'Gradiente'}")
print("-" * 45)

for passo in range(20):
    loss = w ** 2
    gradiente = 2 * w

w = w - lr * gradiente
print(f"{passo:<8} {w:<12.4f} {loss:<12.4f} {gradiente:.4f}")

