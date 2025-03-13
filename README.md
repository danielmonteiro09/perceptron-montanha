# perceptron-montanha

#Exercício 2 - Prática de Inteligência Artificial: Decisão de Ir ao Parque

import numpy as np
from sklearn.linear_model import Perceptron

#Definição dos dados (features)
X = np.array([
    [0, 0, 0],
    [0, 1, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 0, 1],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 1]
])

#Saídas desejadas (rótulos)
y = np.array([0, 1, 1, 1, 0, 0, 0, 0])  

#Criando e treinando o Perceptron
modelo = Perceptron(max_iter=1000, tol=1e-3)  # 1000 iterações para garantir aprendizado
modelo.fit(X, y)

#Testando o modelo com novos exemplos
novos_exemplos = np.array([
    [1, 1, 0],  # Ensolarado, final de semana, não lotado → Esperado: 1 (Sim)
    [0, 0, 1],  # Não ensolarado, dia útil, lotado → Esperado: 0 (Não)
    [1, 0, 1]   # Ensolarado, dia útil, lotado → Esperado: 0 (Não)
])

#Fazendo previsões
previsoes = modelo.predict(novos_exemplos)

#Exibir resultados
for i, exemplo in enumerate(novos_exemplos):
    print(f"Entrada: {exemplo} → Decisão do Perceptron: {'Ir ao parque' if previsoes[i] == 1 else 'Não ir ao parque'}")
    



#Exercício 3 - Prática de Inteligência Artificial: Decisão sobre Comer Fora ou Cozinhar em Casa

import numpy as np
from sklearn.linear_model import Perceptron

#Definição dos dados de entrada (features)
X = np.array([
    [0, 1, 1, 1],
    [1, 0, 1, 1],
    [1, 1, 0, 1],
    [0, 0, 1, 0],
    [1, 1, 1, 0],
    [1, 0, 0, 1],
    [1, 0, 0, 0],
    [0, 0, 0, 1]
])

#Saídas desejadas (rótulos)
y = np.array([0, 1, 0, 1, 1, 1, 0, 0])  

#Criando e treinando o Perceptron
modelo = Perceptron(max_iter=1000, tol=1e-3)
modelo.fit(X, y)

#Testando o modelo com novos exemplos
novos_exemplos = np.array([
    [1, 1, 1, 1],  # Cansado, tem ingredientes, restaurante aberto, pagamento recente → Esperado: 1 (Comer fora)
    [0, 1, 0, 0],  # Não cansado, tem ingredientes, restaurante fechado, sem pagamento → Esperado: 0 (Cozinhar)
    [1, 0, 1, 0]   # Cansado, sem ingredientes, restaurante aberto, sem pagamento → Esperado: 1 (Comer fora)
])

#Fazendo previsões
previsoes = modelo.predict(novos_exemplos)

#Exibir resultados
for i, exemplo in enumerate(novos_exemplos):
    decisao = "Comer fora" if previsoes[i] == 1 else "Cozinhar em casa"
    print(f"Entrada: {exemplo} → Decisão do Perceptron: {decisao}")

