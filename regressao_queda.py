import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Dados do experimento
altura = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5]).reshape(-1, 1)  # Altura em metros
tempo = np.array([0.45, 0.52, 0.58, 0.61, 0.68, 0.72])  # Tempo em segundos

# Realizar a regressão linear
regressor = LinearRegression()
regressor.fit(altura, tempo)

# Coeficientes da reta
coeficiente_angular = regressor.coef_[0]
coeficiente_linear = regressor.intercept_

# Fazer previsões com base no modelo
altura_predita = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5]).reshape(-1, 1)
tempo_predito = regressor.predict(altura_predita)

# Plotar os dados e a reta de regressão
plt.scatter(altura, tempo, color='blue', label='Dados Experimentais')
plt.plot(altura_predita, tempo_predito, color='red', label='Regressão Linear')
plt.xlabel('Altura (metros)')
plt.ylabel('Tempo (segundos)')
plt.legend()
plt.grid(True)
plt.show()

# Imprimir os coeficientes da reta de regressão
print(f"Coeficiente Angular: {coeficiente_angular}")
print(f"Coeficiente Linear: {coeficiente_linear}")
