import numpy as np
import pandas as pd

# Gerar dados aleatórios para velocidade do vento (m/s)
velocidade_vento = np.random.uniform(5, 20, 100)

# Gerar dados aleatórios para intensidade de turbulência (%)
turbulencia = np.random.uniform(0.5, 5, 100)

# Criar um DataFrame com os dados simulados
dados_turbulencia = pd.DataFrame({
    'Velocidade do Vento (m/s)': velocidade_vento,
    'Intensidade de Turbulência (%)': turbulencia
})

# Exibir as primeiras linhas do DataFrame
print(dados_turbulencia.head())
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Gerar dados aleatórios para altura (metros)
altura = np.random.uniform(1.0, 10.0, 100)

# Calcular o tempo de queda aproximado usando a fórmula t = sqrt(2h/g)
# Onde g é a aceleração da gravidade (aproximadamente 9.81 m/s^2)
tempo = np.sqrt(2 * altura / 9.81)

# Adicionar um pouco de ruído aos dados de tempo para torná-los mais realistas
tempo = tempo + np.random.normal(0, 0.1, 100)

# Transformar os arrays em formato de coluna (necessário para usar com scikit-learn)
altura = altura.reshape(-1, 1)
tempo = tempo.reshape(-1, 1)

# Realizar a regressão linear
regressor = LinearRegression()
regressor.fit(altura, tempo)

# Coeficientes da reta
coeficiente_angular = regressor.coef_[0][0]
coeficiente_linear = regressor.intercept_[0]

# Fazer previsões com base no modelo
altura_predita = np.linspace(1.0, 10.0, 100).reshape(-1, 1)
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
