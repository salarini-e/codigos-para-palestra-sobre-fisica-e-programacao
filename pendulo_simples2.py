import numpy as np
import matplotlib.pyplot as plt

# Parâmetros do pêndulo
g = 9.81  # Aceleração da gravidade (m/s^2)
L = 1.0   # Comprimento do pêndulo (metros)

# Solução analítica aproximada do ângulo do pêndulo θ(t)
def pendulum_analytical(t, theta0):
    omega = np.sqrt(g / L)
    return theta0 * np.cos(omega * t)

# Definição da equação diferencial do pêndulo
def pendulum_ode(t, y):
    theta, theta_dot = y
    dtheta_dt = theta_dot
    dtheta_dot_dt = -(g / L) * np.sin(theta)
    return [dtheta_dt, dtheta_dot_dt]

# Método de Runge-Kutta de quarta ordem
def runge_kutta4(f, t, y, dt):
    k1 = f(t, y)
    k2 = f(t + 0.5*dt, [y[i] + 0.5*dt*k1[i] for i in range(len(y))])
    k3 = f(t + 0.5*dt, [y[i] + 0.5*dt*k2[i] for i in range(len(y))])
    k4 = f(t + dt, [y[i] + dt*k3[i] for i in range(len(y))])
    return [y[i] + (dt/6) * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]) for i in range(len(y))]

# Condições iniciais
theta0 = 0.2  # Ângulo inicial (radianos)
theta_dot0 = 0.0  # Velocidade angular inicial (radianos/s)
y0 = [theta0, theta_dot0]

# Configuração do intervalo de tempo e passo de integração
t0 = 0.0
t_final = 6*np.pi
dt = 0.01

# Listas para armazenar os resultados numéricos
times_numeric = [t0]
thetas_numeric = [theta0]

# Integração numérica usando o método de Runge-Kutta de quarta ordem
t = t0
y = y0
while t <= t_final:
    y = runge_kutta4(pendulum_ode, t, y, dt)
    t += dt
    times_numeric.append(t)
    thetas_numeric.append(y[0])

# Solução analítica aproximada para o ângulo do pêndulo θ(t)
times_analytical = np.linspace(t0, t_final, 1000)
thetas_analytical = pendulum_analytical(times_analytical, theta0)

# Plot dos resultados
plt.plot(times_numeric, thetas_numeric, label='Solução Numérica')
plt.plot(times_analytical, thetas_analytical, label='Solução Analítica (aproximada)')
plt.xlabel('Tempo (s)')
plt.ylabel('Ângulo (radianos)')
plt.title('Pêndulo Simples - Ângulo vs. Tempo')
plt.legend()
plt.grid(True)
plt.show()
