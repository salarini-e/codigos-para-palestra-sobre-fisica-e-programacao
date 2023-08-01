import numpy as np
import matplotlib.pyplot as plt

# Constantes
G = 6.67430e-11  # Constante gravitacional (m³/kg/s²)

# Massas (em kg)
mass_sun = 1.989e30
mass_earth = 5.972e24
mass_moon = 7.342e22

# Posições iniciais (em metros)
pos_sun = np.array([0, 0])
pos_earth = np.array([147e9, 0])
pos_moon = np.array([147e9 + 10 * 384.4e6, 0])  # Aumentando a distância da Lua

# Velocidades iniciais (em metros por segundo)
vel_sun = np.array([0.0, 0.0])
vel_earth = np.array([0.0, 30300.0])
vel_moon = np.array([0.0, 30300.0 + 1022.0])

# Lista para armazenar as posições para plotagem posterior
earth_traj = [pos_earth.copy()]
moon_traj = [pos_moon.copy()]

# Parâmetros da simulação
dt = 60 * 60  # Intervalo de tempo (1 hora em segundos)
num_steps = 365 * 24  # Número de passos de tempo (1 ano)

# Simulação usando o método de Runge-Kutta de 4ª ordem
for _ in range(num_steps):
    # Calcula as distâncias entre os corpos
    dist_earth_sun = np.linalg.norm(pos_earth - pos_sun)
    dist_moon_earth = np.linalg.norm(pos_moon - pos_earth)
    dist_moon_sun = np.linalg.norm(pos_moon - pos_sun)

    # Calcula as acelerações devido à gravidade
    acc_earth = -G * mass_sun / dist_earth_sun**3 * (pos_earth - pos_sun)
    acc_moon = -G * mass_earth / dist_moon_earth**3 * (pos_moon - pos_earth) - G * mass_sun / dist_moon_sun**3 * (pos_moon - pos_sun)

    # Atualiza as velocidades e posições usando o método de Runge-Kutta de 4ª ordem
    k1_e = dt * vel_earth
    k1_m = dt * vel_moon
    l1_e = dt * acc_earth
    l1_m = dt * acc_moon

    k2_e = dt * (vel_earth + 0.5 * l1_e)
    k2_m = dt * (vel_moon + 0.5 * l1_m)
    l2_e = dt * (-G * mass_sun / np.linalg.norm(pos_earth + 0.5 * k1_e - pos_sun)**3 * (pos_earth + 0.5 * k1_e - pos_sun))
    l2_m = dt * (-G * mass_earth / np.linalg.norm(pos_moon + 0.5 * k1_m - pos_earth)**3 * (pos_moon + 0.5 * k1_m - pos_earth) -
                 G * mass_sun / np.linalg.norm(pos_moon + 0.5 * k1_m - pos_sun)**3 * (pos_moon + 0.5 * k1_m - pos_sun))

    k3_e = dt * (vel_earth + 0.5 * l2_e)
    k3_m = dt * (vel_moon + 0.5 * l2_m)
    l3_e = dt * (-G * mass_sun / np.linalg.norm(pos_earth + 0.5 * k2_e - pos_sun)**3 * (pos_earth + 0.5 * k2_e - pos_sun))
    l3_m = dt * (-G * mass_earth / np.linalg.norm(pos_moon + 0.5 * k2_m - pos_earth)**3 * (pos_moon + 0.5 * k2_m - pos_earth) -
                 G * mass_sun / np.linalg.norm(pos_moon + 0.5 * k2_m - pos_sun)**3 * (pos_moon + 0.5 * k2_m - pos_sun))

    k4_e = dt * (vel_earth + l3_e)
    k4_m = dt * (vel_moon + l3_m)
    l4_e = dt * (-G * mass_sun / np.linalg.norm(pos_earth + k3_e - pos_sun)**3 * (pos_earth + k3_e - pos_sun))
    l4_m = dt * (-G * mass_earth / np.linalg.norm(pos_moon + k3_m - pos_earth)**3 * (pos_moon + k3_m - pos_earth) -
                 G * mass_sun / np.linalg.norm(pos_moon + k3_m - pos_sun)**3 * (pos_moon + k3_m - pos_sun))

    vel_earth += (l1_e + 2 * l2_e + 2 * l3_e + l4_e) / 6
    vel_moon += (l1_m + 2 * l2_m + 2 * l3_m + l4_m) / 6
    pos_earth += (k1_e + 2 * k2_e + 2 * k3_e + k4_e) / 6
    pos_moon += (k1_m + 2 * k2_m + 2 * k3_m + k4_m) / 6

    # Armazena as posições para a plotagem
    earth_traj.append(pos_earth.copy())
    moon_traj.append(pos_moon.copy())

# Convertendo para unidades astronômicas (UA) para visualização
earth_traj = np.array(earth_traj) / 1.496e11
moon_traj = np.array(moon_traj) / 1.496e11

# Plotagem da órbita da Terra e da Lua
plt.figure(figsize=(10, 8))  # Ajustando o tamanho do gráfico
plt.plot(0, 0, 'yo', markersize=12, label='Sol')
plt.plot(earth_traj[:, 0], earth_traj[:, 1], 'b', label='Terra')
plt.plot(moon_traj[:, 0], moon_traj[:, 1], 'g', label='Lua')
plt.xlabel('Distância em UA')
plt.ylabel('Distância em UA')
plt.title('Órbita da Terra e da Lua em torno do Sol')
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.show()
