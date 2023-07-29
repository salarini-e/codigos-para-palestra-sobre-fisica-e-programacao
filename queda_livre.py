import matplotlib.pyplot as plt
import cv2
import numpy as np

def detect_motion(video_path):
    # Abre o vídeo
    cap = cv2.VideoCapture(video_path)

    # Verifica se o vídeo foi aberto corretamente
    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        return

    # Lê os dois primeiros frames
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    trajectory = []  # Lista para armazenar a trajetória do objeto

    while cap.isOpened():
        # Se não foi possível ler o frame, encerra o loop
        if not ret:
            break

        # Aplicar filtro de média para suavizar o frame
        frame1_smooth = cv2.medianBlur(frame1, 5)
        frame2_smooth = cv2.medianBlur(frame2, 5)

        # Aplicar filtro Gaussiano para suavizar ainda mais o frame
        frame1_smooth = cv2.GaussianBlur(frame1_smooth, (9, 9), 0)
        frame2_smooth = cv2.GaussianBlur(frame2_smooth, (9, 9), 0)

        # Calcula a diferença entre os frames suavizados
        diff = cv2.absdiff(frame1_smooth, frame2_smooth)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, threshold = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)

        # Faz a detecção de contornos na imagem binarizada
        contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Procura o maior contorno (supondo que seja o objeto em queda)
        max_contour = None
        max_area = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                max_contour = contour

        if max_contour is not None:
            # Calcula o centro do objeto como a média das coordenadas dos contornos
            M = cv2.moments(max_contour)
            if M["m00"] != 0:
                center_x = int(M["m10"] / M["m00"])
                center_y = int(M["m01"] / M["m00"])
                trajectory.append((center_x, center_y))

        # Mostra o vídeo com os retângulos
        cv2.imshow("Motion Detection", frame1)

        # Atualiza os frames
        frame1 = frame2
        ret, frame2 = cap.read()

        # Verifica se o usuário pressionou a tecla 'q' para sair
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libera os recursos
    cap.release()
    cv2.destroyAllWindows()

    return trajectory

def calculate_velocity_and_acceleration(trajectory, time_interval):
    # Calcula a velocidade e a aceleração usando as posições da trajetória
    velocities = []
    accelerations = []

    for i in range(1, len(trajectory)):
        x1, y1 = trajectory[i-1]
        x2, y2 = trajectory[i]

        # Calcula a velocidade usando a fórmula da velocidade média
        velocity_x = (x2 - x1) / time_interval
        velocity_y = (y2 - y1) / time_interval
        velocity = np.sqrt(velocity_x**2 + velocity_y**2)

        velocities.append(velocity)

        if i > 1:
            # Calcula a aceleração usando a fórmula da aceleração média
            acceleration_x = (velocity_x - velocities[i-2]) / time_interval
            acceleration_y = (velocity_y - velocities[i-2]) / time_interval
            acceleration = np.sqrt(acceleration_x**2 + acceleration_y**2)

            accelerations.append(acceleration)

    return velocities, accelerations

# Caminho para o arquivo de vídeo (substitua pelo caminho correto)
video_path = "objeto_em_queda.mp4"

# Tempo entre cada frame em segundos (ajuste conforme necessário)
time_interval = 1.0 / 30.0

# Realiza a detecção de movimento e obtém a trajetória do objeto
trajectory = detect_motion(video_path)

# Calcula a velocidade e a aceleração do objeto usando a trajetória
velocities, accelerations = calculate_velocity_and_acceleration(trajectory, time_interval)

# Imprime as velocidades e acelerações
print("Velocidades:")
print(velocities)
print("\nAcelerações:")
print(accelerations)

# Cria uma lista com os instantes de tempo (em segundos)
time_points = [i * time_interval for i in range(len(velocities))]

# Plota o gráfico da velocidade
plt.plot(time_points, velocities)
plt.xlabel("Tempo (s)")
plt.ylabel("Velocidade (unidades por segundo)")
plt.title("Gráfico da Velocidade do Objeto em Queda")
plt.grid(True)
plt.show()



