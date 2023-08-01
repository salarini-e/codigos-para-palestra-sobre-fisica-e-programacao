import cv2

def detect_motion(video_path, output_path):
    # Abre o vídeo
    cap = cv2.VideoCapture(video_path)

    # Verifica se o vídeo foi aberto corretamente
    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        return

    # Obtem as informações do vídeo
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define o codec e o objeto VideoWriter para salvar o vídeo
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Lê os dois primeiros frames
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    while cap.isOpened():
        # Se não foi possível ler o frame, encerra o loop
        if not ret:
            break

        # Calcula a diferença entre os frames
        diff = cv2.absdiff(frame1, frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, threshold = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)

        # Faz a detecção de contornos na imagem binarizada
        contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Desenha retângulos ao redor dos objetos em movimento
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Ajuste o valor do limiar conforme necessário
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Salva o frame com as detecções no vídeo de saída
        out.write(frame1)

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
    out.release()
    cv2.destroyAllWindows()

# Caminho para o arquivo de vídeo (substitua pelo caminho correto)
video_path = "objeto_em_queda.mp4"
output_path = "saida_com_detecao.mp4"

detect_motion(video_path, output_path)
