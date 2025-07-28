import cv2
import os
from ultralytics import YOLO
from collections import defaultdict

# Cargar el modelo YOLOv11
model = YOLO("yolo11n.pt")
clases = model.names

# Ruta del video y nombre base del video
video_path = "../001_DAY_TXT_CLEAR_BRAKECHECK/1.mp4"
nombre_video = os.path.splitext(os.path.basename(video_path))[0]

# Crear carpeta raíz para guardar recortes
output_root = f"recortes/{nombre_video}"
os.makedirs(output_root, exist_ok=True)

# Iniciar captura de video
cap = cv2.VideoCapture(video_path)
frame_id = 0

while cap.isOpened():
    leido, frame = cap.read()
    if not leido:
        break

    resultados = model.track(frame, persist=True)
    frame_id += 1  # Número de frame actual

    if resultados[0].boxes.data is not None:
        cajas = resultados[0].boxes.xyxy.cpu()
        ids = resultados[0].boxes.id.int().cpu().tolist()
        indices_clase = resultados[0].boxes.cls.int().cpu().tolist()
        confidences = resultados[0].boxes.conf.cpu()

        for caja, id_track, id_clase, conf in zip(cajas, ids, indices_clase, confidences):
            x1, y1, x2, y2 = map(int, caja)
            nombre_clase = clases[id_clase]

            # Dibujar en el frame
            cv2.putText(frame, f"ID: {id_track} {nombre_clase}", (x1, y1-10),
                        cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 255), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Recorte del objeto
            recorte = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
            if recorte.size == 0:
                continue  # Saltar si el recorte está fuera del frame

            # Crear carpeta para este ID si no existe
            carpeta_id = os.path.join(output_root, f"vehiculo{id_track}")
            os.makedirs(carpeta_id, exist_ok=True)

            # Guardar imagen del recorte
            nombre_imagen = f"frame_{frame_id:05}.jpg"
            ruta_guardado = os.path.join(carpeta_id, nombre_imagen)
            cv2.imwrite(ruta_guardado, recorte)

    # Mostrar el frame en tiempo real
    cv2.imshow("Seguimiento de vehículos con YOLOv11", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
