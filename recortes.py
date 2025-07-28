# Primeros pasos para la segmentación de los vehículos en cada frame.
# El programa guarda en la carpeta "recortes" cada uno de los vehículos del vídeo en forma de carpeta. En cada carpeta se 
# guardan los recortes del respectivo vehículo en cada uno de los frames del vídeo.

import cv2
import os
from ultralytics import YOLO

# Cargamos el modelo YOLOv11.
model = YOLO("yolo11n.pt")
clases = model.names

# Ruta del video y nombre base del video.
video_path = "../001_DAY_TXT_CLEAR_BRAKECHECK/1.mp4"
nombre_video = os.path.splitext(os.path.basename(video_path))[0]

# Creamos carpeta raíz para guardar recortes.
output_root = f"recortes/{nombre_video}"
os.makedirs(output_root, exist_ok=True)

# Iniciar captura de video.
cap = cv2.VideoCapture(video_path)
frame_id = 0

while cap.isOpened():
    leido, frame = cap.read()
    if not leido:
        break

    frame_id += 1
    frame_sin_anotaciones = frame.copy() # Copia del frame original para no guardar las «bounding boxes».

    resultados = model.track(frame, persist=True)

    if resultados[0].boxes.data is not None:
        cajas = resultados[0].boxes.xyxy.cpu()
        ids = resultados[0].boxes.id.int().cpu().tolist()
        indices_clase = resultados[0].boxes.cls.int().cpu().tolist()
        confidences = resultados[0].boxes.conf.cpu()

        for caja, id_track, id_clase, conf in zip(cajas, ids, indices_clase, confidences):
            x1, y1, x2, y2 = map(int, caja)

            # Recorte limpio del vehículo.
            recorte = frame_sin_anotaciones[max(0, y1):max(0, y2), max(0, x1):max(0, x2)] 
            if recorte.size == 0: # Si el recorte está vacío, ignoramos.
                continue

            # Guardamos el recorte en la carpeta correspondiente al ID del vehículo.
            carpeta_id = os.path.join(output_root, f"id{id_track}")
            os.makedirs(carpeta_id, exist_ok=True)

            # Guardamos el recorte con el frame actual como nombre.
            nombre_imagen = f"frame_{frame_id:05}.jpg"
            ruta_guardado = os.path.join(carpeta_id, nombre_imagen)
            cv2.imwrite(ruta_guardado, recorte)

cap.release()
