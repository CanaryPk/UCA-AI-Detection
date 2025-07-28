# Primeros pasos para la segmentación de los vehículos en cada frame.
# El programa guarda en la carpeta "recortes" cada uno de los vehículos del vídeo en forma de carpeta. En cada carpeta se 
# guardan los recortes del respectivo vehículo en cada uno de los frames del vídeo.

import cv2
import os
from ultralytics import YOLO
from collections import defaultdict

# Cargamos el modelo YOLOv11
model = YOLO("yolo11n.pt")
clases = model.names

# Ruta y nombre del video
video = "../001_DAY_TXT_CLEAR_BRAKECHECK/1.mp4"
nombre_video = os.path.splitext(os.path.basename(video))[0]

# Carpeta para guardar los recortes de cada vehículo
carpetaRecortes = f"recortes/{nombre_video}"
os.makedirs(carpetaRecortes, exist_ok=True)

# Iniciamos la captura del video
cap = cv2.VideoCapture(video)
frame_id = 0

while cap.isOpened():
    leido, frame = cap.read()
    if not leido:
        break

    resultados = model.track(frame, persist=True)
    frame_id += 1  # Frame actual

    frame_sin_anotaciones = frame.copy()

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
            recorte = frame_sin_anotaciones[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
            if recorte.size == 0:
                continue  # Saltar si el recorte está fuera del frame

            # Creamos carpeta para el ID si no existe
            carpeta_id = os.path.join(carpetaRecortes, f"id{id_track}")
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
