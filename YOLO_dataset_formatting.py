# Prueba de uso del framework YOLOv11.
# El programa simplemente recibe un vídeo, detecta cada uno de los vehículos en él y les asigna una ID y demás metadatos. Cada vehículo es
# recuadrado e identificado con su ID correspondiente.

import cv2
from ultralytics import YOLO

model = YOLO("yolo11n.pt") # Cargamos el modelo YOLO.
clases = model.names
cap = cv2.VideoCapture("../videos/002_DAY_NOTXT_CLEAR_BRAKECHECK/Genius_brake_checks_a_semi_in_Nashville_TN.mp4")

leido, frame = cap.read() # Leemos frame uno a uno.
while cap.isOpened() and leido:
    resultados = model.track(frame, persist=True) # Detección y tracking.
    #print(resultados) # Cada uno de los "boxes" contiene las cajas delimitadoras, IDs,
                      # nombres de clases, etc para cada objeto detectado.

    if resultados and resultados[0].boxes is not None: # Procesamos detecciones si y sólo si YOLO devolvió resultados y esos resultados contienen cajas (objetos).
        cajas = resultados[0].boxes.xyxy.cpu() # coordenadas xy de comienzo y fin de la "bounding box" del objeto detectado.
        ids = resultados[0].boxes.id.int().cpu().tolist()
        indices_clase = resultados[0].boxes.cls.int().cpu().tolist()
        confidences = resultados[0].boxes.conf.cpu()

        # iteramos cada objeto detectado.
        for caja, id_track, id_clase, conf in zip(cajas, ids, indices_clase, confidences):
            x1, y1, x2, y2 = map(int, caja)
            nombre_clase = clases[id_clase]

            cv2.putText(frame, f"ID: {id_track} {nombre_clase}", (x1, y1-10),
                        cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 255), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Dynamic vehicle detection with YOLOv11", frame)
    leido, frame = cap.read()

    # Para salir, pulsamos "q".
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
