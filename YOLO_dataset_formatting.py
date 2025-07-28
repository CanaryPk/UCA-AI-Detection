import cv2
from ultralytics import YOLO
from collections import defaultdict

model = YOLO("yolo11n.pt") # Cargamos el modelo YOLO.

clases = model.names

cap = cv2.VideoCapture("../001_DAY_TXT_CLEAR_BRAKECHECK/1.mp4")
while cap.isOpened:
    leido, frame = cap.read() # Leemos frame uno a uno.
    if not leido: # Si no hay un frame para leer...
        break
    resultados = model.track(frame, persist=True) # Detección y tracking.
    #print(resultados) # Cada uno de los "boxes" contiene las cajas delimitadoras, IDs, 
                      # nombres de clases, etc para cada objeto detectado.

    if resultados[0].boxes.data is not None: 
        # Obtenemos las cajas detectadas, sus índices de clase, y sus IDs
        cajas = resultados[0].boxes.xyxy.cpu() # coordenadas xy del comienzo de la caja (primer xy) 
                                               # y del final de la caja (segundo xy).
        ids = resultados[0].boxes.id.int().cpu().tolist()
        indices_clase = resultados[0].boxes.cls.int().cpu().tolist()
        confidences = resultados[0].boxes.conf.cpu()

        # iteramos cada objeto detectado. En este ejemplo, el vehículo que realiza el comportamiento
        # peligroso tiene la ID 8.
        for caja, id_track, id_clase, conf in zip(cajas, ids, indices_clase, confidences):
            x1, y1, x2, y2 = map(int, caja)
            nombre_clase = clases[id_clase]

            cv2.putText(frame, f"ID: {id_track} {nombre_clase}", (x1, y1-10), 
                        cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 255), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    cv2.imshow("Seguimiento de vehículos con YOLOv11", frame)

    # Para salir, pulsamos "q".
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
