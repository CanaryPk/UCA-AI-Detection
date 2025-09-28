# First steps in video segmentation, for saving car snippets in every frame.
# The following program saves each vehicle in the video as a folder in the "recortes" folder.
# Each folder stores the clippings of the respective vehicle in each frame of the video.

import cv2
import os
from ultralytics import YOLO

# Load our model with YOLOv11.
model = YOLO("yolo11n.pt")

base_directory = "../videos"
for base_folder in os.listdir(base_directory):
    if base_folder != "recortes":
        # Path to the original video, video name and folder.
        video_path = os.path.join(base_directory, base_folder)
        video_name = os.listdir(video_path)[0]
        video_path = os.path.join(video_path, video_name)

        # Create root directory to save car snippets.
        output_path = f"../videos/recortes/{base_folder}"
        os.makedirs(output_path, exist_ok=True)

        # Start video capture.
        cap = cv2.VideoCapture(video_path)
        frame_id = 0

        read, frame = cap.read()
        while cap.isOpened() and read:
            print("Vuelta: ", frame_id)
            print("Ruta: ", output_path)
            frame_id += 1
            clean_frame = frame.copy() # Copy of the original frame, without bounding boxes.

            results = model.track(frame, persist=True)
            if results and results[0].boxes is not None and len(results[0].boxes) > 0 and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu()
                ids = results[0].boxes.id.int().cpu().tolist()

                for box, id_track in zip(boxes, ids):
                    x1, y1, x2, y2 = map(int, box)

                    # Clean snippet of the vehicle
                    snippet = clean_frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                    if snippet.size != 0: # If the current clipping is NOT empty...
                        # Save the snippet in the folder corresponding to its vehicle ID.
                        folder_id = os.path.join(output_path, f"id{id_track}")
                        os.makedirs(folder_id, exist_ok=True)

                        # Guardamos el recorte con el frame actual como nombre.
                        img_name = f"frame_{frame_id}.jpg"
                        save_path = os.path.join(folder_id, img_name)
                        cv2.imwrite(save_path, snippet)
            read, frame = cap.read()
        cap.release()
