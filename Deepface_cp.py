import cv2
import os
from deepface import DeepFace

def face_detect(input_folder, output_folder, db_path):

    frame_files = [f for f in os.listdir(input_folder) if f.endswith(".jpg")]
    frame_files.sort()

    os.makedirs(output_folder, exist_ok=True)

    for frame_file in frame_files:
        frame_path = os.path.join(input_folder, frame_file)
        frame = cv2.imread(frame_path)

        if frame is None:
            print(f"Error reading frame {frame_file}")
            continue

        try:
    
            faces = DeepFace.extract_faces(img_path=frame_path, detector_backend='retinaface', enforce_detection=False)
            if faces:
                for face in faces:
    
                    facial_area = face['facial_area']
                    x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
                    detected_face = frame[y:y+h, x:x+w]

     
                    result = DeepFace.find(img_path=detected_face, db_path=db_path, detector_backend='retinaface', model_name='ArcFace', enforce_detection=False)
                    if result and len(result[0]) > 0:
              
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        print(f"Detected target face in {frame_file} at ({x}, {y}), size ({w}, {h})")


            output_frame_path = os.path.join(output_folder, frame_file)
            cv2.imwrite(output_frame_path, frame)
            print(f"Saved frame with detected faces to {output_frame_path}")

        except Exception as e:
            print(f"Error processing frame {frame_file}: {e}")


frame_file = r"images\frames"
detect_file = r"images\detect"
db_path = r"images\images"


face_detect(frame_file, detect_file, db_path)