import cv2
import mediapipe as mp
import numpy as np
import os

# Initialisation de MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

input_folder = "C:/Users/benro/PycharmProjects/PythonProject/mpii_project/images"
output_folder = "C:/Users/benro/PycharmProjects/PythonProject/mpii_project/image_annote"

# Crée le dossier de sortie s'il n'existe pas
os.makedirs(output_folder, exist_ok=True)

# Liste des fichiers images dans le dossier (jpeg, jpg, png)
valid_extensions = (".jpg", ".jpeg", ".png")

with mp_pose.Pose(static_image_mode=True) as pose:
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(valid_extensions):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Impossible de charger l'image : {filename}")
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                def get_point(name):
                    lm = landmarks[mp_pose.PoseLandmark[name].value]
                    return [lm.x, lm.y]

                angles = {}
                # Calcul angle coude gauche
                angles['Left Elbow'] = calculate_angle(
                    get_point('LEFT_SHOULDER'),
                    get_point('LEFT_ELBOW'),
                    get_point('LEFT_WRIST'))

                # Calcul angle tronc gauche (épaule - hanche - verticale)
                shoulder = get_point('LEFT_SHOULDER')
                hip = get_point('LEFT_HIP')
                vertical_ref = [hip[0], hip[1] + 0.3]
                angles['Trunk'] = calculate_angle(shoulder, hip, vertical_ref)

                print(f"{filename} - Angles détectés :", angles)

                # Dessin des articulations sur une copie
                annotated = image.copy()
                mp_drawing.draw_landmarks(annotated, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # Sauvegarde l'image annotée
                save_path = os.path.join(output_folder, filename)
                cv2.imwrite(save_path, annotated)

print("Traitement terminé.")
