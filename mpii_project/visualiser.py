import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import logging

# Désactiver les avertissements TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Initialisation de MediaPipe Pose et Drawing
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, min_tracking_confidence=0.3)

# Fonction pour calculer l'angle
def calculate_angle(point1, point2, point3):
    """Calcule l'angle en degrés entre trois points (point2 est le sommet)."""
    a = np.array(point1)
    b = np.array(point2)
    c = np.array(point3)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

# Évaluation OWAS
def evaluate_owas(trunk_angle, left_arm_angle, right_arm_angle, left_knee_angle, right_knee_angle):
    """Évalue la posture selon OWAS."""
    if trunk_angle <= 20:
        trunk_score = 1  # Droite
    elif trunk_angle <= 60:
        trunk_score = 2  # Flexion modérée
    else:
        trunk_score = 4  # Flexion sévère
    arm_score = 1
    if left_arm_angle > 90 or right_arm_angle > 90:
        arm_score = 2  # Un ou deux bras au-dessus
    leg_score = 1
    if left_knee_angle > 120 or right_knee_angle > 120:
        leg_score = 2  # Accroupi
    owas_code = f"{trunk_score}-{arm_score}-{leg_score}"
    risk_level = 1 if trunk_score == 1 and arm_score == 1 and leg_score == 1 else 2
    return owas_code, risk_level

# Évaluation RULA
def evaluate_rula(trunk_angle, neck_angle, left_arm_angle, left_forearm_angle, right_arm_angle, right_forearm_angle):
    """Évalue la posture selon RULA (simplifié)."""
    if trunk_angle <= 20:
        trunk_score = 1
    elif trunk_angle <= 60:
        trunk_score = 3
    else:
        trunk_score = 5
    if neck_angle <= 10:
        neck_score = 1
    elif neck_angle <= 20:
        neck_score = 2
    else:
        neck_score = 3
    left_arm_score = 2 if left_arm_angle > 20 else 1
    right_arm_score = 2 if right_arm_angle > 20 else 1
    left_forearm_score = 2 if left_forearm_angle > 60 else 1
    right_forearm_score = 2 if right_forearm_angle > 60 else 1
    group_a = max(left_arm_score, right_arm_score) + max(left_forearm_score, right_forearm_score)
    group_b = trunk_score + neck_score
    return min(max(group_a + group_b - 1, 1), 7)

# Évaluation REBA
def evaluate_reba(trunk_angle, neck_angle, left_leg_angle, right_leg_angle, left_arm_angle, left_forearm_angle, right_arm_angle, right_forearm_angle):
    """Évalue la posture selon REBA (simplifié)."""
    if trunk_angle <= 20:
        trunk_score = 1
    elif trunk_angle <= 60:
        trunk_score = 2
    else:
        trunk_score = 4
    if neck_angle <= 20:
        neck_score = 1
    else:
        neck_score = 2
    leg_score = 1
    if left_leg_angle > 60 or right_leg_angle > 60:
        leg_score = 2
    left_arm_score = 2 if left_arm_angle > 20 else 1
    right_arm_score = 2 if right_arm_angle > 20 else 1
    left_forearm_score = 2 if left_forearm_angle > 60 else 1
    right_forearm_score = 2 if right_forearm_angle > 60 else 1
    group_a = trunk_score + neck_score + leg_score
    group_b = max(left_arm_score, right_arm_score) + max(left_forearm_score, right_forearm_score)
    return min(max(group_a + group_b - 1, 1), 15)

# Dossier contenant les images
image_folder = "C:/Users/benro/PycharmProjects/PythonProject/mpii_project/images"
output_folder = "C:/Users/benro/PycharmProjects/PythonProject/mpii_project/annotated_images"
output_csv = "C:/Users/benro/PycharmProjects/PythonProject/mpii_project/posture_analysis.csv"
os.makedirs(output_folder, exist_ok=True)

# Vérifier si le dossier existe
if not os.path.exists(image_folder):
    print(f"Erreur : Le dossier '{image_folder}' n'existe pas. Veuillez vérifier le chemin.")
    exit(1)

# Liste pour stocker les résultats
output_results = []

# Parcourir toutes les images du dossier
for image_name in os.listdir(image_folder):
    if image_name.endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(image_folder, image_name)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Erreur de chargement de l'image : {image_name}")
            continue

        # Convertir l'image en RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape
        results = pose.process(image_rgb)
        annotated_image = image.copy()

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            # Convertir les coordonnées normalisées en pixels
            for landmark in landmarks:
                landmark.x *= w
                landmark.y *= h

            # Coordonnées des points clés
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            mid_hip = [(left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2]
            mid_shoulder = [(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x + landmarks[
                mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x) / 2,
                            (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y + landmarks[
                                mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y) / 2]
            mid_neck = [(landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x + landmarks[
                mp_pose.PoseLandmark.RIGHT_EAR.value].x) / 2,
                        (landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y + landmarks[
                            mp_pose.PoseLandmark.RIGHT_EAR.value].y) / 2]
            left_arm = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_forearm = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            right_arm = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_forearm = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

            # Calculer les angles
            trunk_angle = calculate_angle(mid_shoulder, mid_hip, [mid_hip[0], mid_hip[1] - 100])
            trunk_flexion = abs(180 - trunk_angle)
            neck_angle = calculate_angle(mid_shoulder, mid_neck, [mid_neck[0], mid_neck[1] - 100])
            neck_flexion = abs(180 - neck_angle)
            left_arm_angle = calculate_angle(mid_shoulder, left_arm, left_forearm)
            left_forearm_angle = calculate_angle(left_arm, left_forearm, left_wrist)
            right_arm_angle = calculate_angle(mid_shoulder, right_arm, right_forearm)
            right_forearm_angle = calculate_angle(right_arm, right_forearm, right_wrist)
            left_leg_angle = calculate_angle(left_hip, left_knee, left_ankle)
            right_leg_angle = calculate_angle(right_hip, right_knee, right_ankle)

            # Évaluations
            owas_code, owas_risk = evaluate_owas(trunk_flexion, left_arm_angle, right_arm_angle, left_leg_angle, right_leg_angle)
            rula_score = evaluate_rula(trunk_flexion, neck_flexion, left_arm_angle, left_forearm_angle, right_arm_angle, right_forearm_angle)
            reba_score = evaluate_reba(trunk_flexion, neck_flexion, left_leg_angle, right_leg_angle, left_arm_angle, left_forearm_angle, right_arm_angle, right_forearm_angle)

            # Classification du risque
            # Niveau de risque selon les 3 méthodes
            owas_risk_level = "Élevé" if owas_risk == 2 else "Faible"
            rula_risk_level = "Élevé" if rula_score >= 5 else "Modéré" if rula_score >= 3 else "Faible"
            reba_risk_level = "Élevé" if reba_score >= 7 else "Modéré" if reba_score >= 4 else "Faible"

            # Décision finale plus équilibrée
            if owas_risk_level == "Élevé" or rula_risk_level == "Élevé" or reba_risk_level == "Élevé":
                risk_classification = "À risque"
            elif rula_risk_level == "Modéré" or reba_risk_level == "Modéré":
                risk_classification = "Modérément risqué"
            else:
                risk_classification = "Non à risque"

            # Statut de la posture
            posture_status = f"OWAS: {owas_code} (Risk {owas_risk}), RULA: {rula_score}, REBA: {reba_score}"
            print(f"Image: {image_name} - "
                  f"Trunk: {trunk_flexion:.2f}°, Neck: {neck_flexion:.2f}°, "
                  f"Left Arm: {left_arm_angle:.2f}°, Right Arm: {right_arm_angle:.2f}°, "
                  f"Left Forearm: {left_forearm_angle:.2f}°, Right Forearm: {right_forearm_angle:.2f}°, "
                  f"Left Leg: {left_leg_angle:.2f}°, Right Leg: {right_leg_angle:.2f}° - {posture_status}")
            print(f"Classification du risque : {risk_classification}\n")

            output_results.append({
                "image_name": image_name,
                "trunk_angle": round(trunk_flexion, 2),
                "neck_angle": round(neck_flexion, 2),
                "left_arm_angle": round(left_arm_angle, 2),
                "right_arm_angle": round(right_arm_angle, 2),
                "left_forearm_angle": round(left_forearm_angle, 2),
                "right_forearm_angle": round(right_forearm_angle, 2),
                "left_leg_angle": round(left_leg_angle, 2),
                "right_leg_angle": round(right_leg_angle, 2),
                "owas_code": owas_code,
                "owas_risk": owas_risk,
                "rula_score": rula_score,
                "reba_score": reba_score,
                "posture_status": posture_status,
                "risk_classification": risk_classification
            })

            # Convertir les coordonnées en pixels pour annotation
            mid_hip_px = (int(mid_hip[0]), int(mid_hip[1]))
            mid_shoulder_px = (int(mid_shoulder[0]), int(mid_shoulder[1]))
            mid_neck_px = (int(mid_neck[0]), int(mid_neck[1]))
            vertical_px = (int(mid_hip[0]), int(mid_hip[1] - 50))
            left_elbow_px = (int(left_forearm[0]), int(left_forearm[1]))
            right_elbow_px = (int(right_forearm[0]), int(right_forearm[1]))
            left_knee_px = (int(left_knee[0]), int(left_knee[1]))
            right_knee_px = (int(right_knee[0]), int(right_knee[1]))

            # Dessiner les landmarks et lignes
            mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.line(annotated_image, mid_shoulder_px, mid_hip_px, (0, 255, 0), 2)  # Tronc
            cv2.line(annotated_image, mid_hip_px, vertical_px, (255, 0, 0), 2)  # Verticale
            cv2.line(annotated_image, mid_shoulder_px, mid_neck_px, (0, 0, 255), 2)  # Cou
            cv2.putText(annotated_image, f"Trunk: {trunk_flexion:.2f}°", (mid_hip_px[0], mid_hip_px[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(annotated_image, f"Neck: {neck_flexion:.2f}°", (mid_neck_px[0], mid_neck_px[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(annotated_image, f"L Arm: {left_arm_angle:.2f}°", (left_elbow_px[0], left_elbow_px[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            cv2.putText(annotated_image, f"R Arm: {right_arm_angle:.2f}°", (right_elbow_px[0], right_elbow_px[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            cv2.putText(annotated_image, f"L Leg: {left_leg_angle:.2f}°", (left_knee_px[0], left_knee_px[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.putText(annotated_image, f"R Leg: {right_leg_angle:.2f}°", (right_knee_px[0], right_knee_px[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.putText(annotated_image, f"{posture_status} | {risk_classification}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            # Sauvegarder l'image annotée
            cv2.imwrite(os.path.join(output_folder, f"annotated_{image_name}"), annotated_image)
        else:
            print(f"Image: {image_name} - Aucune pose détectée")
            output_results.append({
                "image_name": image_name,
                "trunk_angle": None,
                "neck_angle": None,
                "left_arm_angle": None,
                "right_arm_angle": None,
                "left_forearm_angle": None,
                "right_forearm_angle": None,
                "left_leg_angle": None,
                "right_leg_angle": None,
                "owas_code": "N/A",
                "owas_risk": "N/A",
                "rula_score": "N/A",
                "reba_score": "N/A",
                "posture_status": "Aucune pose détectée",
                "risk_classification": "N/A"
            })
            cv2.imwrite(os.path.join(output_folder, f"annotated_{image_name}"), annotated_image)

# Fermer le modèle MediaPipe
pose.close()

# Sauvegarder les résultats dans un fichier CSV
if output_results:
    df = pd.DataFrame(output_results)
    df.to_csv(output_csv, index=False)
    print(f"Fichier CSV '{output_csv}' généré avec succès !")
else:
    print("Aucun résultat à enregistrer. Vérifiez que le dossier contient des images valides.")

# Générer des histogrammes pour toutes les parties du corps
angle_types = [
    ("Trunk", [r["trunk_angle"] for r in output_results if r["trunk_angle"] is not None], "skyblue"),
    ("Neck", [r["neck_angle"] for r in output_results if r["neck_angle"] is not None], "lightgreen"),
    ("Left Arm", [r["left_arm_angle"] for r in output_results if r["left_arm_angle"] is not None], "lightcoral"),
    ("Right Arm", [r["right_arm_angle"] for r in output_results if r["right_arm_angle"] is not None], "salmon"),
    ("Left Forearm", [r["left_forearm_angle"] for r in output_results if r["left_forearm_angle"] is not None], "lightblue"),
    ("Right Forearm", [r["right_forearm_angle"] for r in output_results if r["right_forearm_angle"] is not None], "cyan"),
    ("Left Leg", [r["left_leg_angle"] for r in output_results if r["left_leg_angle"] is not None], "lightyellow"),
    ("Right Leg", [r["right_leg_angle"] for r in output_results if r["right_leg_angle"] is not None], "gold")
]

for angle_name, angles, color in angle_types:
    if angles:
        plt.hist(angles, bins=20, color=color, edgecolor='black')
        plt.xlabel(f'Angle de {angle_name} (degrés)')
        plt.ylabel("Nombre d'images")
        plt.title(f'Distribution des angles de {angle_name}')
        plt.savefig(os.path.join(output_folder, f'{angle_name.lower().replace(" ", "_")}_angle_distribution.png'))
        plt.close()