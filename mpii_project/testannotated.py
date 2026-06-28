# -*- coding: utf-8 -*-
import os
import csv
import cv2
import numpy as np
import mediapipe as mp

# =========================
# Configurations
# =========================
INPUT_FOLDER = "C:/Users/benro/PycharmProjects/PythonProject/mpii_project/images"
OUTPUT_FOLDER = "C:/Users/benro/PycharmProjects/PythonProject/mpii_project/image_annote"
CSV_OUTPUT = "C:/Users/benro/PycharmProjects/PythonProject/mpii_project/pose3_analysis.csv"
VALID_EXT = (".jpg", ".jpeg", ".png")

# Seuils ergonomiques (ajustables)
THRESHOLDS = {
    'Trunk_med': 20.0,   # degrés depuis la verticale
    'Trunk_hi':  60.0,
    'Neck_med':  20.0,
    'Neck_hi':   45.0,
    'Shoulder_med': 60.0,
    'Shoulder_hi':  90.0,
    'Elbow_low': 70.0,     # zone sûre ~ [70, 160]
    'Elbow_high': 160.0,
    'Knee_med':  150.0,    # >=150 sûr, 120-150 moyen, <120 élevé
    'Knee_hi':   120.0,
    'Hip_med':   150.0,    # >=150 sûr, 120-150 moyen, <120 élevé
    'Hip_hi':    120.0,
}

# =========================
# Utilitaires géométriques
# =========================
def unit(v, eps=1e-6):
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n

def safe_angle_abc(a, b, c, eps=1e-6):
    a, b, c = np.array(a, dtype=float), np.array(b, dtype=float), np.array(c, dtype=float)
    ba = a - b
    bc = c - b
    nba = np.linalg.norm(ba)
    nbc = np.linalg.norm(bc)
    if nba < eps or nbc < eps:
        return None
    cosv = np.dot(ba, bc) / (nba * nbc)
    cosv = np.clip(cosv, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosv)))

def angle_with_vertical(p_top, p_bottom):
    v = np.array(p_bottom, dtype=float) - np.array(p_top, dtype=float)
    v = unit(v)
    v_vert = np.array([0.0, 1.0], dtype=float)  # verticale image (y croissant vers le bas)
    cosv = np.clip(np.dot(v, v_vert), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosv)))  # 0° = vertical, 90° = horizontal

# =========================
# Évaluation des risques
# =========================
def evaluate_joint_risk(angles, thresholds):
    r = {}
    # Tronc (0° = droit; plus grand = incliné)
    trunk = angles.get('Trunk')
    if trunk is None:
        r['Trunk'] = 'NA'
    else:
        if trunk >= thresholds['Trunk_hi']:
            r['Trunk'] = 'High Risk'
        elif trunk >= thresholds['Trunk_med']:
            r['Trunk'] = 'Medium Risk'
        else:
            r['Trunk'] = 'Safe'

    # Cou
    neck = angles.get('Neck')
    if neck is None:
        r['Neck'] = 'NA'
    else:
        if neck >= thresholds['Neck_hi']:
            r['Neck'] = 'High Risk'
        elif neck >= thresholds['Neck_med']:
            r['Neck'] = 'Medium Risk'
        else:
            r['Neck'] = 'Safe'

    # Épaules (angle au niveau de l'épaule: 0 bas, grand = bras levé)
    for side in ['Left', 'Right']:
        val = angles.get(f'{side} Shoulder')
        key = f'{side} Shoulder'
        if val is None:
            r[key] = 'NA'
        else:
            if val >= thresholds['Shoulder_hi']:
                r[key] = 'High Risk'
            elif val >= thresholds['Shoulder_med']:
                r[key] = 'Medium Risk'
            else:
                r[key] = 'Safe'

    # Coudes (180 = tendu; zone sûre ~ [70,160])
    for side in ['Left', 'Right']:
        val = angles.get(f'{side} Elbow')
        key = f'{side} Elbow'
        if val is None:
            r[key] = 'NA'
        else:
            if val < 60 or val > 170:
                r[key] = 'High Risk'
            elif val < THRESHOLDS['Elbow_low'] or val > THRESHOLDS['Elbow_high']:
                r[key] = 'Medium Risk'
            else:
                r[key] = 'Safe'

    # Genoux (>=150 sûr; 120-150 moyen; <120 élevé)
    for side in ['Left', 'Right']:
        val = angles.get(f'{side} Knee')
        key = f'{side} Knee'
        if val is None:
            r[key] = 'NA'
        else:
            if val < THRESHOLDS['Knee_hi']:
                r[key] = 'High Risk'
            elif val < THRESHOLDS['Knee_med']:
                r[key] = 'Medium Risk'
            else:
                r[key] = 'Safe'

    # Hanches (>=150 sûr; 120-150 moyen; <120 élevé)
    for side in ['Left', 'Right']:
        val = angles.get(f'{side} Hip')
        key = f'{side} Hip'
        if val is None:
            r[key] = 'NA'
        else:
            if val < THRESHOLDS['Hip_hi']:
                r[key] = 'High Risk'
            elif val < THRESHOLDS['Hip_med']:
                r[key] = 'Medium Risk'
            else:
                r[key] = 'Safe'

    # Risque global: ignore les NA; si tout NA -> 'NA'
    vals = [v for v in r.values() if v != 'NA']
    if not vals:
        overall = 'NA'
    elif 'High Risk' in vals:
        overall = 'High Risk'
    elif 'Medium Risk' in vals:
        overall = 'Medium Risk'
    else:
        overall = 'Safe'
    return r, overall

# =========================
# Extraction angles depuis MediaPipe
# =========================
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def get_point(landmarks, name):
    lm = landmarks[mp_pose.PoseLandmark[name].value]
    return np.array([lm.x, lm.y], dtype=float), float(lm.visibility)

def compute_angles_from_landmarks(landmarks):
    # Points nécessaires
    L_SHO, v_ls = get_point(landmarks, 'LEFT_SHOULDER')
    R_SHO, v_rs = get_point(landmarks, 'RIGHT_SHOULDER')
    L_ELB, v_le = get_point(landmarks, 'LEFT_ELBOW')
    R_ELB, v_re = get_point(landmarks, 'RIGHT_ELBOW')
    L_WRI, v_lw = get_point(landmarks, 'LEFT_WRIST')
    R_WRI, v_rw = get_point(landmarks, 'RIGHT_WRIST')
    L_HIP, v_lh = get_point(landmarks, 'LEFT_HIP')
    R_HIP, v_rh = get_point(landmarks, 'RIGHT_HIP')
    L_KNE, v_lk = get_point(landmarks, 'LEFT_KNEE')
    R_KNE, v_rk = get_point(landmarks, 'RIGHT_KNEE')
    L_ANK, v_la = get_point(landmarks, 'LEFT_ANKLE')
    R_ANK, v_ra = get_point(landmarks, 'RIGHT_ANKLE')
    NOSE, v_n = get_point(landmarks, 'NOSE')

    # Centres
    vis_sho = (v_ls > 0.5 and v_rs > 0.5)
    vis_hip = (v_lh > 0.5 and v_rh > 0.5)
    m_sho = (L_SHO + R_SHO) / 2.0 if vis_sho else None
    m_hip = (L_HIP + R_HIP) / 2.0 if vis_hip else None

    angles = {}

    # Coudes
    if v_ls > 0.5 and v_le > 0.5 and v_lw > 0.5:
        angles['Left Elbow'] = safe_angle_abc(L_SHO, L_ELB, L_WRI)
    else:
        angles['Left Elbow'] = None
    if v_rs > 0.5 and v_re > 0.5 and v_rw > 0.5:
        angles['Right Elbow'] = safe_angle_abc(R_SHO, R_ELB, R_WRI)
    else:
        angles['Right Elbow'] = None

    # Épaules (angle épaule entre coude et hanche)
    if v_le > 0.5 and v_ls > 0.5 and v_lh > 0.5:
        angles['Left Shoulder'] = safe_angle_abc(L_ELB, L_SHO, L_HIP)
    else:
        angles['Left Shoulder'] = None
    if v_re > 0.5 and v_rs > 0.5 and v_rh > 0.5:
        angles['Right Shoulder'] = safe_angle_abc(R_ELB, R_SHO, R_HIP)
    else:
        angles['Right Shoulder'] = None

    # Genoux
    if v_lh > 0.5 and v_lk > 0.5 and v_la > 0.5:
        angles['Left Knee'] = safe_angle_abc(L_HIP, L_KNE, L_ANK)
    else:
        angles['Left Knee'] = None
    if v_rh > 0.5 and v_rk > 0.5 and v_ra > 0.5:
        angles['Right Knee'] = safe_angle_abc(R_HIP, R_KNE, R_ANK)
    else:
        angles['Right Knee'] = None

    # Hanches
    if v_ls > 0.5 and v_lh > 0.5 and v_lk > 0.5:
        angles['Left Hip'] = safe_angle_abc(L_SHO, L_HIP, L_KNE)
    else:
        angles['Left Hip'] = None
    if v_rs > 0.5 and v_rh > 0.5 and v_rk > 0.5:
        angles['Right Hip'] = safe_angle_abc(R_SHO, R_HIP, R_KNE)
    else:
        angles['Right Hip'] = None

    # Tronc (angle avec verticale du segment épaules->hanches)
    if m_sho is not None and m_hip is not None:
        angles['Trunk'] = angle_with_vertical(m_sho, m_hip)
    else:
        angles['Trunk'] = None

    # Cou (angle avec verticale du segment épaules->nez)
    if m_sho is not None and v_n > 0.5:
        angles['Neck'] = angle_with_vertical(m_sho, NOSE)
    else:
        angles['Neck'] = None

    # Visibilité moyenne utile pour le CSV
    vis_stats = {
        'vis_shoulder': float((v_ls + v_rs) / 2.0),
        'vis_hip': float((v_lh + v_rh) / 2.0),
        'vis_elbow': float((v_le + v_re) / 2.0),
        'vis_wrist': float((v_lw + v_rw) / 2.0),
        'vis_knee': float((v_lk + v_rk) / 2.0),
        'vis_ankle': float((v_la + v_ra) / 2.0),
        'vis_nose': float(v_n),
    }

    return angles, vis_stats

# =========================
# Pipeline images -> CSV
# =========================
def main_images():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    angle_cols = [
        'Neck', 'Trunk',
        'Left Shoulder', 'Right Shoulder',
        'Left Elbow', 'Right Elbow',
        'Left Hip', 'Right Hip',
        'Left Knee', 'Right Knee',
    ]

    # Flags de dépassement de seuils (0/1)
    flag_cols = [
        'F_Trunk_med', 'F_Trunk_hi',
        'F_Neck_med', 'F_Neck_hi',
        'F_ShoulderL_med', 'F_ShoulderL_hi',
        'F_ShoulderR_med', 'F_ShoulderR_hi',
        'F_ElbowL_outside', 'F_ElbowR_outside',
        'F_KneeL_med', 'F_KneeL_hi',
        'F_KneeR_med', 'F_KneeR_hi',
        'F_HipL_med', 'F_HipL_hi',
        'F_HipR_med', 'F_HipR_hi',
    ]

    vis_cols = ['vis_shoulder', 'vis_hip', 'vis_elbow', 'vis_wrist', 'vis_knee', 'vis_ankle', 'vis_nose']

    headers = ['Filename'] + angle_cols + flag_cols + vis_cols + ['Overall Risk']

    with open(CSV_OUTPUT, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        with mp_pose.Pose(static_image_mode=True) as pose:
            for filename in os.listdir(INPUT_FOLDER):
                if not filename.lower().endswith(VALID_EXT):
                    continue

                img_path = os.path.join(INPUT_FOLDER, filename)
                image = cv2.imread(img_path)
                if image is None:
                    print(f"❌ Impossible de charger l'image: {filename}")
                    continue

                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)

                if not results.pose_landmarks:
                    print(f"⚠️ Aucune pose détectée: {filename}")
                    row = [filename] + [None]*len(angle_cols) + [0]*len(flag_cols) + [0]*len(vis_cols) + ['NA']
                    writer.writerow(row)
                    continue

                angles, vis_stats = compute_angles_from_landmarks(results.pose_landmarks.landmark)

                # Évalue risques
                joint_risks, overall_risk = evaluate_joint_risk(angles, THRESHOLDS)

                # Flags
                def flag(val, cond):
                    if val is None:
                        return 0
                    return 1 if cond else 0

                flags = []
                # Tronc
                t = angles['Trunk']
                flags += [flag(t, t >= THRESHOLDS['Trunk_med']),
                          flag(t, t >= THRESHOLDS['Trunk_hi'])]
                # Cou
                n = angles['Neck']
                flags += [flag(n, n >= THRESHOLDS['Neck_med']),
                          flag(n, n >= THRESHOLDS['Neck_hi'])]
                # Épaules
                sl = angles['Left Shoulder']; sr = angles['Right Shoulder']
                flags += [flag(sl, sl is not None and sl >= THRESHOLDS['Shoulder_med']),
                          flag(sl, sl is not None and sl >= THRESHOLDS['Shoulder_hi'])]
                flags += [flag(sr, sr is not None and sr >= THRESHOLDS['Shoulder_med']),
                          flag(sr, sr is not None and sr >= THRESHOLDS['Shoulder_hi'])]
                # Coudes (en dehors de [70,160] = outside)
                el = angles['Left Elbow']; er = angles['Right Elbow']
                flags += [flag(el, el is not None and (el < 70 or el > 160)),
                          flag(er, er is not None and (er < 70 or er > 160))]
                # Genoux
                kl = angles['Left Knee']; kr = angles['Right Knee']
                flags += [flag(kl, kl is not None and kl < THRESHOLDS['Knee_med']),
                          flag(kl, kl is not None and kl < THRESHOLDS['Knee_hi'])]
                flags += [flag(kr, kr is not None and kr < THRESHOLDS['Knee_med']),
                          flag(kr, kr is not None and kr < THRESHOLDS['Knee_hi'])]
                # Hanches
                hl = angles['Left Hip']; hr = angles['Right Hip']
                flags += [flag(hl, hl is not None and hl < THRESHOLDS['Hip_med']),
                          flag(hl, hl is not None and hl < THRESHOLDS['Hip_hi'])]
                flags += [flag(hr, hr is not None and hr < THRESHOLDS['Hip_med']),
                          flag(hr, hr is not None and hr < THRESHOLDS['Hip_hi'])]

                # Ligne CSV
                row = [filename]
                for col in angle_cols:
                    v = angles.get(col)
                    row.append(None if v is None else round(float(v), 2))
                row += flags
                row += [round(float(vis_stats[c]), 3) for c in vis_cols]
                row += [overall_risk]
                writer.writerow(row)

                # Sauvegarde image annotée
                annotated = image.copy()
                mp_drawing.draw_landmarks(annotated, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                # Cadre coloré selon overall
                color = (0, 255, 0) if overall_risk == 'Safe' else ((0, 165, 255) if overall_risk == 'Medium Risk' else (0, 0, 255))
                h, w = annotated.shape[:2]
                cv2.rectangle(annotated, (0, 0), (w - 1, h - 1), color, 6)
                out_path = os.path.join(OUTPUT_FOLDER, filename)
                cv2.imwrite(out_path, annotated)

    print(f"✅ Terminé. CSV: {CSV_OUTPUT}, images annotées: {OUTPUT_FOLDER}")

if __name__ == "__main__":
    main_images()