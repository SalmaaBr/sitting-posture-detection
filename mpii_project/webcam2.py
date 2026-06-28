# -*- coding: utf-8 -*-
import cv2
import time
import numpy as np
import mediapipe as mp

# =========================
# Configuration
# =========================
SHOULDER_ELEVATION_MODE = True
THRESHOLDS = {
    'Trunk_med': 5.0,  # degrés depuis la verticale (0 = vertical)
    'Trunk_hi': 15.0,
    'Neck_med': 10.0,  # flexion/extension (verticale)
    'Neck_hi': 20.0,
    'NeckLat_med': 17.0,  # inclinaison latérale de la tête
    'NeckLat_hi': 30.0,
    'Shoulder_med': 45.0 if SHOULDER_ELEVATION_MODE else 70.0,
    'Shoulder_hi': 90.0 if SHOULDER_ELEVATION_MODE else 100.0,
    'Elbow_low': 50.0,
    'Elbow_high': 200.0,
    'Knee_med': 150.0,  # >=150 sûr, 120-150 moyen, <120 élevé
    'Knee_hi': 120.0,
    'Hip_med': 150.0,
    'Hip_hi': 120.0,
}
P_HIGH, P_LOW = 0.9, 0.6
T_HIGH, T_LOW = 1.8, 0.8  # secondes
EMA_ALPHA = 0.3
CALIB_FRAMES = 60  # ~2 s si ~30 fps
CALIBRATE_NECK_ROLL = False
SITTING_THIGH_ANGLE_THRESHOLD = 45.0
SITTING_KNEE_ANGLE_THRESHOLD = 140.0
POSTURE_T_CHANGE = 0.3  # Temps pour confirmer un changement de posture
WRIST_VISIBILITY_THRESHOLD = 0.5  # Seuil de visibilité pour les poignets

# =========================
# Utils géométriques
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
    ang = float(np.degrees(np.arccos(cosv)))
    return ang if np.isfinite(ang) else None

def angle_with_vertical(p_top, p_bottom):
    v = np.array(p_bottom, dtype=float) - np.array(p_top, dtype=float)
    v = unit(v)
    v_vert = np.array([0.0, 1.0], dtype=float)
    cosv = np.clip(np.dot(v, v_vert), -1.0, 1.0)
    ang = float(np.degrees(np.arccos(cosv)))
    if not np.isfinite(ang):
        return None
    return float(np.clip(ang, 0.0, 90.0))

def angle_with_horizontal(p_left, p_right):
    v = np.array(p_right, dtype=float) - np.array(p_left, dtype=float)
    n = np.linalg.norm(v)
    if not np.isfinite(n) or n < 1e-6:
        return None
    ang = float(np.degrees(np.arctan2(v[1], v[0])))
    ang = abs(ang)
    if ang > 90.0:
        ang = 180.0 - ang
    return float(ang)

# =========================
# EMA pour lisser les angles
# =========================
class EMA:
    def __init__(self, alpha=EMA_ALPHA):
        self.alpha = alpha
        self.state = {}

    def update(self, key, value):
        if value is None:
            return self.state.get(key, None)
        if key not in self.state or self.state[key] is None:
            self.state[key] = value
        else:
            self.state[key] = self.alpha * value + (1 - self.alpha) * self.state[key]
        return self.state[key]

# =========================
# MediaPipe et angles
# =========================
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def get_point(landmarks, name):
    lm = landmarks[mp_pose.PoseLandmark[name].value]
    return np.array([lm.x, lm.y], dtype=float), float(lm.visibility)

def shoulder_elevation(shoulder, distal):
    if shoulder is None or distal is None:
        return None
    return angle_with_vertical(shoulder, distal)

def compute_angles_from_landmarks(landmarks):
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
    L_EAR, v_ler = get_point(landmarks, 'LEFT_EAR')
    R_EAR, v_rer = get_point(landmarks, 'RIGHT_EAR')
    L_EYE, v_ley = get_point(landmarks, 'LEFT_EYE')
    R_EYE, v_rey = get_point(landmarks, 'RIGHT_EYE')

    # Centres
    if v_ls > 0.5 and v_rs > 0.5:
        m_sho = (L_SHO + R_SHO) / 2.0
    elif v_ls > 0.5:
        m_sho = L_SHO
    elif v_rs > 0.5:
        m_sho = R_SHO
    else:
        m_sho = None
    if v_lh > 0.5 and v_rh > 0.5:
        m_hip = (L_HIP + R_HIP) / 2.0
    elif v_lh > 0.5:
        m_hip = L_HIP
    elif v_rh > 0.5:
        m_hip = R_HIP
    else:
        m_hip = None

    angles = {}
    # Coudes
    angles['Left Elbow'] = safe_angle_abc(L_SHO, L_ELB, L_WRI) if (v_ls > 0.5 and v_le > 0.5 and v_lw > 0.5) else None
    angles['Right Elbow'] = safe_angle_abc(R_SHO, R_ELB, R_WRI) if (v_rs > 0.5 and v_re > 0.5 and v_rw > 0.5) else None
    # Épaules
    if SHOULDER_ELEVATION_MODE:
        angles['Left Shoulder'] = shoulder_elevation(L_SHO if v_ls > 0.5 else None, L_ELB if v_le > 0.5 else None)
        angles['Right Shoulder'] = shoulder_elevation(R_SHO if v_rs > 0.5 else None, R_ELB if v_re > 0.5 else None)
    else:
        angles['Left Shoulder'] = safe_angle_abc(L_ELB, L_SHO, L_HIP) if (v_le > 0.5 and v_ls > 0.5 and v_lh > 0.5) else None
        angles['Right Shoulder'] = safe_angle_abc(R_ELB, R_SHO, R_HIP) if (v_re > 0.5 and v_rs > 0.5 and v_rh > 0.5) else None
    # Genoux et hanches
    angles['Left Knee'] = safe_angle_abc(L_HIP, L_KNE, L_ANK) if (v_lh > 0.5 and v_lk > 0.5 and v_la > 0.5) else None
    angles['Right Knee'] = safe_angle_abc(R_HIP, R_KNE, R_ANK) if (v_rh > 0.5 and v_rk > 0.5 and v_ra > 0.5) else None
    angles['Left Hip'] = safe_angle_abc(L_SHO, L_HIP, L_KNE) if (v_ls > 0.5 and v_lh > 0.5 and v_lk > 0.5) else None
    angles['Right Hip'] = safe_angle_abc(R_SHO, R_HIP, R_KNE) if (v_rs > 0.5 and v_rh > 0.5 and v_rk > 0.5) else None
    # Tronc et cou
    angles['Trunk'] = angle_with_vertical(m_sho, m_hip) if (m_sho is not None and m_hip is not None) else None
    angles['Neck'] = angle_with_vertical(m_sho, NOSE) if (m_sho is not None and v_n > 0.5) else None
    # Cuisse pour posture
    angles['Left Thigh'] = angle_with_vertical(L_HIP, L_KNE) if (v_lh > 0.5 and v_lk > 0.5) else None
    angles['Right Thigh'] = angle_with_vertical(R_HIP, R_KNE) if (v_rh > 0.5 and v_rk > 0.5) else None
    # Inclinaison horizontale épaules + roll tête
    shoulder_tilt = angle_with_horizontal(L_SHO, R_SHO) if (v_ls > 0.5 and v_rs > 0.5) else None
    head_roll = None
    if v_ler > 0.5 and v_rer > 0.5:
        head_roll = angle_with_horizontal(L_EAR, R_EAR)
    elif v_ley > 0.5 and v_rey > 0.5:
        head_roll = angle_with_horizontal(L_EYE, R_EYE)
    neck_lateral = None
    if head_roll is not None and shoulder_tilt is not None:
        neck_lateral = abs(head_roll - shoulder_tilt)
    angles['Shoulder Tilt'] = shoulder_tilt
    angles['Head Roll'] = head_roll
    angles['Neck Lateral'] = neck_lateral
    # Vérifier si une seule main est détectée
    one_hand_detected = (v_lw > WRIST_VISIBILITY_THRESHOLD) != (v_rw > WRIST_VISIBILITY_THRESHOLD)
    return angles, one_hand_detected

# =========================
# Règles de risque robustes
# =========================
def risk_score_from_angles(angles, thresholds, is_sitting=False, one_hand_detected=False):
    votes_high = 0
    votes_med = 0

    def vote(val, med, hi, invert=False):
        nonlocal votes_high, votes_med
        if val is None:
            return
        if invert:
            if val < hi:
                votes_high += 1
            elif val < med:
                votes_med += 1
        else:
            if val >= hi:
                votes_high += 1
            elif val >= med:
                votes_med += 1

    t = angles.get('Trunk')
    n = angles.get('Neck')
    nlat = angles.get('Neck Lateral')
    vote(t, thresholds['Trunk_med'], thresholds['Trunk_hi'], invert=False)
    if not one_hand_detected:
        vote(n, thresholds['Neck_med'], thresholds['Neck_hi'], invert=False)
        vote(nlat, thresholds['NeckLat_med'], thresholds['NeckLat_hi'], invert=False)
    vote(angles.get('Left Shoulder'), thresholds['Shoulder_med'], thresholds['Shoulder_hi'], invert=False)
    vote(angles.get('Right Shoulder'), thresholds['Shoulder_med'], thresholds['Shoulder_hi'], invert=False)
    for e in ['Left Elbow', 'Right Elbow']:
        val = angles.get(e)
        if val is None:
            continue
        if val < thresholds['Elbow_low'] or val > thresholds['Elbow_high']:
            votes_high += 1
        else:
            if val < thresholds['Elbow_low'] + 10 or val > thresholds['Elbow_high'] - 10:
                votes_med += 1
    if not is_sitting:
        for k in ['Left Knee', 'Right Knee']:
            vote(angles.get(k), thresholds['Knee_med'], thresholds['Knee_hi'], invert=True)
        for h in ['Left Hip', 'Right Hip']:
            vote(angles.get(h), thresholds['Hip_med'], thresholds['Hip_hi'], invert=True)
    trunk_high = (t is not None and t >= thresholds['Trunk_hi'])
    neck_high = (n is not None and n >= thresholds['Neck_hi'] and not one_hand_detected)
    necklat_high = (nlat is not None and nlat >= thresholds['NeckLat_hi'] and not one_hand_detected)
    if votes_high >= 2 or trunk_high or neck_high or necklat_high:
        return 1.0, 'High Risk', votes_high, votes_med
    if votes_med >= 1:
        return 0.7, 'Medium Risk', votes_high, votes_med
    return 0.0, 'Safe', votes_high, votes_med

# =========================
# Hystérésis
# =========================
class AlertState:
    def __init__(self):
        self.on = False
        self.t_on = 0.0
        self.t_off = 0.0
        self.last_ts = None

    def update(self, p, now_ts):
        if self.last_ts is None:
            self.last_ts = now_ts
            return self.on
        dt = now_ts - self.last_ts
        self.last_ts = now_ts
        if self.on:
            if p < P_LOW:
                self.t_off += dt
                if self.t_off >= T_LOW:
                    self.on = False
                    self.t_off = 0.0
            else:
                self.t_off = 0.0
        else:
            if p > P_HIGH:
                self.t_on += dt
                if self.t_on >= T_HIGH:
                    self.on = True
                    self.t_on = 0.0
            else:
                self.t_on = 0.0
        return self.on

class PostureState:
    def __init__(self):
        self.is_sitting = True
        self.t_change = 0.0
        self.last_ts = None

    def update(self, candidate_sitting, now_ts):
        if self.last_ts is None:
            self.last_ts = now_ts
            return self.is_sitting
        dt = now_ts - self.last_ts
        self.last_ts = now_ts
        if self.is_sitting == candidate_sitting:
            self.t_change = 0.0
        else:
            self.t_change += dt
            if self.t_change >= POSTURE_T_CHANGE:
                self.is_sitting = candidate_sitting
                self.t_change = 0.0
        return self.is_sitting

# =========================
# Main webcam
# =========================
def main():
    ema = EMA(alpha=EMA_ALPHA)
    alert = AlertState()
    posture_state = PostureState()
    offset_cam_deg = {'Trunk': 0.0, 'Neck': 0.0, 'Neck Lateral': 0.0}
    calib_done = False
    calib_buffer = {'Trunk': [], 'Neck': [], 'Neck Lateral': []}
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Impossible d'ouvrir la webcam.")
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.6, min_tracking_confidence=0.6) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Impossible de recevoir le flux vidéo.")
                break
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            annotated = frame.copy()
            overall_label = 'NA'
            risk_p = 0.0
            votes_high = votes_med = 0
            is_sitting = False
            posture_label = "NA"
            one_hand_detected = False
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(annotated, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                angles, one_hand_detected = compute_angles_from_landmarks(results.pose_landmarks.landmark)
                smooth_angles = {k: ema.update(k, v) for k, v in angles.items()}
                if not calib_done:
                    for k in ['Trunk', 'Neck'] + (['Neck Lateral'] if CALIBRATE_NECK_ROLL else []):
                        if smooth_angles.get(k) is not None:
                            calib_buffer[k].append(smooth_angles[k])
                    have_enough = all(len(calib_buffer[k]) >= CALIB_FRAMES for k in ['Trunk', 'Neck'] + (['Neck Lateral'] if CALIBRATE_NECK_ROLL else []))
                    overall_label = 'Calibrating...'
                    if have_enough:
                        for k in ['Trunk', 'Neck'] + (['Neck Lateral'] if CALIBRATE_NECK_ROLL else []):
                            offset_cam_deg[k] = float(np.median(calib_buffer[k][-CALIB_FRAMES:]))
                        calib_done = True
                        print('Calibration done. Offsets: ' +
                              ', '.join([f'{k}={offset_cam_deg[k]:.1f}°' for k in offset_cam_deg if (CALIBRATE_NECK_ROLL or k != 'Neck Lateral')]))
                else:
                    for k in ['Trunk', 'Neck']:
                        if smooth_angles.get(k) is not None:
                            smooth_angles[k] = max(0.0, smooth_angles[k] - offset_cam_deg[k])
                    if CALIBRATE_NECK_ROLL and smooth_angles.get('Neck Lateral') is not None:
                        smooth_angles['Neck Lateral'] = max(0.0, smooth_angles['Neck Lateral'] - offset_cam_deg['Neck Lateral'])
                    th_l = smooth_angles.get('Left Thigh')
                    th_r = smooth_angles.get('Right Thigh')
                    knee_l = smooth_angles.get('Left Knee')
                    knee_r = smooth_angles.get('Right Knee')
                    th_vals = [th for th in [th_l, th_r] if th is not None]
                    knee_vals = [knee for knee in [knee_l, knee_r] if knee is not None]
                    candidate_sitting = False
                    if th_vals:
                        avg_th = np.mean(th_vals)
                        if avg_th > 60:
                            candidate_sitting = True
                    elif knee_vals:
                        avg_knee = np.mean(knee_vals)
                        if avg_knee < 150:
                            candidate_sitting = True
                    else:
                        candidate_sitting = True
                    now = time.time()
                    is_sitting = posture_state.update(candidate_sitting, now)
                    posture_label = "Sitting" if is_sitting else ("One Hand Detected (Neck Ignored)" if one_hand_detected else "Standing")
                    risk_p, rule_label, votes_high, votes_med = risk_score_from_angles(smooth_angles, THRESHOLDS, is_sitting=is_sitting, one_hand_detected=one_hand_detected)
                    on = alert.update(risk_p, now)
                    overall_label = 'High/Active' if (on and rule_label == 'High Risk') else ('Medium/Active' if on else rule_label)
            h, w = annotated.shape[:2]
            if overall_label.startswith('High'):
                color = (0, 0, 255); text = 'High Risk'
            elif overall_label.startswith('Medium'):
                color = (0, 165, 255); text = 'Medium Risk'
            elif overall_label == 'Safe':
                color = (0, 255, 0); text = 'Safe'
            elif overall_label == 'Calibrating...':
                color = (255, 255, 0); text = 'Calibrating... stand upright'
            else:
                color = (128, 128, 128); text = 'NA'
            cv2.rectangle(annotated, (0, 0), (w - 1, h - 1), color, 8)
            cv2.putText(annotated, text, (20, 40), font, 1.0, color, 3, cv2.LINE_AA)
            def fmt(v): return 'NA' if (v is None or not np.isfinite(v)) else f'{v:.1f}'
            y0 = 80
            trunk_txt = fmt(smooth_angles.get('Trunk')) if results.pose_landmarks else 'NA'
            neck_txt = fmt(smooth_angles.get('Neck')) if results.pose_landmarks else 'NA'
            neck_lat_txt = fmt(smooth_angles.get('Neck Lateral')) if results.pose_landmarks else 'NA'
            head_roll_txt = fmt(smooth_angles.get('Head Roll')) if results.pose_landmarks else 'NA'
            shoulder_tilt_txt = fmt(smooth_angles.get('Shoulder Tilt')) if results.pose_landmarks else 'NA'
            cv2.putText(annotated, f'Trunk(deg): {trunk_txt}', (20, y0), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated, f'Neck(deg): {neck_txt}', (20, y0+30), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated, f'NeckLat(deg): {neck_lat_txt}', (20, y0+60), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated, f'HeadRoll(deg): {head_roll_txt}', (20, y0+90), font, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
            cv2.putText(annotated, f'ShoulderTilt(deg): {shoulder_tilt_txt}', (20, y0+120), font, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
            cv2.putText(annotated, f'Votes H/M: {votes_high}/{votes_med}', (20, y0+150), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated, f'Posture: {posture_label}', (20, y0+180), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow('Pose Analysis (Ergo)', annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Traitement terminé.")

if __name__ == "__main__":
    main()