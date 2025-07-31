# streamlit_pose_app_clean.py

import cv2
import streamlit as st
import numpy as np
import mediapipe as mp
import torch
import pickle
import time
import warnings
import logging

# ────────── 1. Basic Configuration ──────────
warnings.filterwarnings("ignore", category=UserWarning, module="mediapipe")
logging.getLogger("mediapipe").setLevel(logging.ERROR)
logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)

st.set_page_config(page_title="AI Posture-Coach", layout="centered")

# ────────── 2. Device & Optional YOLO ──────────
if torch.cuda.is_available():
    default_dev = "cuda"
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    default_dev = "mps"
else:
    default_dev = "cpu"
st.sidebar.write(f"**Torch device detected:** {default_dev.upper()}")

use_yolo = st.sidebar.checkbox("Use YOLO Detection", value=False)
if use_yolo:
    device = torch.device(default_dev)
    st.sidebar.write(f"**YOLO running on:** {device}")
    model_yolo = torch.hub.load("ultralytics/yolov5", "custom",
                                path="./models/best_big_bounding.pt")
    model_yolo.to(device).eval()
else:
    device = torch.device("cpu")

# ────────── 3. Helper Functions ──────────
def calc_angle(a, b, c):
    a, b, c = map(np.array, (a, b, c))
    ang = abs(np.degrees(
        np.arctan2(c[1]-b[1], c[0]-b[0])
      - np.arctan2(a[1]-b[1], a[0]-b[0])
    ))
    return 360 - ang if ang > 180 else ang

def most_freq(lst):
    return max(lst, key=lst.count) if lst else None

# ────────── 4. UI ──────────
st.title("Real-time AI Posture Coach")

exercise = st.selectbox("Choose Exercise",
    ("Squat")
)

counter = 0
stage   = ""   # used in logic but not displayed
counter_box = st.sidebar.empty()
counter_box.header("Current Counter: 0")

# place to show the latest prediction
pred_box = st.sidebar.empty()

# angle_box = {
#     n: st.sidebar.empty() for n in [
#         "neck","left_shoulder","right_shoulder","left_elbow","right_elbow",
#         "left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"
#     ]
# }

feedback_box, last_feedback = st.empty(), 0.0

# ────────── 5. Load Classification Model ──────────
model_paths = {

    "Squat":       "./models/squat/squat_merged_2.pkl",

}
with open(model_paths[exercise], "rb") as f:
    clf = pickle.load(f)
logging.info(f"Loaded model for {exercise}: classes = {clf.classes_}")

# ────────── 6. Feedback mapping ──────────
feedback_map = {
    "s_correct":                                ("Great posture! ✅", st.success),
    "Knee_genuvalgus":                          ("Knees caving in — push them out! ❌", st.error),
    "Pelvis_anterior_tilt_and_lumbar_hyperextension": (
        "Anterior pelvic tilt & lumbar over-arch — keep neutral spine! ❌", st.error
    ),
    "Shifting_weight_to_left_side":             ("Weight shifted left — keep it centred! ❌", st.error),
    "Trunk_leaning_foward":                     ("Trunk leaning forward — lift your chest! ❌", st.error),
}

# feedback_map = {
#     "Knee_genuvalgus":                          ("Knees caving in — push them out! ❌", st.error),
#     "Shifting_weight_to_left_side":             ("Weight shifted left — keep it centred! ❌", st.error),
# }


# ────────── 7. Mediapipe Pose ──────────
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.7,
    model_complexity=1
)

frame_view = st.image([])
cam = cv2.VideoCapture(1)
hist = []

# ────────── 8. Main Loop ──────────
while True:
    ok, frame = cam.read()
    if not ok:
        st.error("Cannot access camera")
        break

    # flip mirror & convert to RGB
    frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)

    res = pose.process(frame)
    if not res.pose_landmarks:
        frame_view.image(frame)
        continue

    lms = res.pose_landmarks.landmark

    # Calculate key angles
    p = {lm.name.lower(): [lms[lm.value].x, lms[lm.value].y]
         for lm in mp_pose.PoseLandmark}

    neck = (calc_angle(p["left_shoulder"], p["nose"], p["left_hip"]) +
            calc_angle(p["right_shoulder"], p["nose"], p["right_hip"])) / 2

    ang = {
        "left_elbow":  calc_angle(p["left_shoulder"], p["left_elbow"], p["left_wrist"]),
        "right_elbow": calc_angle(p["right_shoulder"], p["right_elbow"], p["right_wrist"]),
        "left_shoulder":  calc_angle(p["left_elbow"], p["left_shoulder"], p["left_hip"]),
        "right_shoulder": calc_angle(p["right_elbow"], p["right_shoulder"], p["right_hip"]),
        "left_hip":   calc_angle(p["left_shoulder"], p["left_hip"], p["left_knee"]),
        "right_hip":  calc_angle(p["right_shoulder"], p["right_hip"], p["right_knee"]),
        "left_knee":  calc_angle(p["left_hip"], p["left_knee"], p["left_ankle"]),
        "right_knee": calc_angle(p["right_hip"], p["right_knee"], p["right_ankle"]),
        "left_ankle":  calc_angle(p["left_knee"], p["left_ankle"], p["left_heel"]),
        "right_ankle": calc_angle(p["right_knee"], p["right_ankle"], p["right_heel"]),
    }

    # update angle readouts
    # angle_box["neck"].text(f"Neck {neck:5.1f}°")
    # for k, v in ang.items():
    #     angle_box[k].text(f"{k.replace('_',' ').title()} {v:5.1f}°")

     # ─── Prediction ───
    # vec = [c for lm in lms for c in (lm.x, lm.y, lm.z, lm.visibility)]
    # vec += [neck] + list(ang.values())
    # pred = clf.predict(np.asarray(vec).reshape(1, -1))[0]

    vec = [c for lm in lms for c in (lm.x, lm.y, lm.z, lm.visibility)]
    pred = clf.predict(np.asarray(vec).reshape(1, -1))[0]

    # ─── Overlay teks pada frame ───
    # Buat label putih di pojok kiri atas, lalu tulis teks hitam
    text = f"{pred} | Count: {counter}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.7, 2
    (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
    margin = 8
    # background putih
    cv2.rectangle(frame,
                  (10 - margin, 10 - margin),
                  (10 + w + margin, 10 + h + margin),
                  (255, 255, 255), -1)
    # teks hitam
    cv2.putText(frame, text,
                (10, 10 + h),
                font, scale,
                (0, 0, 0), thickness)

    # ─── Feedback & drawing landmarks seperti biasa ───
    # … kode feedback_map dan hist …
    mp.solutions.drawing_utils.draw_landmarks(
        frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
        mp.solutions.drawing_styles.get_default_pose_landmarks_style())

    # tampilkan frame
    frame_view.image(frame)

# ────────── Cleanup ──────────
cam.release()
cv2.destroyAllWindows()
pose.close()
