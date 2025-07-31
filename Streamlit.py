# streamlit_pose_app_clean.py
import cv2, streamlit as st, numpy as np, mediapipe as mp
import torch, pickle, pygame, time, warnings, logging

# ────────── 1. Basic Configuration ──────────
warnings.filterwarnings("ignore", category=UserWarning, module="mediapipe")
logging.getLogger("mediapipe").setLevel(logging.ERROR)
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
    ang = abs(np.degrees(np.arctan2(c[1]-b[1], c[0]-b[0]) -
                         np.arctan2(a[1]-b[1], a[0]-b[0])))
    return 360 - ang if ang > 180 else ang

def most_freq(lst):
    return max(lst, key=lst.count) if lst else None

# ────────── 4. UI ──────────
st.title("Real-time AI Posture Coach")
pygame.mixer.init()

exercise = st.selectbox("Choose Exercise",
                        ("Bench Press", "Squat", "Deadlift"))
counter = 0
stage   = ""           # used in logic but not displayed
counter_box = st.sidebar.empty()
counter_box.header("Current Counter: 0")

angle_box = {n: st.sidebar.empty() for n in [
    "neck","left_shoulder","right_shoulder","left_elbow","right_elbow",
    "left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"]}
feedback_box, last_feedback = st.empty(), 0.0

# ────────── 5. Load Classification Model ──────────
with open({
    "Bench Press": "./models/benchpress/benchpress.pkl",
    "Squat":       "./models/squat/squat_angle.pkl",
    "Deadlift":    "./models/deadlift/deadlift.pkl"}[exercise], "rb") as f:
    clf = pickle.load(f)

# ────────── 6. Mediapipe Pose ──────────
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5,
                    min_tracking_confidence=0.7,
                    model_complexity=1)

frame_view = st.image([])
cam = cv2.VideoCapture(1)
hist = []

while True:
    ok, frame = cam.read()
    if not ok:
        break
    frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)

    res = pose.process(frame)
    if not res.pose_landmarks:
        frame_view.image(frame); continue
    lms = res.pose_landmarks.landmark

    # ─── Calculate Angles ───
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

    angle_box["neck"].text(f"Neck {neck:5.1f}°")
    for k,v in ang.items():
        angle_box[k].text(f"{k.replace('_',' ').title()} {v:5.1f}°")

    # ─── Prediction ───
    vec = [c for lm in lms for c in (lm.x,lm.y,lm.z,lm.visibility)]
    vec += [neck] + list(ang.values())
    pred = clf.predict(np.asarray(vec).reshape(1,-1))[0]

    # ─── Counter (without displaying stage) ───
    if "down" in pred:
        stage = "down"
    elif stage=="down" and "up" in pred:
        stage, counter = "up", counter+1
        counter_box.header(f"Current Counter: {counter}")

    # ─── Feedback, display for 3 seconds then disappear ───
    if stage=="up":
        hist.append(pred)
        if len(hist) >= 10:
            dom = most_freq(hist)
            now = time.time()
            if dom and now - last_feedback >= 3:
                if   "correct" in dom:
                    feedback_box.info("Great posture! ✅")
                    pygame.mixer.music.load("./resources/sounds/correct.mp3")
                elif "excessive_arch" in dom:
                    feedback_box.error("Lower-back over-arched — keep neutral.")
                    pygame.mixer.music.load("./resources/sounds/excessive_arch_1.mp3")
                elif "arms_spread" in dom:
                    feedback_box.error("Grip too wide — narrow a bit.")
                    pygame.mixer.music.load("./resources/sounds/arms_spread_1.mp3")
                elif "spine_neutral" in dom:
                    feedback_box.error("Keep spine neutral: lift chest.")
                    pygame.mixer.music.load("./resources/sounds/spine_neutral_feedback_1.mp3")
                elif "caved_in_knees" in dom:
                    feedback_box.error("Push knees out — no cave-in.")
                    pygame.mixer.music.load("./resources/sounds/caved_in_knees_feedback_1.mp3")
                elif "feet_spread" in dom:
                    feedback_box.error("Stance too wide — shoulder width.")
                    pygame.mixer.music.load("./resources/sounds/feet_spread.mp3")
                elif "arms_narrow" in dom:
                    feedback_box.error("Grip too narrow — widen.")
                    pygame.mixer.music.load("./resources/sounds/arms_narrow.mp3")
                pygame.mixer.music.play()
                last_feedback, hist = now, []

    if time.time() - last_feedback > 2:
        feedback_box.empty()

    # ─── Display Frame ───
    mp.solutions.drawing_utils.draw_landmarks(
        frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
        mp.solutions.drawing_styles.get_default_pose_landmarks_style())
    frame_view.image(frame)
