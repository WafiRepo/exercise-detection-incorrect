import cv2
import pickle
import numpy as np
import mediapipe as mp

with open("./models/squat/squat_2.pkl", "rb") as f:
    clf = pickle.load(f) 

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.7,
)


cap = cv2.VideoCapture(r"E:\Holowellness\algorithm\Code\resources\videos\Squat -20250610T082930Z-1-001\Squat\incorrect\Trunk leaning foward (front).mp4")
fps = cap.get(cv2.CAP_PROP_FPS) or 30
delay = int(1000 / fps)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)


    res = pose.process(frame)
    if not res.pose_landmarks:
        cv2.imshow("Test Video", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break
        continue

    vec = []
    for lm in res.pose_landmarks.landmark:
        vec.extend([lm.x, lm.y, lm.z, lm.visibility])



    pred = clf.predict(np.array(vec).reshape(1, -1))[0]

    mp.solutions.drawing_utils.draw_landmarks(
        frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS
    )
    cv2.putText(frame, pred, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)


    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow("Test Video", bgr)
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
