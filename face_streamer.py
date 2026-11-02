import cv2
import json
import math
import numpy as np
import socket
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ==============================================================================
# CONFIGURATION & TUNABLE WEIGHTS
# ==============================================================================
UDP_IP = "127.0.0.1"
UDP_PORT = 4242
MODEL_PATH = "face_landmarker.task"
MAX_FACES = 5

# --- Scoring Weights (Higher score = higher priority rank) ---
# Formula: Score = (W_CENTER * CenterScore) + (W_DEPTH * DepthScore) + (W_GAZE * GazeScore)
WEIGHT_CENTER = 0.25    # Score based on how centered the face is in the frame (0.0 to 1.0)
WEIGHT_DEPTH = 0.60     # Score based on closeness to camera (larger face bounding area)
WEIGHT_GAZE = 0.15      # Score penalty/bonus based on alignment with target gaze vector

# --- Physical Setup Geometry & Gaze Tolerances ---
# Camera is ~30 cm above the figure's eye-line.
# When a child looks at the figure, their face tilts slightly DOWN relative to the camera vector.
TARGET_PITCH_DEG = 0.0   # Pitch offset in degrees expected when looking at figure (negative = pitched down)
TARGET_YAW_DEG = 0.0      # Yaw offset (0 = facing straight at screen/figure)

GAZE_MAX_DEVIATION_DEG = 79.0  # Deviation angle outside which gaze score drops to 0.0

# Initial State
SHOW_DEBUG = True

# ==============================================================================
# HELPER FUNCTIONS: 3D HEAD POSE & MATH
# ==============================================================================
def estimate_head_pose(landmarks, image_width, image_height):
    """
    Estimates Pitch, Yaw, Roll using OpenCV solvePnP with key 3D facial landmarks.
    """
    # 2D landmark indices from MediaPipe Face Mesh
    # Nose tip (1), Chin (152), Left Eye Corner (33), Right Eye Corner (263), Left Mouth Corner (61), Right Mouth Corner (291)
    image_points = np.array([
        (landmarks[1].x * image_width, landmarks[1].y * image_height),     # Nose tip
        (landmarks[152].x * image_width, landmarks[152].y * image_height), # Chin
        (landmarks[33].x * image_width, landmarks[33].y * image_height),   # Left eye corner
        (landmarks[263].x * image_width, landmarks[263].y * image_height), # Right eye corner
        (landmarks[61].x * image_width, landmarks[61].y * image_height),   # Left mouth corner
        (landmarks[291].x * image_width, landmarks[291].y * image_height)  # Right mouth corner
    ], dtype="double")

    # Generic 3D facial model coordinates (in mm)
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye corner
        (225.0, 170.0, -135.0),      # Right eye corner
        (-150.0, -150.0, -125.0),    # Left mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ])

    focal_length = image_width
    center = (image_width / 2.0, image_height / 2.0)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4, 1)) # Assume minimal lens distortion

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return 0.0, 0.0, 0.0, (0, 0, 0), (0, 0)

    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    proj_matrix = np.hstack((rotation_matrix, translation_vector))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)

    pitch = float(euler_angles[0][0])
    yaw = float(euler_angles[1][0])
    roll = float(euler_angles[2][0])

#    # Project a gaze vector 150mm forward from the nose tip for debug drawing
#    nose_end_point2D, _ = cv2.projectPoints(
#        np.array([(0.0, 0.0, 500.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs
#    )
#    p1 = (int(image_points[0][0]), int(image_points[0][1]))
#    p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
#
    
    # Compute eye midpoint (Left Eye Pupil/Center #468, Right Eye Pupil/Center #473, or inner corners #13, #362)
    # Using eye inner corners #133 and #362:
    left_eye = (landmarks[133].x * image_width, landmarks[133].y * image_height)
    right_eye = (landmarks[362].x * image_width, landmarks[362].y * image_height)
    eye_midpoint = (int((left_eye[0] + right_eye[0]) / 2), int((left_eye[1] + right_eye[1]) / 2))
    
    # In 3D model space, eye center sits ~35mm above the nose tip (0.0, 35.0, -20.0)
    nose_end_point2D, _ = cv2.projectPoints(
        np.array([(0.0, 35.0, 500.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs
    )
    
    p1 = eye_midpoint
    p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

    return pitch, yaw, roll, p1, p2

def compute_gaze_score(pitch, yaw):
    """
    Calculates how closely the person's head orientation points toward the figure screen setup.
    """
    delta_pitch = pitch - TARGET_PITCH_DEG
    delta_yaw = yaw - TARGET_YAW_DEG
    angular_distance = math.sqrt(delta_pitch**2 + delta_yaw**2)
    
    # Linear decay score from 1.0 (perfect alignment) to 0.0 (outside threshold angle)
    score = max(0.0, (GAZE_MAX_DEVIATION_DEG/angular_distance))

    # print(GAZE_MAX_DEVIATION_DEG, angular_distance, (GAZE_MAX_DEVIATION_DEG/angular_distance), score)
    return round(score, 3)

# ==============================================================================
# MAIN TRACKING PIPELINE
# ==============================================================================
def main():
    global SHOW_DEBUG

    # UDP Setup
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_address = (UDP_IP, UDP_PORT)

    # MediaPipe FaceLandmarker Initialization
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=MAX_FACES,
        output_facial_transformation_matrixes=True
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(2)
    if not cap.isOpened():
        print("[ERROR] Cannot access webcam.")
        return

    print(f"[INFO] Streaming face tracking data to {UDP_IP}:{UDP_PORT} over UDP...")
    print("[INFO] Press 'd' in window to toggle debug visualizer. Press 'q' to quit.")

    frame_timestamp_ms = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame_timestamp_ms = int(time.time() * 1000)
        h, w, _ = frame.shape

        # Convert image to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Run Face Detection
        results = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        faces_data = []

        if results.face_landmarks:
            for idx, landmarks in enumerate(results.face_landmarks):
                # 1. Bounding Box & Center Position
                xs = [lm.x for lm in landmarks]
                ys = [lm.y for lm in landmarks]
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)
                
                cx = (min_x + max_x) / 2.0
                cy = (min_y + max_y) / 2.0
                
                # Z metric fallback (from relative scale or transform matrix)
                cz = landmarks[1].z # Nose tip normalized Z

                # 2. Score Components
                # Center Score: 1.0 at screen center (0.5), drops to 0 at edges
                dist_from_center = math.sqrt((cx - 0.5)**2 + (cy - 0.5)**2)
                center_score = max(0.0, 1.0 - (dist_from_center / 0.707))

                # Depth Score: Bounding area proxy (larger box = closer to camera)
                bbox_area = (max_x - min_x) * (max_y - min_y)
                depth_score = min(1.0, bbox_area * 5.0)

                # 3. Head Pose & Gaze Score
                pitch, yaw, roll, p1, p2 = estimate_head_pose(landmarks, w, h)
                gaze_score = compute_gaze_score(pitch, yaw)
                is_looking_at = gaze_score > 0.45

                # 4. Total Priority Score
                total_score = (WEIGHT_CENTER * center_score) + \
                              (WEIGHT_DEPTH * depth_score) + \
                              (WEIGHT_GAZE * gaze_score)

                faces_data.append({
                    "raw_landmarks": landmarks,
                    "bbox": (int(min_x * w), int(min_y * h), int(max_x * w), int(max_y * h)),
                    "center": [round(cx, 4), round(cy, 4), round(cz, 4)],
                    "pose": {"pitch": round(pitch, 1), "yaw": round(yaw, 1), "roll": round(roll, 1)},
                    "gaze_score": gaze_score,
                    "looking_at": is_looking_at,
                    "total_score": round(total_score, 4),
                    "gaze_vector_pts": (p1, p2)
                })

        # Rank Order Faces by Highest Total Priority Score
        faces_data.sort(key=lambda item: item["total_score"], reverse=True)

        # Truncate stream payload to top 5 candidates
        faces_data = faces_data[:MAX_FACES]

        # Prepare JSON Payload for UDP Packet
        network_payload = []
        for rank, face in enumerate(faces_data, start=1):
            network_payload.append({
                "rank": rank,
                "score": face["total_score"],
                "looking": face["looking_at"],
                "pos_x": face["center"][0],
                "pos_y": face["center"][1],
                "pos_z": face["center"][2],
                "pitch": face["pose"]["pitch"],
                "yaw": face["pose"]["yaw"],
                "roll": face["pose"]["roll"]
            })

        # Broadcast packet to Godot
        payload_bytes = json.dumps({"faces": network_payload}).encode('utf-8')
        sock.sendto(payload_bytes, server_address)

        # ----------------------------------------------------------------------
        # DEBUG VISUALIZATION OVERLAY
        # ----------------------------------------------------------------------
        if SHOW_DEBUG:
            for rank, face in enumerate(faces_data, start=1):
                bx1, by1, bx2, by2 = face["bbox"]
                looking = face["looking_at"]

                # Green border if looking at figure, Yellow if facing away
                color = (0, 255, 0) if looking else (0, 215, 255)

                # Draw bounding box
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)

                # Draw gaze direction vector
                p1, p2 = face["gaze_vector_pts"]
                cv2.arrowedLine(frame, p1, p2, (255, 0, 255), 2, tipLength=0.3)

                # Annotate Rank and Score Info
                label_rank = f"#{rank} Score:{face['total_score']:.2f}"
                label_pose = f"P:{face['pose']['pitch']:.0f} Y:{face['pose']['yaw']:.0f}"
                label_status = "LOOKING" if looking else "AWAY"

                cv2.putText(frame, label_rank, (bx1, max(20, by1 - 25)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                cv2.putText(frame, f"{label_pose} [{label_status}]", (bx1, max(40, by1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            cv2.imshow("Pumpkin Tracker Debug (Press 'd' to toggle, 'q' to quit)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                SHOW_DEBUG = not SHOW_DEBUG
                cv2.destroyAllWindows()

    cap.release()
    cv2.destroyAllWindows()
    sock.close()

if __name__ == "__main__":
    main()
