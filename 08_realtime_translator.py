"""
PASO 4: Traductor de LSM en vivo (webcam -> letra -> palabra)
Version actualizada para usar landmarks + la API nueva de MediaPipe Tasks.

Como funciona:
1. MediaPipe detecta la mano en cada frame de la webcam usando HandLandmarker.
2. Se extraen y normalizan los 21 landmarks (igual que en el entrenamiento).
3. Se le pasan al modelo entrenado, que predice una letra.
4. Si la misma letra se sostiene "estable" varios frames seguidos, se puede
   confirmar con ESPACIO y se agrega a la palabra.

Controles en la ventana:
    ESPACIO -> confirma la letra actual y la agrega a la palabra
    BACKSPACE -> borra la ultima letra
    C -> limpia la palabra completa
    ESC -> salir

Uso:
    python 08_realtime_translator.py --model model_output/lsm_landmarks_model_final.h5 --mediapipe_model hand_landmarker.task
"""

import os
import json
import argparse
from collections import deque, Counter

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def load_class_indices(model_dir: str):
    with open(os.path.join(model_dir, "class_indices.json"), "r", encoding="utf-8") as f:
        idx_to_class = json.load(f)
    return {int(k): v for k, v in idx_to_class.items()}


def normalize_landmarks(hand_landmarks):
    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])
    wrist = coords[0].copy()
    coords -= wrist
    max_dist = np.max(np.linalg.norm(coords, axis=1))
    if max_dist > 0:
        coords /= max_dist
    return coords.flatten()


def get_hand_bbox(hand_landmarks, frame_w, frame_h, margin=30):
    xs = [lm.x * frame_w for lm in hand_landmarks]
    ys = [lm.y * frame_h for lm in hand_landmarks]
    x_min, x_max = int(min(xs)) - margin, int(max(xs)) + margin
    y_min, y_max = int(min(ys)) - margin, int(max(ys)) + margin
    x_min, y_min = max(0, x_min), max(0, y_min)
    x_max, y_max = min(frame_w, x_max), min(frame_h, y_max)
    return x_min, y_min, x_max, y_max


def draw_hand_landmarks(display, hand_landmarks, frame_w, frame_h):
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),          # pulgar
        (0, 5), (5, 6), (6, 7), (7, 8),          # indice
        (5, 9), (9, 10), (10, 11), (11, 12),     # medio
        (9, 13), (13, 14), (14, 15), (15, 16),   # anular
        (13, 17), (17, 18), (18, 19), (19, 20),  # menique
        (0, 17),
    ]
    points = [(int(lm.x * frame_w), int(lm.y * frame_h)) for lm in hand_landmarks]
    for a, b in connections:
        cv2.line(display, points[a], points[b], (0, 255, 0), 2)
    for p in points:
        cv2.circle(display, p, 4, (0, 200, 255), -1)


def main(model_path: str, mediapipe_model_path: str, background_mode: str, stable_frames: int, cam_index: int):
    model_dir = os.path.dirname(model_path)
    idx_to_class = load_class_indices(model_dir)
    model = tf.keras.models.load_model(model_path)

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=mediapipe_model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    landmarker = HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la camara. Revisa el indice --cam_index.")

    recent_predictions = deque(maxlen=stable_frames)
    word = ""
    frame_timestamp_ms = 0
    flash_counter = 0  # cuenta frames restantes para mostrar el "destello" al confirmar letra

    print("Controles: ESPACIO=agregar letra | BACKSPACE=borrar | C=limpiar | ESC=salir")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        frame_timestamp_ms += 33  # aprox 30 fps
        result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        if background_mode == "black":
            display = np.zeros_like(frame)
        else:
            display = frame.copy()

        current_letter = ""

        if result.hand_landmarks:
            hand_landmarks = result.hand_landmarks[0]
            draw_hand_landmarks(display, hand_landmarks, w, h)

            landmark_vector = normalize_landmarks(hand_landmarks).astype("float32")
            X = np.expand_dims(landmark_vector, axis=0)
            preds = model.predict(X, verbose=0)[0]
            top_idx = int(np.argmax(preds))
            confidence = preds[top_idx]

            if confidence > 0.6:
                current_letter = idx_to_class[top_idx]
                recent_predictions.append(current_letter)

            x_min, y_min, x_max, y_max = get_hand_bbox(hand_landmarks, w, h)
            cv2.rectangle(display, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        stable_letter = ""
        if len(recent_predictions) == stable_frames:
            most_common, count = Counter(recent_predictions).most_common(1)[0]
            if count >= int(stable_frames * 0.7):
                stable_letter = most_common

        cv2.putText(display, f"Letra actual: {current_letter}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (203, 105, 255), 2)
        cv2.putText(display, f"Letra Estable: {stable_letter}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        word_color = (0, 255, 255) if flash_counter > 0 else (255, 255, 255)
        cv2.putText(display, f"Palabra: {word}", (20, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, word_color, 2)
        if flash_counter > 0:
            flash_counter -= 1

        cv2.imshow("Traductor LSM", display)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == 32 and stable_letter:
            word += stable_letter
            recent_predictions.clear()
            flash_counter = 10  # dura ~10 frames mostrando el destello amarillo
        elif key == 8:
            word = word[:-1]
        elif key in (ord("c"), ord("C")):
            word = ""

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="model_output/lsm_landmarks_model_final.h5")
    parser.add_argument("--mediapipe_model", default="hand_landmarker.task")
    parser.add_argument("--background", choices=["black", "video"], default="black")
    parser.add_argument("--stable_frames", type=int, default=15)
    parser.add_argument("--cam_index", type=int, default=0)
    args = parser.parse_args()

    main(args.model, args.mediapipe_model, args.background, args.stable_frames, args.cam_index)
