"""
PASO 4: Traductor de LSM en vivo (webcam -> letra -> palabra)

Cómo funciona:
1. MediaPipe detecta la mano en cada frame de la webcam.
2. Se recorta la región de la mano (con margen) y se le pasa al modelo entrenado.
3. El modelo predice una letra. Si la misma letra se sostiene "estable" varios
   frames seguidos (para evitar predicciones parpadeantes), se agrega a la palabra.
4. Se muestra todo sobre una pantalla negra (o puedes mostrar el video real,
   tú decides con --background).

Controles en la ventana:
    ESPACIO -> confirma la letra actual y la agrega a la palabra
    BACKSPACE -> borra la última letra
    C -> limpia la palabra completa
    ESC -> salir

Uso:
    python 04_realtime_translator.py --model model_output/lsm_model_final.h5
    python 04_realtime_translator.py --model model_output/lsm_model_final.h5 --background video
"""

import os
import json
import argparse
from collections import deque, Counter

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf


def load_class_indices(model_dir: str):
    with open(os.path.join(model_dir, "class_indices.json"), "r", encoding="utf-8") as f:
        idx_to_class = json.load(f)
    return {int(k): v for k, v in idx_to_class.items()}


def get_hand_bbox(hand_landmarks, frame_w, frame_h, margin=40):
    xs = [lm.x * frame_w for lm in hand_landmarks.landmark]
    ys = [lm.y * frame_h for lm in hand_landmarks.landmark]
    x_min, x_max = int(min(xs)) - margin, int(max(xs)) + margin
    y_min, y_max = int(min(ys)) - margin, int(max(ys)) + margin
    x_min, y_min = max(0, x_min), max(0, y_min)
    x_max, y_max = min(frame_w, x_max), min(frame_h, y_max)
    return x_min, y_min, x_max, y_max


def main(model_path: str, img_size: int, background_mode: str, stable_frames: int, cam_index: int):
    model_dir = os.path.dirname(model_path)
    idx_to_class = load_class_indices(model_dir)
    model = tf.keras.models.load_model(model_path)

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara. Revisa el índice --cam_index.")

    recent_predictions = deque(maxlen=stable_frames)
    current_letter = ""
    word = ""

    print("Controles: ESPACIO=agregar letra | BACKSPACE=borrar | C=limpiar | ESC=salir")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if background_mode == "black":
            display = np.zeros_like(frame)
        else:
            display = frame.copy()

        current_letter = ""

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            # Dibujar el esqueleto de la mano sobre la pantalla (negra o de video)
            mp_draw.draw_landmarks(display, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            x_min, y_min, x_max, y_max = get_hand_bbox(hand_landmarks, w, h)
            hand_crop = frame[y_min:y_max, x_min:x_max]

            if hand_crop.size > 0:
                crop_resized = cv2.resize(hand_crop, (img_size, img_size))
                crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
                crop_norm = crop_rgb.astype("float32") / 255.0
                crop_batch = np.expand_dims(crop_norm, axis=0)

                preds = model.predict(crop_batch, verbose=0)[0]
                top_idx = int(np.argmax(preds))
                confidence = preds[top_idx]

                if confidence > 0.6:
                    current_letter = idx_to_class[top_idx]
                    recent_predictions.append(current_letter)

                cv2.rectangle(display, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # Letra "estable": la más común en las últimas N predicciones
        stable_letter = ""
        if len(recent_predictions) == stable_frames:
            most_common, count = Counter(recent_predictions).most_common(1)[0]
            if count >= int(stable_frames * 0.7):
                stable_letter = most_common

        # ---- Overlay de texto ----
        cv2.putText(display, f"Letra actual: {current_letter}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display, f"Estable: {stable_letter}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)
        cv2.putText(display, f"Palabra: {word}", (20, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        cv2.imshow("Traductor LSM", display)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == 32 and stable_letter:  # ESPACIO
            word += stable_letter
            recent_predictions.clear()
        elif key == 8:  # BACKSPACE
            word = word[:-1]
        elif key in (ord("c"), ord("C")):
            word = ""

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="model_output/lsm_model_final.h5")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--background", choices=["black", "video"], default="black",
                         help="Fondo negro (default) o mostrar el video real de la cámara")
    parser.add_argument("--stable_frames", type=int, default=15,
                         help="Cuántos frames seguidos para considerar una letra 'estable'")
    parser.add_argument("--cam_index", type=int, default=0)
    args = parser.parse_args()

    main(args.model, args.img_size, args.background, args.stable_frames, args.cam_index)
