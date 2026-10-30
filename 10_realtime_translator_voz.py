"""
PASO 4 (version con voz): Traductor de LSM en vivo, pensado para videollamada.

Agrega DOS funciones nuevas sobre el traductor original:

1) LSM -> voz: al completar una palabra (ej. "HOLA"), se lee en voz alta.
   - ESPACIO: agrega la letra estable actual a la palabra (igual que antes)
   - ENTER:   completa la palabra, la DICE EN VOZ ALTA, y limpia para la siguiente
   - BACKSPACE: borra la ultima letra
   - C: limpia la palabra sin decirla
   - ESC: salir

2) Voz -> texto: mientras tanto, se escucha el microfono (la persona que SI
   oye, hablando) y se transcribe en vivo como subtitulos en la parte de
   abajo de la pantalla. Corre en un hilo aparte para no trabar el video.

3) Modo zurda: tecla L para prender/apagar. Espeja las coordenadas X de los
   landmarks (no la imagen) antes de pasarselos al modelo, para que una
   persona zurda firme y el modelo (entrenado mayormente con mano derecha)
   lo reconozca igual de bien. La red neuronal NO se modifica.

Requiere instalar (agregado a requirements.txt):
    pyttsx3            (texto a voz, funciona offline)
    SpeechRecognition  (voz a texto)
    pyaudio            (acceso al microfono; en Windows puede requerir
                         instalarse con: pip install pipwin && pipwin install pyaudio)

Uso:
    python 10_realtime_translator_voz.py --model model_output/lsm_landmarks_model_final.h5
"""

import os
import json
import argparse
import threading
from collections import deque, Counter

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pyttsx3
import speech_recognition as sr

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def load_class_indices(model_dir):
    with open(os.path.join(model_dir, "class_indices.json"), "r", encoding="utf-8") as f:
        idx_to_class = json.load(f)
    return {int(k): v for k, v in idx_to_class.items()}


def normalize_landmarks(hand_landmarks, mirror_x=False):
    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])
    if mirror_x:
        # Modo zurda: espeja la coordenada X de cada landmark (1 - x).
        # Esto convierte la "forma matematica" de una mano izquierda en la
        # de una mano derecha, que es con la que se entreno el modelo.
        coords[:, 0] = 1.0 - coords[:, 0]
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
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20),
        (0, 17),
    ]
    points = [(int(lm.x * frame_w), int(lm.y * frame_h)) for lm in hand_landmarks]
    for a, b in connections:
        cv2.line(display, points[a], points[b], (0, 255, 0), 2)
    for p in points:
        cv2.circle(display, p, 4, (0, 200, 255), -1)


# ---------------------------------------------------------------------
# Hilo de texto-a-voz (LSM -> voz): para no congelar el video mientras habla
# ---------------------------------------------------------------------
def speak_async(text):
    def _speak():
        engine = pyttsx3.init()
        engine.setProperty("rate", 165)
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    threading.Thread(target=_speak, daemon=True).start()


# ---------------------------------------------------------------------
# Hilo de voz-a-texto (persona que oye, hablando -> subtitulos)
# ---------------------------------------------------------------------
class VoiceCaptioner:
    def __init__(self, language="es-MX"):
        self.recognizer = sr.Recognizer()
        self.language = language
        self.last_text = ""
        self.running = True
        self.mic_available = True
        try:
            self.microphone = sr.Microphone()
        except OSError:
            self.mic_available = False
            self.last_text = "[Microfono no disponible]"

    def start(self):
        if not self.mic_available:
            return
        threading.Thread(target=self._listen_loop, daemon=True).start()

    def _listen_loop(self):
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
        while self.running:
            try:
                with self.microphone as source:
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=8)
                text = self.recognizer.recognize_google(audio, language=self.language)
                if text:
                    self.last_text = text
            except sr.WaitTimeoutError:
                continue
            except sr.UnknownValueError:
                continue
            except sr.RequestError:
                self.last_text = "[Sin conexion a internet para voz->texto]"
            except Exception:
                continue

    def stop(self):
        self.running = False


def main(model_path, mediapipe_model_path, background_mode, stable_frames, cam_index, voice_lang):
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

    captioner = VoiceCaptioner(language=voice_lang)
    captioner.start()

    recent_predictions = deque(maxlen=stable_frames)
    word = ""
    frame_timestamp_ms = 0
    flash_counter = 0
    last_spoken = ""
    left_handed_mode = False

    print("Controles: ESPACIO=agregar letra | ENTER=completar palabra y decirla | BACKSPACE=borrar | C=limpiar | L=modo zurda | ESC=salir")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        frame_timestamp_ms += 33
        result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        if background_mode == "black":
            display = np.zeros_like(frame)
        else:
            display = frame.copy()

        current_letter = ""

        if result.hand_landmarks:
            hand_landmarks = result.hand_landmarks[0]
            draw_hand_landmarks(display, hand_landmarks, w, h)

            landmark_vector = normalize_landmarks(hand_landmarks, mirror_x=left_handed_mode).astype("float32")
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

        # ---- Textos: LSM -> texto ----
        modo_texto = "Zurda" if left_handed_mode else "Diestra"
        cv2.putText(display, f"Modo: {modo_texto} (L para cambiar)", (20, h - 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        cv2.putText(display, f"Letra actual: {current_letter}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (203, 105, 255), 2)
        cv2.putText(display, f"Estable: {stable_letter}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        word_color = (0, 255, 255) if flash_counter > 0 else (255, 255, 255)
        cv2.putText(display, f"LSM -> Texto: {word}", (20, h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, word_color, 2)
        if flash_counter > 0:
            flash_counter -= 1

        # ---- Texto: Voz -> texto (subtitulo de la persona que oye) ----
        caption = captioner.last_text
        cv2.putText(display, f"Voz -> Texto: {caption}", (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (150, 220, 255), 2)

        cv2.imshow("Traductor LSM <-> Voz", display)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == 32 and stable_letter:  # ESPACIO
            word += stable_letter
            recent_predictions.clear()
            flash_counter = 10
        elif key == 13:  # ENTER: completar palabra y decirla en voz alta
            if word:
                speak_async(word)
                last_spoken = word
                word = ""
                recent_predictions.clear()
        elif key == 8:  # BACKSPACE
            word = word[:-1]
        elif key in (ord("c"), ord("C")):
            word = ""
        elif key in (ord("l"), ord("L")):
            left_handed_mode = not left_handed_mode
            recent_predictions.clear()  # limpia predicciones viejas al cambiar de modo

    captioner.stop()
    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="model_output/lsm_landmarks_model_final.h5")
    parser.add_argument("--mediapipe_model", default="hand_landmarker.task")
    parser.add_argument("--background", choices=["black", "video"], default="video")
    parser.add_argument("--stable_frames", type=int, default=15)
    parser.add_argument("--cam_index", type=int, default=0)
    parser.add_argument("--voice_lang", default="es-MX", help="Idioma para voz->texto, ej. es-MX, es-ES, en-US")
    args = parser.parse_args()

    main(args.model, args.mediapipe_model, args.background, args.stable_frames, args.cam_index, args.voice_lang)
