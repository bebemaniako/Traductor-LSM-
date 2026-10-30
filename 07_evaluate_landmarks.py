"""
PASO 3 (version landmarks): evaluar el modelo sobre el set de test completo
y analizar en que letras se equivoca mas (matriz de confusion).

Tambien incluye un modo para probar con UNA imagen suelta por ruta (le saca
los landmarks primero, igual que hace el traductor en vivo despues).

Uso:

1) Evaluar sobre todo landmarks_test.csv:
    python 07_evaluate_landmarks.py --mode evaluate --data_dir landmarks_data --model model_output/lsm_landmarks_model_final.h5

2) Probar con una imagen suelta (por ruta):
    python 07_evaluate_landmarks.py --mode predict --image ruta/a/imagen.jpg --model model_output/lsm_landmarks_model_final.h5 --mediapipe_model hand_landmarker.task
"""

import os
import json
import argparse

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
import mediapipe as mp

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


def evaluate(data_dir: str, model_path: str):
    model_dir = os.path.dirname(model_path)
    idx_to_class = load_class_indices(model_dir)
    class_to_idx = {v: k for k, v in idx_to_class.items()}

    model = tf.keras.models.load_model(model_path)

    test_df = pd.read_csv(os.path.join(data_dir, "landmarks_test.csv"))
    feature_cols = [c for c in test_df.columns if c != "label"]

    X_test = test_df[feature_cols].values.astype("float32")
    y_true = test_df["label"].map(class_to_idx).values

    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    print("\n=== Reporte de clasificacion ===")
    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Prediccion")
    plt.ylabel("Real")
    plt.title("Matriz de confusion")
    out_path = os.path.join(model_dir, "confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nMatriz de confusion guardada en {out_path}")

    errors = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i][j] > 0:
                errors.append((class_names[i], class_names[j], cm[i][j]))
    errors.sort(key=lambda x: -x[2])

    print("\n=== Top confusiones (real -> predicho : cuantas veces) ===")
    for real, pred, count in errors[:10]:
        print(f"  {real} -> {pred} : {count} veces")


def predict_image(image_path: str, model_path: str, mediapipe_model_path: str):
    model_dir = os.path.dirname(model_path)
    idx_to_class = load_class_indices(model_dir)
    model = tf.keras.models.load_model(model_path)

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=mediapipe_model_path),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=1,
    )
    with HandLandmarker.create_from_options(options) as landmarker:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result = landmarker.detect(mp_image)

        if not result.hand_landmarks:
            print("No se detecto ninguna mano en esa imagen.")
            return

        landmark_vector = normalize_landmarks(result.hand_landmarks[0])

    X = np.expand_dims(landmark_vector.astype("float32"), axis=0)
    preds = model.predict(X)[0]
    top_idx = int(np.argmax(preds))
    letra = idx_to_class[top_idx]
    confianza = preds[top_idx]

    print(f"\nPrediccion: '{letra}'  (confianza: {confianza:.2%})")

    top3_idx = np.argsort(preds)[-3:][::-1]
    print("Top 3:")
    for idx in top3_idx:
        print(f"  {idx_to_class[idx]}: {preds[idx]:.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["evaluate", "predict"], required=True)
    parser.add_argument("--data_dir", default="landmarks_data")
    parser.add_argument("--image", help="Ruta de la imagen (solo modo predict)")
    parser.add_argument("--model", default="model_output/lsm_landmarks_model_final.h5")
    parser.add_argument("--mediapipe_model", default="hand_landmarker.task")
    args = parser.parse_args()

    if args.mode == "evaluate":
        evaluate(args.data_dir, args.model)
    else:
        if not args.image:
            raise ValueError("Debes dar --image ruta/a/imagen.jpg en modo predict")
        predict_image(args.image, args.model, args.mediapipe_model)
