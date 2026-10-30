"""
PASO 1 de la optimizacion: convertir el dataset de IMAGENES en un dataset de LANDMARKS

Usa la API nueva de MediaPipe (Tasks), compatible con mediapipe 1.x.
Necesita el archivo de modelo hand_landmarker.task en la misma carpeta
(o indica su ruta con --model_path). Descargalo de:
    https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

Que hace:
- Recorre dataset/train y dataset/test
- Por cada imagen, detecta la mano con MediaPipe y extrae sus 21 landmarks (x, y, z)
- Normaliza los landmarks: relativos a la muneca y escalados por el tamano de la mano
- Guarda todo en dos CSV: landmarks_train.csv y landmarks_test.csv
- Reporta cuantas imagenes NO se pudieron procesar (sin mano detectada)

Uso:
    python 05_extract_landmarks.py --data_dir dataset --output_dir landmarks_data --model_path hand_landmarker.task
"""

import os
import argparse
import random

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from tqdm import tqdm

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def normalize_landmarks(hand_landmarks):
    """
    Convierte los 21 puntos (x,y,z) en coordenadas relativas a la muneca (punto 0)
    y escaladas por el tamano de la mano. Esto hace que el modelo generalice
    sin importar distancia a la camara ni posicion dentro del frame.
    """
    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])

    wrist = coords[0].copy()
    coords -= wrist  # relativo a la muneca

    max_dist = np.max(np.linalg.norm(coords, axis=1))
    if max_dist > 0:
        coords /= max_dist  # escalado por el tamano de la mano

    return coords.flatten()  # 21 * 3 = 63 valores


def process_split(data_dir_split: str, landmarker, max_per_class: int = None, seed: int = 42):
    rows = []
    failed = 0
    total = 0

    classes = sorted([
        d for d in os.listdir(data_dir_split)
        if os.path.isdir(os.path.join(data_dir_split, d))
    ])

    random.seed(seed)

    for cls in classes:
        cls_folder = os.path.join(data_dir_split, cls)
        images = [f for f in os.listdir(cls_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        if max_per_class is not None and len(images) > max_per_class:
            images = random.sample(images, max_per_class)

        for img_name in tqdm(images, desc=f"Procesando '{cls}'"):
            total += 1
            img_path = os.path.join(cls_folder, img_name)
            img = cv2.imread(img_path)
            if img is None:
                failed += 1
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

            result = landmarker.detect(mp_image)

            if not result.hand_landmarks:
                failed += 1
                continue

            hand_landmarks = result.hand_landmarks[0]
            landmark_vector = normalize_landmarks(hand_landmarks)

            row = list(landmark_vector) + [cls]
            rows.append(row)

    columns = [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")] + ["label"]
    df = pd.DataFrame(rows, columns=columns)

    print(f"\nTotal imagenes: {total} | Procesadas OK: {len(rows)} | Fallidas (sin mano detectada): {failed}")
    if total > 0:
        print(f"Tasa de fallo: {failed/total:.1%}  <- guarda este numero, es un dato relevante para tu reporte")

    return df


def main(data_dir: str, output_dir: str, model_path: str, max_per_class: int = None):
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No encontre el modelo en '{model_path}'. Descargalo de:\n"
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task\n"
            "y ponlo en la carpeta del proyecto (o indica la ruta con --model_path)."
        )

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
    )

    with HandLandmarker.create_from_options(options) as landmarker:
        for split in ["train", "test"]:
            print(f"\n=== Procesando split: {split} ===")
            split_dir = os.path.join(data_dir, split)
            split_max = max_per_class if split == "train" else (
                max(50, max_per_class // 5) if max_per_class else None
            )
            df = process_split(split_dir, landmarker, max_per_class=split_max)
            out_path = os.path.join(output_dir, f"landmarks_{split}.csv")
            df.to_csv(out_path, index=False)
            print(f"Guardado: {out_path}  ({len(df)} muestras, {df['label'].nunique()} clases)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="dataset")
    parser.add_argument("--output_dir", default="landmarks_data")
    parser.add_argument("--model_path", default="hand_landmarker.task")
    parser.add_argument("--max_per_class", type=int, default=600,
                         help="Maximo de imagenes de TRAIN a procesar por letra (test usa 1/5 de esto). "
                              "Usa 0 o negativo para procesar TODO el dataset (tarda mucho mas).")
    args = parser.parse_args()

    max_per_class = args.max_per_class if args.max_per_class and args.max_per_class > 0 else None
    main(args.data_dir, args.output_dir, args.model_path, max_per_class)
