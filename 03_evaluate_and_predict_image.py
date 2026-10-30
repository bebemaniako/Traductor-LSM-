"""
PASO 3: Cargar el modelo entrenado, analizar errores, y probar con UNA imagen por ruta
(esto es el "primero que funcione con imágenes con ruta" antes de pasar a video)

Dos modos de uso:

1) Evaluar sobre todo el set de test y generar matriz de confusión + errores:
    python 03_evaluate_and_predict_image.py --mode evaluate --data_dir dataset --model model_output/lsm_model_final.h5

2) Probar con una sola imagen (por ruta):
    python 03_evaluate_and_predict_image.py --mode predict --image ruta/a/imagen.jpg --model model_output/lsm_model_final.h5
"""

import os
import json
import argparse

import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_class_indices(model_dir: str):
    with open(os.path.join(model_dir, "class_indices.json"), "r", encoding="utf-8") as f:
        idx_to_class = json.load(f)
    # las keys vienen como strings ("0","1",...) al leer JSON
    return {int(k): v for k, v in idx_to_class.items()}


def evaluate(data_dir: str, model_path: str, img_size: int = 224):
    model_dir = os.path.dirname(model_path)
    idx_to_class = load_class_indices(model_dir)

    model = tf.keras.models.load_model(model_path)

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_gen = test_datagen.flow_from_directory(
        os.path.join(data_dir, "test"),
        target_size=(img_size, img_size),
        batch_size=32,
        class_mode="categorical",
        shuffle=False,
    )

    y_true = test_gen.classes
    y_pred_probs = model.predict(test_gen)
    y_pred = np.argmax(y_pred_probs, axis=1)

    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    print("\n=== Reporte de clasificación ===")
    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.title("Matriz de confusión")
    out_path = os.path.join(model_dir, "confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nMatriz de confusión guardada en {out_path}")

    # Analizar errores: qué letras se confunden más entre sí
    errors = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i][j] > 0:
                errors.append((class_names[i], class_names[j], cm[i][j]))
    errors.sort(key=lambda x: -x[2])

    print("\n=== Top confusiones (real -> predicho : cuántas veces) ===")
    for real, pred, count in errors[:10]:
        print(f"  {real} -> {pred} : {count} veces")


def predict_image(image_path: str, model_path: str, img_size: int = 224):
    model_dir = os.path.dirname(model_path)
    idx_to_class = load_class_indices(model_dir)
    model = tf.keras.models.load_model(model_path)

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (img_size, img_size))
    img_norm = img_resized.astype("float32") / 255.0
    img_batch = np.expand_dims(img_norm, axis=0)

    preds = model.predict(img_batch)[0]
    top_idx = int(np.argmax(preds))
    letra = idx_to_class[top_idx]
    confianza = preds[top_idx]

    print(f"\nPredicción: '{letra}'  (confianza: {confianza:.2%})")

    # Top 3 para ver qué tan seguro está
    top3_idx = np.argsort(preds)[-3:][::-1]
    print("Top 3:")
    for idx in top3_idx:
        print(f"  {idx_to_class[idx]}: {preds[idx]:.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["evaluate", "predict"], required=True)
    parser.add_argument("--data_dir", default="dataset")
    parser.add_argument("--image", help="Ruta de la imagen (solo modo predict)")
    parser.add_argument("--model", default="model_output/lsm_model_final.h5")
    parser.add_argument("--img_size", type=int, default=224)
    args = parser.parse_args()

    if args.mode == "evaluate":
        evaluate(args.data_dir, args.model, args.img_size)
    else:
        if not args.image:
            raise ValueError("Debes dar --image ruta/a/imagen.jpg en modo predict")
        predict_image(args.image, args.model, args.img_size)
