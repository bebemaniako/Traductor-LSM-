"""
PASO 2 (version landmarks): entrenar una red neuronal simple sobre los landmarks
extraidos en el paso anterior (05_extract_landmarks.py).

Por que este modelo es distinto al 02_train_model.py original:
- Ya NO usamos imagenes crudas ni MobileNetV2 (transfer learning de vision).
- Aqui entrenamos una red densa pequena directo sobre los 63 numeros
  (21 landmarks x,y,z) que ya extrajimos. Es mucho mas rapida (minutos, no horas)
  y generaliza mejor porque no depende del fondo/iluminacion de la foto.

Que hace este script:
1. Carga landmarks_train.csv y landmarks_test.csv
2. Convierte las letras (texto) a numeros con un encoder, y guarda ese mapeo en JSON
3. Entrena una red densa pequena (Dense -> Dropout -> Dense -> Dropout -> Softmax)
4. Guarda el historial de entrenamiento con pandas (CSV)
5. Genera graficas de accuracy/loss (train vs val)
6. Evalua "aprendio o memorizo" comparando accuracy final de train vs val
7. Guarda el modelo entrenado (.h5) listo para usarse en el traductor en vivo

Uso:
    python 06_train_landmarks_model.py --data_dir landmarks_data --output_dir model_output
"""

import os
import json
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder


def load_data(data_dir: str):
    train_df = pd.read_csv(os.path.join(data_dir, "landmarks_train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "landmarks_test.csv"))

    feature_cols = [c for c in train_df.columns if c != "label"]

    X_train = train_df[feature_cols].values.astype("float32")
    y_train_raw = train_df["label"].values

    X_test = test_df[feature_cols].values.astype("float32")
    y_test_raw = test_df["label"].values

    return X_train, y_train_raw, X_test, y_test_raw, feature_cols


def build_model(input_dim: int, num_classes: int):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main(data_dir: str, output_dir: str, epochs: int, batch_size: int):
    os.makedirs(output_dir, exist_ok=True)

    print("Cargando datos de landmarks...")
    X_train, y_train_raw, X_test, y_test_raw, feature_cols = load_data(data_dir)
    print(f"Train: {X_train.shape[0]} muestras | Test: {X_test.shape[0]} muestras | Features: {X_train.shape[1]}")

    # Codificar las letras (texto) como numeros (0, 1, 2, ...)
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train_raw)
    y_test = encoder.transform(y_test_raw)

    idx_to_class = {int(i): cls for i, cls in enumerate(encoder.classes_)}
    with open(os.path.join(output_dir, "class_indices.json"), "w", encoding="utf-8") as f:
        json.dump(idx_to_class, f, ensure_ascii=False, indent=2)
    print(f"Clases: {idx_to_class}")

    model = build_model(input_dim=X_train.shape[1], num_classes=len(idx_to_class))
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True, monitor="val_accuracy"),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(output_dir, "lsm_landmarks_model.h5"), save_best_only=True, monitor="val_accuracy"
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
    )

    model.save(os.path.join(output_dir, "lsm_landmarks_model_final.h5"))

    # ---- Guardar historial con pandas ----
    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(os.path.join(output_dir, "training_history.csv"), index=False)
    print(f"\nHistorial guardado en {output_dir}/training_history.csv")

    # ---- Graficas ----
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(hist_df["accuracy"], label="Train accuracy")
    plt.plot(hist_df["val_accuracy"], label="Val accuracy")
    plt.title("Accuracy por epoca")
    plt.xlabel("Epoca")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(hist_df["loss"], label="Train loss")
    plt.plot(hist_df["val_loss"], label="Val loss")
    plt.title("Loss por epoca")
    plt.xlabel("Epoca")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_graphs.png"), dpi=150)
    print(f"Graficas guardadas en {output_dir}/training_graphs.png")

    # ---- Diagnostico: aprendio o memorizo ----
    final_train_acc = hist_df["accuracy"].iloc[-1]
    final_val_acc = hist_df["val_accuracy"].iloc[-1]
    gap = final_train_acc - final_val_acc
    print(f"\nAccuracy final train: {final_train_acc:.3f}")
    print(f"Accuracy final val:   {final_val_acc:.3f}")
    if gap > 0.10:
        print("DIAGNOSTICO: Gap notable entre train y val -> posible sobreajuste (memorizo).")
        print("  Sugerencias: mas dropout, menos epocas, o revisar calidad de landmarks.")
    else:
        print("DIAGNOSTICO: Gap pequeno -> el modelo esta generalizando razonablemente bien.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="landmarks_data")
    parser.add_argument("--output_dir", default="model_output")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    main(args.data_dir, args.output_dir, args.epochs, args.batch_size)
