"""
PASO 2: Entrenar la red neuronal usando Transfer Learning (MobileNetV2)

Por qué MobileNetV2 y no entrenar desde cero:
- Con pocas horas no te alcanza para entrenar una CNN desde cero bien.
- MobileNetV2 ya "sabe ver" bordes, texturas y formas (entrenada en ImageNet).
  Solo le enseñamos a distinguir las 21 letras encima de ese conocimiento.
- Es ligera -> entrena rápido incluso sin GPU potente.

Qué hace este script:
1. Carga las imágenes de dataset/train y dataset/test (generadas en el paso 1)
2. Aplica data augmentation (rotación, zoom, brillo) para que generalice mejor
3. Congela MobileNetV2 y le agrega una cabeza de clasificación propia
4. Entrena, guarda el historial en un CSV con pandas
5. Genera gráficas de accuracy/loss (train vs val) -> para "aprendió o memorizó"
6. Guarda el modelo entrenado (.h5) y class_indices.json actualizado

Uso:
    python 02_train_model.py --data_dir dataset --epochs 15 --img_size 224
"""

import os
import json
import argparse

import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def build_model(num_classes: int, img_size: int):
    base_model = MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False  # Congelamos el backbone (transfer learning clásico)

    inputs = tf.keras.Input(shape=(img_size, img_size, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base_model


def main(data_dir: str, epochs: int, img_size: int, batch_size: int, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        zoom_range=0.15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=[0.8, 1.2],
    )
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        train_dir, target_size=(img_size, img_size),
        batch_size=batch_size, class_mode="categorical",
    )
    test_gen = test_datagen.flow_from_directory(
        test_dir, target_size=(img_size, img_size),
        batch_size=batch_size, class_mode="categorical", shuffle=False,
    )

    num_classes = train_gen.num_classes
    print(f"Clases detectadas: {train_gen.class_indices}")

    # Guardar/actualizar el JSON de clases (índice -> letra, como lo usa el modelo)
    idx_to_class = {v: k for k, v in train_gen.class_indices.items()}
    with open(os.path.join(output_dir, "class_indices.json"), "w", encoding="utf-8") as f:
        json.dump(idx_to_class, f, ensure_ascii=False, indent=2)

    model, base_model = build_model(num_classes, img_size)
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True, monitor="val_accuracy"),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(output_dir, "lsm_model.h5"), save_best_only=True, monitor="val_accuracy"
        ),
    ]

    history = model.fit(
        train_gen, validation_data=test_gen, epochs=epochs, callbacks=callbacks,
    )

    # (Opcional pero recomendado si te sobra tiempo) Fine-tuning:
    # descongelar las últimas capas de MobileNetV2 y entrenar unas épocas más
    # con learning rate bajo para exprimir un poco más de accuracy.
    print("\nIniciando fine-tuning (descongelando últimas capas)...")
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    history_fine = model.fit(
        train_gen, validation_data=test_gen, epochs=5, callbacks=callbacks,
    )

    model.save(os.path.join(output_dir, "lsm_model_final.h5"))

    # ---- Guardar historial combinado con pandas ----
    hist_df = pd.DataFrame(history.history)
    hist_fine_df = pd.DataFrame(history_fine.history)
    full_hist = pd.concat([hist_df, hist_fine_df], ignore_index=True)
    full_hist.to_csv(os.path.join(output_dir, "training_history.csv"), index=False)
    print(f"\nHistorial guardado en {output_dir}/training_history.csv")

    # ---- Gráficas de accuracy y loss ----
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(full_hist["accuracy"], label="Train accuracy")
    plt.plot(full_hist["val_accuracy"], label="Val accuracy")
    plt.axvline(x=len(hist_df) - 1, color="gray", linestyle="--", label="Inicio fine-tuning")
    plt.title("Accuracy por época")
    plt.xlabel("Época")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(full_hist["loss"], label="Train loss")
    plt.plot(full_hist["val_loss"], label="Val loss")
    plt.axvline(x=len(hist_df) - 1, color="gray", linestyle="--", label="Inicio fine-tuning")
    plt.title("Loss por época")
    plt.xlabel("Época")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_graphs.png"), dpi=150)
    print(f"Gráficas guardadas en {output_dir}/training_graphs.png")

    # ---- Diagnóstico rápido: ¿aprendió o memorizó? ----
    final_train_acc = full_hist["accuracy"].iloc[-1]
    final_val_acc = full_hist["val_accuracy"].iloc[-1]
    gap = final_train_acc - final_val_acc
    print(f"\nAccuracy final train: {final_train_acc:.3f}")
    print(f"Accuracy final val:   {final_val_acc:.3f}")
    if gap > 0.15:
        print("DIAGNÓSTICO: Gap grande entre train y val -> probable OVERFITTING (memorizó).")
        print("  Sugerencias: más data augmentation, más dropout, o más datos.")
    else:
        print("DIAGNÓSTICO: Gap pequeño -> el modelo está generalizando razonablemente bien.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="dataset")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", default="model_output")
    args = parser.parse_args()

    main(args.data_dir, args.epochs, args.img_size, args.batch_size, args.output_dir)
