"""
Verifica visualmente que el flip (espejado) de landmarks tenga sentido,
antes de usarlo como data augmentation en el entrenamiento.

Que hace:
- Carga un CSV de landmarks (generado por 05_extract_landmarks.py)
- Toma N ejemplos de una letra especifica
- Grafica lado a lado: landmarks originales vs landmarks espejados
- Guarda la figura como PNG para que la revises

Uso:
    python 09_verificar_flip.py --csv landmarks_data/landmarks_train.csv --letra A --n 3
    python 09_verificar_flip.py --csv landmarks_data/landmarks_train.csv --letra J --n 5
"""

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Conexiones estandar de MediaPipe Hands (pares de indices de landmarks)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # pulgar
    (0, 5), (5, 6), (6, 7), (7, 8),          # indice
    (0, 9), (9, 10), (10, 11), (11, 12),     # medio
    (0, 13), (13, 14), (14, 15), (15, 16),   # anular
    (0, 17), (17, 18), (18, 19), (19, 20),   # menique
    (5, 9), (9, 13), (13, 17),               # palma
]


def flip_landmarks(landmark_vector):
    """
    Espeja el vector de landmarks normalizados en el eje x.
    Como ya estan centrados en la muneca (no en rango 0-1), el flip es x = -x.
    Simula la mano contraria (diestro <-> zurdo) sin volver a correr MediaPipe.
    """
    coords = landmark_vector.reshape(21, 3).copy()
    coords[:, 0] *= -1  # invierte solo el eje x
    return coords.flatten()


def plot_hand(ax, landmark_vector, title):
    coords = landmark_vector.reshape(21, 3)
    xs, ys = coords[:, 0], coords[:, 1]

    for i, j in HAND_CONNECTIONS:
        ax.plot([xs[i], xs[j]], [ys[i], ys[j]], 'b-', linewidth=1)
    ax.scatter(xs, ys, c='red', s=20, zorder=3)

    ax.invert_yaxis()  # el eje y de MediaPipe crece hacia abajo, como en imagenes
    ax.set_aspect('equal')
    ax.set_title(title)


def verificar_flip(csv_path: str, letra: str, n_ejemplos: int = 3):
    df = pd.read_csv(csv_path)
    subset = df[df['label'] == letra]

    if subset.empty:
        letras_disponibles = sorted(df['label'].unique())
        raise ValueError(
            f"No hay muestras con label='{letra}' en {csv_path}.\n"
            f"Letras disponibles: {letras_disponibles}"
        )

    n_ejemplos = min(n_ejemplos, len(subset))
    muestras = subset.sample(n=n_ejemplos, random_state=42)

    fig, axes = plt.subplots(n_ejemplos, 2, figsize=(6, 3 * n_ejemplos))
    if n_ejemplos == 1:
        axes = axes.reshape(1, -1)

    for idx, (_, row) in enumerate(muestras.iterrows()):
        landmark_vector = row.drop('label').values.astype(float)
        flipped = flip_landmarks(landmark_vector)

        plot_hand(axes[idx, 0], landmark_vector, f"'{letra}' original")
        plot_hand(axes[idx, 1], flipped, f"'{letra}' espejado")

    plt.tight_layout()
    out_name = f'verificacion_flip_{letra}.png'
    plt.savefig(out_name, dpi=150)
    print(f"Guardado: {out_name}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Ruta al CSV de landmarks (ej. landmarks_data/landmarks_train.csv)")
    parser.add_argument("--letra", required=True, help="Letra/clase a verificar (ej. A)")
    parser.add_argument("--n", type=int, default=3, help="Numero de ejemplos a graficar")
    args = parser.parse_args()

    verificar_flip(args.csv, args.letra, args.n)
