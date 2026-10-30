"""
Junta las carpetas lsm-abc-A, lsm-abc-B, lsm-abc-C (que ya vienen con train/test
y carpetas por letra) en una sola estructura unificada:

    dataset/
        train/
            A/  (imágenes de los 3 grupos juntas)
            B/
            ...
        test/
            A/
            B/
            ...

Uso:
    python 00_merge_groups.py --input MSL-ABC --output dataset
"""

import os
import shutil
import argparse


def merge_groups(input_dir: str, output_dir: str):
    group_folders = [
        d for d in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, d)) and d.lower().startswith("lsm-abc-")
    ]

    if not group_folders:
        raise ValueError(
            f"No encontré carpetas tipo 'lsm-abc-A' dentro de {input_dir}. "
            "Revisa el nombre exacto de las carpetas y ajusta el filtro si hace falta."
        )

    print(f"Grupos encontrados: {group_folders}")

    for split in ["train", "test"]:
        for group in group_folders:
            group_split_path = os.path.join(input_dir, group, split)
            if not os.path.isdir(group_split_path):
                print(f"  Aviso: no existe {group_split_path}, se omite.")
                continue

            letters = [
                d for d in os.listdir(group_split_path)
                if os.path.isdir(os.path.join(group_split_path, d))
            ]

            for letter in letters:
                src_folder = os.path.join(group_split_path, letter)
                dst_folder = os.path.join(output_dir, split, letter)
                os.makedirs(dst_folder, exist_ok=True)

                images = [
                    f for f in os.listdir(src_folder)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ]

                for img in images:
                    # Prefijamos con el nombre del grupo para evitar que se
                    # sobrescriban archivos con el mismo nombre entre grupos
                    new_name = f"{group}_{img}"
                    shutil.move(
                        os.path.join(src_folder, img),
                        os.path.join(dst_folder, new_name),
                    )

            print(f"  {group}/{split}: copiado ({len(letters)} letras)")

    # Reporte final
    for split in ["train", "test"]:
        split_dir = os.path.join(output_dir, split)
        if os.path.isdir(split_dir):
            print(f"\n=== Resumen {split} ===")
            for letter in sorted(os.listdir(split_dir)):
                letter_dir = os.path.join(split_dir, letter)
                if os.path.isdir(letter_dir):
                    count = len(os.listdir(letter_dir))
                    print(f"  {letter}: {count} imágenes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="MSL-ABC", help="Carpeta con lsm-abc-A, lsm-abc-B, lsm-abc-C")
    parser.add_argument("--output", default="dataset", help="Carpeta de salida unificada")
    args = parser.parse_args()

    merge_groups(args.input, args.output)
