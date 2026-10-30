# Traductor LSM en vivo — Proyecto modular (3 días)

## Estructura del proyecto

```
lsm_project/
├── 01_organize_dataset.py          # divide el dataset en train/test 80/20 + genera JSON de clases
├── 02_train_model.py               # entrena la red con transfer learning (MobileNetV2)
├── 03_evaluate_and_predict_image.py # matriz de confusión, análisis de errores, predicción por ruta de imagen
├── 04_realtime_translator.py       # traductor en vivo con webcam
├── requirements.txt
└── dataset_raw/                    # AQUÍ pones el dataset descargado (una carpeta por letra)
```

## Sobre los puntos que mencionaste

- **"Transfer Learning Medium" / "Version 8"**: probablemente el tutorial que viste usaba
  MobileNetV2 (muy común en artículos de Medium sobre clasificación de imágenes en tiempo real
  por lo ligero que es) o quizás YOLOv8 (para *detectar* la mano, no para clasificar la letra).
  Aquí usamos **MobileNetV2** para clasificar la letra recortada, y **MediaPipe** para detectar
  y recortar la mano — es la combinación más rápida de implementar en 3 días y corre bien sin GPU.
  Si el tutorial que viste era específicamente sobre YOLOv8, dime y te explico cómo integrarlo,
  pero para tu plazo esto es más simple y funciona igual de bien.

## Datasets recomendados (elige UNO)

1. Dataset de 21 señas estáticas del alfabeto LSM (300 imágenes por letra, ya con variaciones
   de rotación/escala): buscar "LSM Static Sign Recognition with Deep Learning-Based Features"
   en ResearchGate/repositorios académicos.
2. Dataset más grande (279,716 imágenes, 21 señas estáticas, 20 participantes):
   https://www.sciencedirect.com/science/article/pii/S2352340926004397 (también en PMC)
3. Dataset en Zenodo con alfabeto, números y palabras simples de LSM:
   https://doi.org/10.5281/zenodo.6554337

Descarga uno, descomprímelo, y acomódalo así dentro de `dataset_raw/`:
```
dataset_raw/
    A/  img1.jpg img2.jpg ...
    B/  img1.jpg ...
    C/  ...
```
Si el dataset que descargas viene en otro formato (por ejemplo con un CSV de anotaciones
en vez de carpetas), dime cómo viene exactamente y ajusto el script 01.

## Plan día por día

### Día 1 — Dataset + primer entrenamiento con imágenes
```bash
pip install -r requirements.txt

python 01_organize_dataset.py --input dataset_raw --output dataset --split 0.8

python 02_train_model.py --data_dir dataset --epochs 15 --img_size 224 --output_dir model_output
```
Esto te deja: `model_output/lsm_model_final.h5`, `class_indices.json`, `training_history.csv`
y `training_graphs.png`.

### Día 2 — Evaluación + predicción con imágenes por ruta (antes de meter video)
```bash
# Ver qué tan bien clasifica y qué letras confunde
python 03_evaluate_and_predict_image.py --mode evaluate --data_dir dataset --model model_output/lsm_model_final.h5

# Probar con una imagen suelta, como pediste ("primero que funcione con imágenes con ruta")
python 03_evaluate_and_predict_image.py --mode predict --image ruta/a/una/imagen.jpg --model model_output/lsm_model_final.h5
```
Revisa `model_output/confusion_matrix.png` y la salida en consola de "Top confusiones".
Si el accuracy de validación es mucho más bajo que el de entrenamiento (lo dice el script
al final: "aprendió" vs "memorizó"), antes de seguir aumenta el dropout o el data augmentation
en `02_train_model.py` y vuelve a entrenar — no avances a video con un modelo que memorizó.

### Día 3 — Traductor en vivo (video) + armar palabras
```bash
python 04_realtime_translator.py --model model_output/lsm_model_final.h5 --background black
```
- Haz la seña frente a la cámara.
- Cuando la letra se marque como "Estable", presiona ESPACIO para agregarla a la palabra.
- BACKSPACE borra, C limpia todo, ESC sale.
- Si prefieres ver tu video real en vez de pantalla negra: `--background video`.

## Notas para la entrega / defensa del proyecto

- **Modularidad**: cada paso es un script independiente que lee/escribe archivos, así que
  puedes explicar cada módulo por separado (captura, extracción, entrenamiento, inferencia).
- **Optimizaciones futuras** (para cuando digas "que esté más perron"):
  - Cambiar de letra-por-letra a reconocimiento de palabras completas con LSTM sobre
    secuencias de landmarks de MediaPipe (sin necesidad de imágenes, más ligero).
  - Agregar las 6 letras dinámicas (J, K, Ñ, Q, X, Z) usando el dataset de video mencionado arriba.
  - Fine-tuning más agresivo o probar EfficientNetB0 en vez de MobileNetV2.
  - Autocompletado de palabras en español para que no tengas que deletrear todo.
