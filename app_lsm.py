"""
Traductor LSM en tiempo real - Panel lateral
=============================================
Interfaz de panel lateral (estilo chat de videollamada) construida sobre el
pipeline REAL del proyecto: 08_realtime_translator.py

Se conserva exactamente la lógica del script original:
  - cv2.flip(frame, 1) ANTES de detectar (el modelo ve la imagen en espejo)
  - MediaPipe HandLandmarker (VIDEO, 1 mano, confianza 0.6)
  - normalización: relativa a la muñeca / distancia máxima a la muñeca
  - clases leídas de class_indices.json (junto al modelo)
  - solo se aceptan predicciones con confianza > 0.6
  - letra estable = >= 70 % de las últimas 15 predicciones aceptadas
  - ESPACIO confirma, RETROCESO borra, C limpia, ESC sale

Se agrega: panel lateral con chat, tracking de dedos (esqueleto coloreado +
indicadores por dedo), top-3 de predicciones y fondo negro/video.

Uso (desde la carpeta del proyecto, la misma de 08_realtime_translator.py):
    python app_lsm.py
    python app_lsm.py --model model_output/lsm_landmarks_model_final.h5 --mediapipe_model hand_landmarker.task
    python app_lsm.py --background video --theme light --cam_index 1
    python app_lsm.py --demo          (solo interfaz, con letras simuladas)

Si el reconocimiento real no puede cargarse, la app AVISA con una ventana y
abre en MODO DEMO (letras simuladas). No es el reconocimiento real.

Estructura del archivo:
  1. CONFIGURACIÓN  2. LÓGICA  3. COMPONENTES  4. INTERFAZ (clase App)
"""

from __future__ import annotations

import argparse
import json
import time
import tkinter as tk
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import messagebox

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageTk

# =============================================================================
# 1. CONFIGURACIÓN
# =============================================================================
MODEL_PATH = Path("model_output/lsm_landmarks_model_final.h5")   # igual que el script 08
HAND_MODEL_PATH = Path("hand_landmarker.task")

CAMERA_INDEX = 0
FRAME_DELAY_MS = 15                 # pausa entre cuadros de la interfaz

# --- Pipeline (idéntico a 08_realtime_translator.py) -------------------------
MIRROR_CAMERA = True                # voltea el cuadro ANTES de reconocer (como el original)
BACKGROUND = "black"                # "black": solo esqueleto sobre negro | "video": cámara
MIN_DETECTION_CONFIDENCE = 0.6      # MediaPipe
MIN_TRACKING_CONFIDENCE = 0.6       # MediaPipe
MIN_CONFIDENCE = 0.6                # el modelo: se acepta solo si confianza > 0.6
WINDOW_SIZE = 15                    # --stable_frames
STABILITY_THRESHOLD = 0.70          # la letra debe ocupar >= 70 % de la ventana
FLASH_FRAMES = 10                   # destello de la palabra al confirmar una letra

# Solo para el modo demo (en modo real las clases salen de class_indices.json)
DEMO_CLASSES = list("ABCDEFGHILMNOPRSTUVWY")

# --- Panel lateral -----------------------------------------------------------
PANEL_W = 360
PAD = 16
VIDEO_W = PANEL_W - 2 * PAD
VIDEO_H = 210
START_PINNED = True                 # arrancar siempre-encima
DOCK_RIGHT = True                   # pegar la ventana al borde derecho

FONT = "Segoe UI"

THEME_NAME = "dark"                 # "dark" o "light" (también con --theme)
THEMES = {
    "dark": {
        "bg": "#0f1117", "surface": "#181b24", "surface2": "#232736",
        "text": "#eceff4", "muted": "#8b93a7",
        "accent": "#5eead4", "accent_text": "#06231f",
        "bubble": "#2c3a5e", "bubble_text": "#eef2ff",
        "ok": "#4ade80", "warn": "#fbbf24", "danger": "#f87171",
    },
    "light": {
        "bg": "#f4f6fa", "surface": "#ffffff", "surface2": "#eaedf4",
        "text": "#1b2130", "muted": "#6b7488",
        "accent": "#0d9488", "accent_text": "#ffffff",
        "bubble": "#dbeafe", "bubble_text": "#1e3a8a",
        "ok": "#16a34a", "warn": "#d97706", "danger": "#dc2626",
    },
}
T = dict(THEMES[THEME_NAME])        # copia: se puede cambiar el tema sin tocar THEMES

# --- Dedos -------------------------------------------------------------------
FINGER_NAMES = ["Pulgar", "Índice", "Medio", "Anular", "Meñique"]
FINGER_CHAINS = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12],
                 [13, 14, 15, 16], [17, 18, 19, 20]]
FINGER_COLORS = ["#f59e0b", "#4ade80", "#38bdf8", "#a78bfa", "#f472b6"]
PALM_LINKS = [(0, 1), (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)]
FINGER_EXTEND_MARGIN = 1.08         # qué tan más lejos debe estar la punta para "extendido"


# =============================================================================
# 2. LÓGICA (independiente de la interfaz)
# =============================================================================
def to_points(landmarks):
    """Landmarks de MediaPipe (o un arreglo numpy) -> arreglo (21, 3)."""
    if isinstance(landmarks, np.ndarray):
        return landmarks.astype(np.float32).copy()
    return np.array([[p.x, p.y, p.z] for p in landmarks], dtype=np.float32)


def normalize_landmarks(landmarks):
    """Igual que en el entrenamiento: relativo a la muñeca y escalado por la
    distancia máxima a la muñeca. Devuelve 63 valores (21 x 3)."""
    coords = to_points(landmarks)
    coords -= coords[0].copy()
    max_dist = np.max(np.linalg.norm(coords, axis=1))
    if max_dist > 0:
        coords /= max_dist
    return coords.flatten()


def finger_states(landmarks, aspect=1.0, margin=FINGER_EXTEND_MARGIN):
    """
    [pulgar, índice, medio, anular, meñique] -> True si el dedo está extendido.

    Heurística geométrica (no usa el modelo): un dedo está extendido si su
    punta queda más lejos de la muñeca que su articulación media. `aspect`
    (ancho/alto del cuadro) corrige la deformación de las coordenadas
    normalizadas de MediaPipe.
    """
    pts = to_points(landmarks)
    pts[:, 0] *= aspect
    dist = lambda a, b: float(np.linalg.norm(pts[a] - pts[b]))   # noqa: E731
    # Pulgar: se mide contra el nudillo del meñique (no se alinea con la muñeca)
    states = [dist(4, 17) > dist(3, 17) * margin]
    for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18)):
        states.append(dist(tip, 0) > dist(pip, 0) * margin)
    return states


def get_hand_bbox(landmarks, frame_w, frame_h, margin=30):
    """Caja que rodea la mano (igual que el script original)."""
    xs = [lm.x * frame_w for lm in landmarks]
    ys = [lm.y * frame_h for lm in landmarks]
    x_min, x_max = int(min(xs)) - margin, int(max(xs)) + margin
    y_min, y_max = int(min(ys)) - margin, int(max(ys)) + margin
    return max(0, x_min), max(0, y_min), min(frame_w, x_max), min(frame_h, y_max)


def load_classes(model_path):
    """Lee class_indices.json (junto al modelo) -> letras en el orden de salida del modelo."""
    json_path = Path(model_path).parent / "class_indices.json"
    if not json_path.exists():
        raise FileNotFoundError(f"no se encontró {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        mapping = {int(k): v for k, v in json.load(f).items()}
    if sorted(mapping) != list(range(len(mapping))):
        raise ValueError("class_indices.json no tiene índices consecutivos desde 0")
    return [mapping[i] for i in range(len(mapping))]


def load_classifier(model_path):
    """Carga el modelo Keras y sus clases, verificando que sean consistentes."""
    from tensorflow import keras

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"no se encontró el modelo {model_path}")
    classes = load_classes(model_path)
    model = keras.models.load_model(model_path, compile=False)
    try:
        n_out, n_in = int(model.output_shape[-1]), int(model.input_shape[-1])
    except Exception:  # noqa: BLE001 - algunos modelos no exponen las formas
        return model, classes
    if n_out != len(classes):
        raise ValueError(f"el modelo tiene {n_out} salidas pero class_indices.json {len(classes)} clases")
    if n_in != 63:
        raise ValueError(f"el modelo espera {n_in} entradas y se le dan 63 (21 landmarks x 3)")
    return model, classes


@dataclass
class Reading:
    """Resultado de procesar un cuadro."""
    landmarks: object = None                    # 21 landmarks de MediaPipe (o None)
    letter: str | None = None                   # letra aceptada (None si confianza <= umbral)
    raw_letter: str | None = None               # mejor letra aunque tenga poca confianza
    confidence: float = 0.0
    top3: list = field(default_factory=list)    # [(letra, prob), ...]
    fingers: list | None = None                 # [bool x5] extendido / doblado


class HandRecognizer:
    """MediaPipe HandLandmarker + modelo Keras. Si algo falla, pasa a modo demo."""

    def __init__(self, model_path=MODEL_PATH, mediapipe_path=HAND_MODEL_PATH, force_demo=False):
        self.demo_mode = False
        self.status = "Listo"
        self.classes = list(DEMO_CLASSES)
        self._last_ts = 0
        self._demo_start = time.time()
        self.landmarker = None
        self.model = None
        self._mp = None
        if force_demo:
            self.demo_mode = True
            self.status = "Modo demo (solicitado con --demo)"
        else:
            self._load(model_path, mediapipe_path)

    def _load(self, model_path, mediapipe_path):
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            if not Path(mediapipe_path).exists():
                raise FileNotFoundError(f"no se encontró {mediapipe_path}")
            self.model, self.classes = load_classifier(model_path)
            options = vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(mediapipe_path)),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=1,
                min_hand_detection_confidence=MIN_DETECTION_CONFIDENCE,
                min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
            )
            self.landmarker = vision.HandLandmarker.create_from_options(options)
            self._mp = mp
            self.status = f"Modelo: {Path(model_path).name} · {len(self.classes)} clases"
        except Exception as error:  # noqa: BLE001 - cualquier fallo -> demo (con aviso)
            self.demo_mode = True
            self.classes = list(DEMO_CLASSES)
            self.status = f"Modo demo: {error}"

    def process(self, frame_bgr) -> Reading:
        """Procesa un cuadro (ya volteado si corresponde) y devuelve un Reading."""
        if self.demo_mode:
            return self._process_demo()

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        ts = max(int(time.time() * 1000), self._last_ts + 1)   # debe ser creciente
        self._last_ts = ts
        result = self.landmarker.detect_for_video(mp_image, ts)

        if not result.hand_landmarks:
            return Reading()

        landmarks = result.hand_landmarks[0]
        h, w = frame_bgr.shape[:2]
        x = normalize_landmarks(landmarks).astype("float32").reshape(1, -1)
        probs = self.model(x, training=False).numpy()[0]
        order = np.argsort(probs)[::-1][:3]
        top3 = [(self.classes[i], float(probs[i])) for i in order]
        best, conf = top3[0]
        return Reading(
            landmarks=landmarks,
            letter=best if conf > MIN_CONFIDENCE else None,
            raw_letter=best,
            confidence=conf,
            top3=top3,
            fingers=finger_states(landmarks, aspect=w / h),
        )

    def reset_demo_clock(self):
        self._demo_start = time.time()

    def _process_demo(self) -> Reading:
        """Datos simulados: cambia cada 2 s para probar estabilizador e interfaz."""
        n = len(self.classes)
        i = int((time.time() - self._demo_start) // 2) % n
        letter = self.classes[i]
        top3 = [(letter, 0.86), (self.classes[(i + 1) % n], 0.09), (self.classes[(i + 2) % n], 0.03)]
        fingers = [bool((i >> k) & 1) for k in range(5)]
        return Reading(None, letter, letter, 0.86, top3, fingers)

    def close(self):
        if self.landmarker is not None:
            self.landmarker.close()


class Stabilizer:
    """Misma regla del script original: solo se registran predicciones aceptadas
    (confianza > 0.6) y la letra es estable si ocupa >= 70 % de las últimas N."""

    def __init__(self, window=WINDOW_SIZE, threshold=STABILITY_THRESHOLD):
        self.buffer = deque(maxlen=window)
        self.needed = max(1, int(window * threshold))

    def push(self, letter):
        if letter is not None:
            self.buffer.append(letter)
        return self.stable()

    def stable(self):
        if len(self.buffer) < self.buffer.maxlen:
            return None
        value, count = Counter(self.buffer).most_common(1)[0]
        return value if count >= self.needed else None

    def progress(self):
        """0.0 - 1.0: qué tan cerca está la letra dominante de ser 'estable'."""
        if not self.buffer:
            return 0.0
        _, count = Counter(self.buffer).most_common(1)[0]
        return min(len(self.buffer) / self.buffer.maxlen, count / self.needed, 1.0)

    def reset(self):
        self.buffer.clear()


def hex_to_bgr(color):
    return tuple(int(color[i:i + 2], 16) for i in (5, 3, 1))


def draw_landmarks(frame, landmarks):
    """Dibuja la mano sobre el cuadro, un color por dedo (modifica `frame`)."""
    h, w = frame.shape[:2]
    pts = [(int(p.x * w), int(p.y * h)) for p in landmarks]
    for a, b in PALM_LINKS:
        cv2.line(frame, pts[a], pts[b], (170, 170, 170), 2)
    for chain, color in zip(FINGER_CHAINS, FINGER_COLORS):
        bgr = hex_to_bgr(color)
        for a, b in zip(chain, chain[1:]):
            cv2.line(frame, pts[a], pts[b], bgr, 3)
        for idx in chain[:-1]:
            cv2.circle(frame, pts[idx], 4, bgr, -1)
        cv2.circle(frame, pts[chain[-1]], 7, bgr, -1)             # punta del dedo
        cv2.circle(frame, pts[chain[-1]], 7, (255, 255, 255), 1)


def fit_cover(frame, w, h):
    """Recorta al centro con la proporción w:h y redimensiona (sin deformar)."""
    fh, fw = frame.shape[:2]
    target = w / h
    if fw / fh > target:
        new_w = int(fh * target)
        x0 = (fw - new_w) // 2
        frame = frame[:, x0:x0 + new_w]
    else:
        new_h = int(fw / target)
        y0 = (fh - new_h) // 2
        frame = frame[y0:y0 + new_h, :]
    return cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)


# =============================================================================
# 3. COMPONENTES DE INTERFAZ
# =============================================================================
def blend(c1, c2, t):
    """Mezcla dos colores hex (t=0 -> c1, t=1 -> c2). Útil para estados hover."""
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#%02x%02x%02x" % tuple(round(x + (y - x) * t) for x, y in zip(a, b))


def round_rect(canvas, x1, y1, x2, y2, r, **kw):
    """Rectángulo con esquinas redondeadas dibujado sobre un Canvas."""
    pts = [x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r, x2, y2 - r, x2, y2,
           x2 - r, y2, x1 + r, y2, x1, y2, x1, y2 - r, x1, y1 + r, x1, y1]
    return canvas.create_polygon(pts, smooth=True, **kw)


class FlatButton(tk.Label):
    """Botón plano con efecto hover. Estilos: 'primary', 'ghost', 'outline'."""

    def __init__(self, parent, text, command, style="ghost", padx=12, pady=7):
        self._command = command
        super().__init__(parent, text=text, padx=padx, pady=pady,
                         font=(FONT, 10, "bold"), cursor="hand2")
        self.restyle(style)
        self.bind("<Enter>", lambda e: self.config(bg=self._hover))
        self.bind("<Leave>", lambda e: self.config(bg=self._bg))
        self.bind("<Button-1>", lambda e: self._command())

    def restyle(self, style):
        palette = {
            "primary": (T["accent"], T["accent_text"]),
            "ghost": (T["surface2"], T["text"]),
            "outline": (T["surface2"], T["accent"]),
        }
        self._bg, fg = palette[style]
        self._hover = blend(self._bg, T["text"], 0.14)
        self.config(bg=self._bg, fg=fg)

    def set_text(self, text):
        self.config(text=text)


# =============================================================================
# 4. INTERFAZ
# =============================================================================
class App:
    def __init__(self, root, model_path=MODEL_PATH, mediapipe_path=HAND_MODEL_PATH,
                 cam_index=CAMERA_INDEX, stable_frames=WINDOW_SIZE,
                 background=BACKGROUND, demo=False):
        self.root = root
        self.root.title("Traductor LSM")
        self.root.configure(bg=T["bg"])
        self._place_window()

        self.cam_index = cam_index
        self.background = background
        self.demo_requested = demo
        self.recognizer = HandRecognizer(model_path, mediapipe_path, force_demo=demo)
        self.stabilizer = Stabilizer(window=stable_frames)
        self.cap = None
        self.running = False
        self.preview_visible = True
        self.pinned = False

        self.word = ""
        self.reading = Reading()
        self.stable_letter = None
        self._flash = 0
        self._fps = 0.0
        self._last_frame_time = time.time()
        self._photo = None
        self._has_messages = False
        self._at_bottom = True

        # Máscara para esquinas redondeadas del video
        self._mask = Image.new("L", (VIDEO_W, VIDEO_H), 0)
        ImageDraw.Draw(self._mask).rounded_rectangle(
            (0, 0, VIDEO_W - 1, VIDEO_H - 1), radius=16, fill=255)

        self._build_ui()
        self._bind_keys()
        self._refresh_ui()
        if START_PINNED:
            self.toggle_pin()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        if self.recognizer.demo_mode and not self.demo_requested:
            self.root.after(400, self._warn_demo)

    def _warn_demo(self):
        messagebox.showwarning(
            "Modo demo",
            "No se pudo cargar el reconocimiento real:\n\n"
            f"{self.recognizer.status}\n\n"
            "La app abrirá con letras SIMULADAS (solo sirve para ver la interfaz).\n"
            "Ejecútala desde la carpeta del proyecto o indica las rutas con "
            "--model y --mediapipe_model.")

    def _place_window(self):
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        height = min(800, sh - 90)
        x = sw - PANEL_W - 8 if DOCK_RIGHT else 40
        self.root.geometry(f"{PANEL_W}x{height}+{x}+8")
        self.root.minsize(PANEL_W, 640)
        self.root.resizable(False, True)

    # ---------- Construcción de la interfaz (modificar aquí el diseño) ----------
    def _build_ui(self):
        outer = tk.Frame(self.root, bg=T["bg"], padx=PAD, pady=PAD)
        outer.pack(fill="both", expand=True)
        # El orden importa: lo que va abajo se empaqueta antes del chat (que se expande)
        self._build_header(outer)
        self._build_preview(outer)
        self._build_footer(outer)
        self._build_composer(outer)
        self._build_chat(outer)

    def _build_header(self, parent):
        self.header = tk.Frame(parent, bg=T["bg"])
        self.header.pack(fill="x")

        left = tk.Frame(self.header, bg=T["bg"])
        left.pack(side="left")
        self.dot = tk.Label(left, text="●", font=(FONT, 10), bg=T["bg"], fg=T["muted"])
        self.dot.pack(side="left", padx=(0, 8))
        tk.Label(left, text="Traductor LSM", font=(FONT, 13, "bold"),
                 bg=T["bg"], fg=T["text"]).pack(side="left")

        right = tk.Frame(self.header, bg=T["bg"])
        right.pack(side="right")
        self.pin_btn = FlatButton(right, "Fijar", self.toggle_pin, "ghost", padx=10, pady=5)
        self.pin_btn.pack(side="left", padx=(0, 6))
        self.cam_btn = FlatButton(right, "Iniciar", self.toggle_camera, "primary", padx=12, pady=5)
        self.cam_btn.pack(side="left")

    def _build_preview(self, parent):
        # Fila de título (siempre visible) + cuerpo (se puede ocultar)
        self.preview_row = tk.Frame(parent, bg=T["bg"])
        self.preview_row.pack(fill="x", pady=(14, 6))
        tk.Label(self.preview_row, text="Seguimiento de la mano", font=(FONT, 9),
                 bg=T["bg"], fg=T["muted"]).pack(side="left")
        self.toggle_preview_lbl = tk.Label(self.preview_row, text="Ocultar", font=(FONT, 9, "bold"),
                                           bg=T["bg"], fg=T["accent"], cursor="hand2")
        self.toggle_preview_lbl.pack(side="right")
        self.toggle_preview_lbl.bind("<Button-1>", lambda e: self.toggle_preview())
        self.bg_lbl = tk.Label(self.preview_row, font=(FONT, 9, "bold"), bg=T["bg"],
                               fg=T["accent"], cursor="hand2")
        self.bg_lbl.pack(side="right", padx=(0, 14))
        self.bg_lbl.bind("<Button-1>", lambda e: self.toggle_background())
        self._update_bg_label()

        self.preview_body = tk.Frame(parent, bg=T["bg"])
        self.preview_body.pack(fill="x", after=self.preview_row)

        c = tk.Canvas(self.preview_body, width=VIDEO_W, height=VIDEO_H,
                      bg=T["bg"], highlightthickness=0)
        c.pack()
        self.video_canvas = c
        self._video_item = c.create_image(0, 0, anchor="nw")

        # Estado "cámara apagada"
        self._ph_bg = round_rect(c, 0, 0, VIDEO_W, VIDEO_H, 16, fill=T["surface"], outline="")
        self._ph_text = c.create_text(VIDEO_W // 2, VIDEO_H // 2,
                                      text="Cámara apagada\nPulsa «Iniciar»",
                                      font=(FONT, 11), fill=T["muted"], justify="center")

        # Chip con la letra detectada (esquina inferior izquierda)
        self._chip_bg = round_rect(c, 12, VIDEO_H - 62, 70, VIDEO_H - 12, 14,
                                   fill=T["surface2"], outline="")
        self._chip_text = c.create_text(41, VIDEO_H - 37, text="–",
                                        font=(FONT, 22, "bold"), fill=T["text"])
        # Aviso visible de modo demo
        self._demo_text = c.create_text(VIDEO_W // 2, 16, text="MODO DEMO · letras simuladas",
                                        font=(FONT, 9, "bold"), fill=T["warn"])
        self._set_overlay_visible(False)

        # Barra de estabilidad
        self.stab = tk.Canvas(self.preview_body, width=VIDEO_W, height=5,
                              bg=T["surface2"], highlightthickness=0)
        self.stab.pack(pady=(8, 0))
        self._stab_bar = self.stab.create_rectangle(0, 0, 0, 5, fill=T["accent"], width=0)

        # Tracking de dedos: una "píldora" por dedo, del mismo color que en la mano
        fc = tk.Canvas(self.preview_body, width=VIDEO_W, height=24,
                       bg=T["bg"], highlightthickness=0)
        fc.pack(pady=(10, 0))
        self.fingers_canvas = fc
        gap = 6
        cell = (VIDEO_W - gap * 4) / 5
        self._finger_items = []
        for i, name in enumerate(FINGER_NAMES):
            x1 = i * (cell + gap)
            x2 = x1 + cell
            bg = round_rect(fc, x1, 0, x2, 24, 8, fill=T["surface2"], outline="")
            tx = fc.create_text((x1 + x2) / 2, 12, text=name,
                                font=(FONT, 8, "bold"), fill=T["muted"])
            self._finger_items.append((bg, tx))

        # Top-3 de predicciones del modelo
        top = tk.Frame(self.preview_body, bg=T["bg"])
        top.pack(fill="x", pady=(8, 0))
        tk.Label(top, text="Predicción", font=(FONT, 8), bg=T["bg"], fg=T["muted"]).pack(side="left")
        self.top_labels = []
        for _ in range(3):
            lbl = tk.Label(top, text="–", font=(FONT, 9, "bold"), bg=T["bg"], fg=T["muted"])
            lbl.pack(side="left", padx=(12, 0))
            self.top_labels.append(lbl)

    def _build_chat(self, parent):
        wrap = tk.Frame(parent, bg=T["bg"])
        wrap.pack(fill="both", expand=True, pady=(14, 10))
        tk.Label(wrap, text="Conversación", font=(FONT, 9), bg=T["bg"],
                 fg=T["muted"]).pack(anchor="w", pady=(0, 6))

        holder = tk.Frame(wrap, bg=T["surface"])
        holder.pack(fill="both", expand=True)
        self.chat_canvas = tk.Canvas(holder, bg=T["surface"], highlightthickness=0)
        self.chat_canvas.pack(fill="both", expand=True)
        self.chat_inner = tk.Frame(self.chat_canvas, bg=T["surface"], padx=10, pady=10)
        self._chat_win = self.chat_canvas.create_window((0, 0), window=self.chat_inner, anchor="nw")

        self.chat_inner.bind("<Configure>", lambda e: self.chat_canvas.configure(
            scrollregion=self.chat_canvas.bbox("all")))
        self.chat_canvas.bind("<Configure>", self._on_chat_resize)
        self.chat_canvas.bind_all("<MouseWheel>", self._on_chat_wheel)

        self.empty_lbl = tk.Label(self.chat_inner, text="Las palabras que envíes\naparecerán aquí.",
                                  font=(FONT, 10), bg=T["surface"], fg=T["muted"], justify="center")
        self.empty_lbl.pack(pady=40, fill="x")

    def _on_chat_resize(self, event):
        self.chat_canvas.itemconfigure(self._chat_win, width=event.width)
        if self._at_bottom:     # si el usuario estaba al final, seguir ahí
            self.chat_canvas.after_idle(lambda: self.chat_canvas.yview_moveto(1.0))

    def _on_chat_wheel(self, event):
        self.chat_canvas.yview_scroll(int(-event.delta / 120), "units")
        self._at_bottom = self.chat_canvas.yview()[1] >= 0.999

    def _build_composer(self, parent):
        box = tk.Frame(parent, bg=T["surface"], padx=14, pady=12)
        box.pack(side="bottom", fill="x")

        self.hint_lbl = tk.Label(box, text="", font=(FONT, 9), bg=T["surface"],
                                 fg=T["muted"], anchor="w")
        self.hint_lbl.pack(fill="x")

        self.word_lbl = tk.Label(box, text="", font=(FONT, 20, "bold"), bg=T["surface"],
                                 fg=T["text"], anchor="w", justify="left",
                                 wraplength=PANEL_W - 2 * PAD - 28)
        self.word_lbl.pack(fill="x", pady=(2, 10))

        FlatButton(box, "Añadir letra", self.confirm_letter, "primary").pack(fill="x")

        row = tk.Frame(box, bg=T["surface"])
        row.pack(fill="x", pady=(6, 0))
        for col, (text, cmd) in enumerate([("Borrar", self.delete_letter),
                                           ("Limpiar", self.clear_word),
                                           ("Enviar", self.send_word)]):
            row.columnconfigure(col, weight=1, uniform="b")
            FlatButton(row, text, cmd, "ghost", pady=6).grid(
                row=0, column=col, sticky="ew", padx=(0 if col == 0 else 3, 0 if col == 2 else 3))

    def _build_footer(self, parent):
        footer = tk.Frame(parent, bg=T["bg"])
        footer.pack(side="bottom", fill="x", pady=(10, 0))
        self.status_var = tk.StringVar(value=self.recognizer.status)
        self.status_lbl = tk.Label(footer, textvariable=self.status_var, font=(FONT, 8), bg=T["bg"],
                                   fg=T["warn"] if self.recognizer.demo_mode else T["muted"],
                                   anchor="w", justify="left", wraplength=PANEL_W - 2 * PAD)
        self.status_lbl.pack(fill="x")
        tk.Label(footer, text="Espacio: añadir  ·  Retroceso: borrar\nEnter: enviar  ·  C: limpiar",
                 font=(FONT, 8), bg=T["bg"], fg=T["muted"], anchor="w",
                 wraplength=PANEL_W - 2 * PAD, justify="left").pack(fill="x")

    def _bind_keys(self):
        self.root.bind("<space>", lambda e: self.confirm_letter())
        self.root.bind("<BackSpace>", lambda e: self.delete_letter())
        self.root.bind("<Return>", lambda e: self.send_word())
        self.root.bind("<c>", lambda e: self.clear_word())
        self.root.bind("<C>", lambda e: self.clear_word())
        self.root.bind("<Escape>", lambda e: self.on_close())

    # ---------- Ventana ----------
    def toggle_pin(self):
        self.pinned = not self.pinned
        self.root.attributes("-topmost", self.pinned)
        self.pin_btn.set_text("Fijado" if self.pinned else "Fijar")
        self.pin_btn.restyle("outline" if self.pinned else "ghost")

    def toggle_preview(self):
        self.preview_visible = not self.preview_visible
        if self.preview_visible:
            self.preview_body.pack(fill="x", after=self.preview_row)
            self.toggle_preview_lbl.config(text="Ocultar")
        else:
            self.preview_body.pack_forget()
            self.toggle_preview_lbl.config(text="Mostrar")

    def toggle_background(self):
        self.background = "video" if self.background == "black" else "black"
        self._update_bg_label()

    def _update_bg_label(self):
        self.bg_lbl.config(text="Fondo: negro" if self.background == "black" else "Fondo: video")

    # ---------- Cámara ----------
    def toggle_camera(self):
        if self.running:
            self.stop_camera()
        else:
            self.start_camera()

    def start_camera(self):
        cap = cv2.VideoCapture(self.cam_index)
        if cap.isOpened():
            self.cap = cap
        else:
            cap.release()
            if not self.recognizer.demo_mode:
                messagebox.showerror(
                    "Cámara",
                    "No se pudo abrir la cámara.\nCierra otras apps que la estén usando "
                    "(Meet, Zoom...), revisa los permisos de Windows o prueba otro "
                    "índice con --cam_index.")
                return
            self.cap = None          # demo sin cámara: se usa un cuadro sintético
        self.running = True
        self.recognizer.reset_demo_clock()
        self.cam_btn.set_text("Detener")
        self.cam_btn.restyle("ghost")
        self.stabilizer.reset()
        self.video_canvas.itemconfigure(self._ph_bg, state="hidden")
        self.video_canvas.itemconfigure(self._ph_text, state="hidden")
        self._set_overlay_visible(True)
        self._update_frame()

    def stop_camera(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.cam_btn.set_text("Iniciar")
        self.cam_btn.restyle("primary")
        self.video_canvas.itemconfigure(self._video_item, image="")
        self.video_canvas.itemconfigure(self._ph_bg, state="normal")
        self.video_canvas.itemconfigure(self._ph_text, state="normal")
        self._set_overlay_visible(False)
        self.reading = Reading()
        self.stable_letter = None
        self._refresh_ui()
        self.status_var.set(self.recognizer.status)

    def _read_frame(self):
        if self.cap is None:     # modo demo sin cámara
            frame = np.full((480, 640, 3), 40, np.uint8)
            cv2.putText(frame, "SIN CAMARA (DEMO)", (140, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (160, 160, 160), 2)
            return cv2.flip(frame, 1) if MIRROR_CAMERA else frame   # se volteará de nuevo abajo
        ok, frame = self.cap.read()
        return frame if ok else None

    def _update_frame(self):
        if not self.running:
            return
        try:
            frame = self._read_frame()
            if frame is not None:
                if MIRROR_CAMERA:                       # igual que cv2.flip(frame, 1) del original
                    frame = cv2.flip(frame, 1)
                self.reading = self.recognizer.process(frame)
                self.stable_letter = self.stabilizer.push(self.reading.letter)

                display = np.zeros_like(frame) if self.background == "black" else frame
                lms = self.reading.landmarks
                if lms is not None:
                    h, w = frame.shape[:2]
                    draw_landmarks(display, lms)
                    x0, y0, x1, y1 = get_hand_bbox(lms, w, h)
                    cv2.rectangle(display, (x0, y0), (x1, y1), hex_to_bgr(T["accent"]), 2)
                if self.preview_visible:
                    self._render_frame(display)
                if self._flash > 0:
                    self._flash -= 1
                self._refresh_ui()
                self._update_fps()
        except Exception as error:  # noqa: BLE001 - no detener el ciclo por un cuadro malo
            self.status_var.set(f"Error en cuadro: {error}")
        self.root.after(FRAME_DELAY_MS, self._update_frame)

    # ---------- Render y refresco ----------
    def _render_frame(self, frame_bgr):
        frame_bgr = fit_cover(frame_bgr, VIDEO_W, VIDEO_H)
        img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
        img.putalpha(self._mask)                                  # esquinas redondeadas
        self._photo = ImageTk.PhotoImage(img)                     # evitar recolección
        self.video_canvas.itemconfigure(self._video_item, image=self._photo)
        self.video_canvas.tag_lower(self._video_item)

    def _set_overlay_visible(self, visible):
        state = "normal" if visible else "hidden"
        self.video_canvas.itemconfigure(self._chip_bg, state=state)
        self.video_canvas.itemconfigure(self._chip_text, state=state)
        demo_state = "normal" if (visible and self.recognizer.demo_mode) else "hidden"
        self.video_canvas.itemconfigure(self._demo_text, state=demo_state)

    def _refresh_ui(self):
        r = self.reading
        stable = self.stable_letter is not None

        # Chip sobre el video: letra, "?" si hay mano pero poca confianza, "–" si no hay mano
        chip = r.letter or ("?" if r.raw_letter else "–")
        self.video_canvas.itemconfigure(self._chip_bg, fill=T["accent"] if stable else T["surface2"])
        self.video_canvas.itemconfigure(self._chip_text, text=chip,
                                        fill=T["accent_text"] if stable else T["text"])

        # Barra de estabilidad
        self.stab.coords(self._stab_bar, 0, 0, VIDEO_W * self.stabilizer.progress(), 5)

        # Píldoras de dedos (mismo color que el dedo dibujado en la mano)
        for i, (bg, tx) in enumerate(self._finger_items):
            on = bool(self.running and r.fingers and r.fingers[i])
            self.fingers_canvas.itemconfigure(bg, fill=FINGER_COLORS[i] if on else T["surface2"])
            self.fingers_canvas.itemconfigure(tx, fill="#10131a" if on else T["muted"])

        # Top-3 de predicciones
        for k, lbl in enumerate(self.top_labels):
            if self.running and k < len(r.top3):
                letter, prob = r.top3[k]
                color = T["muted"] if k else (T["accent"] if r.letter else T["warn"])
                lbl.config(text=f"{letter} {prob * 100:.0f}%", fg=color)
            else:
                lbl.config(text="–", fg=T["muted"])

        # Indicador de estado (punto del encabezado)
        if not self.running:
            self.dot.config(fg=T["muted"])
        else:
            self.dot.config(fg=T["warn"] if self.recognizer.demo_mode else T["ok"])

        # Pista y palabra en construcción
        if not self.running:
            hint, color = "Cámara apagada", T["muted"]
        elif stable:
            hint, color = f"Letra lista: {self.stable_letter}  ·  Espacio para añadir", T["accent"]
        elif r.letter:
            hint, color = f"Detectando: {r.letter}…", T["muted"]
        elif r.raw_letter:
            hint, color = f"Poca confianza ({r.confidence * 100:.0f}%): ¿{r.raw_letter}?", T["warn"]
        else:
            hint, color = "Muestra tu mano a la cámara", T["muted"]
        self.hint_lbl.config(text=hint, fg=color)
        if self.word:
            self.word_lbl.config(text=self.word, fg=T["accent"] if self._flash > 0 else T["text"])
        else:
            self.word_lbl.config(text="Tu mensaje…", fg=T["muted"])

    def _update_fps(self):
        now = time.time()
        dt = now - self._last_frame_time
        self._last_frame_time = now
        if dt > 0:
            self._fps = 0.9 * self._fps + 0.1 * (1 / dt)
        self.status_var.set(f"{self.recognizer.status}  ·  {self._fps:.0f} FPS")

    # ---------- Construcción de la palabra ----------
    def confirm_letter(self):
        if self.stable_letter:
            self.word += self.stable_letter
            self.stabilizer.reset()      # igual que recent_predictions.clear() del original
            self.stable_letter = None
            self._flash = FLASH_FRAMES   # destello de la palabra
            self._refresh_ui()

    def delete_letter(self):
        self.word = self.word[:-1]
        self._refresh_ui()

    def clear_word(self):
        self.word = ""
        self.stabilizer.reset()
        self._refresh_ui()

    def send_word(self):
        """Envía la palabra al historial como un mensaje (burbuja)."""
        if not self.word:
            return
        self._add_bubble(self.word)
        self.word = ""
        self.stabilizer.reset()
        self._refresh_ui()

    def _add_bubble(self, text):
        if not self._has_messages:
            self.empty_lbl.destroy()
            self._has_messages = True
        row = tk.Frame(self.chat_inner, bg=T["surface"])
        row.pack(fill="x", pady=(0, 8))
        tk.Label(row, text=text, font=(FONT, 12), bg=T["bubble"], fg=T["bubble_text"],
                 padx=14, pady=8, wraplength=240, justify="left").pack(anchor="e")
        tk.Label(row, text=time.strftime("%H:%M"), font=(FONT, 8), bg=T["surface"],
                 fg=T["muted"]).pack(anchor="e", pady=(2, 0))
        self.root.update_idletasks()
        self.chat_canvas.yview_moveto(1.0)

    # ---------- Cierre ----------
    def on_close(self):
        self.stop_camera()
        self.recognizer.close()
        self.root.destroy()


def parse_args():
    p = argparse.ArgumentParser(description="Traductor LSM en tiempo real (panel lateral)")
    p.add_argument("--model", default=str(MODEL_PATH))
    p.add_argument("--mediapipe_model", default=str(HAND_MODEL_PATH))
    p.add_argument("--background", choices=["black", "video"], default=BACKGROUND)
    p.add_argument("--stable_frames", type=int, default=WINDOW_SIZE)
    p.add_argument("--cam_index", type=int, default=CAMERA_INDEX)
    p.add_argument("--theme", choices=list(THEMES), default=THEME_NAME)
    p.add_argument("--demo", action="store_true", help="solo interfaz, con letras simuladas")
    return p.parse_args()


def main():
    args = parse_args()
    T.update(THEMES[args.theme])
    root = tk.Tk()
    App(root, model_path=args.model, mediapipe_path=args.mediapipe_model,
        cam_index=args.cam_index, stable_frames=args.stable_frames,
        background=args.background, demo=args.demo)
    root.mainloop()


if __name__ == "__main__":
    main()