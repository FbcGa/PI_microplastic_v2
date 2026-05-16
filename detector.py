import math
from dataclasses import dataclass

import cv2
import numpy as np

# --- Parámetros ajustables ---
BLUR_KERNEL = 5        # Tamaño del kernel Gaussiano (impar)
MIN_AREA = 500         # Área mínima en px² — excluye arena fina
MAX_AREA = 200000      # Área máxima en px² — excluye objetos grandes
MIN_CIRCULARITY = 0.1  # 0.0 = desactivado, 1.0 = círculo perfecto

# Filtro de burbujas — centro brillante
BUBBLE_THRESHOLD = 80

# Filtro de anillo de burbuja — el borde oscuro rodea un halo brillante.
# HALO_MULT: cuántas veces el radio del blob se expande para buscar el halo.
# HALO_THRESHOLD / HALO_MAX: el halo de una burbuja es moderado (luz interior).
#   Un halo MUY alto indica microplástico cerca del fondo brillante, no burbuja.
# RING_CENTER_MIN: umbral mínimo del centro para activar el chequeo de halo.
BUBBLE_HALO_MULT = 2.5
BUBBLE_HALO_THRESHOLD = 100   # halo mínimo para sospechar burbuja
BUBBLE_HALO_MAX = 150         # halo máximo — por encima es fondo brillante, no burbuja
BUBBLE_RING_CENTER_MIN = 55   # centro mínimo para activar halo check

# Activa ventanas de debug por etapa (True/False)
DEBUG = True


@dataclass
class Detection:
    centroid: tuple[int, int]
    area: float
    bbox: tuple[int, int, int, int]


def _make_detector() -> cv2.SimpleBlobDetector:
    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = True
    params.blobColor = 0
    params.filterByArea = True
    params.minArea = MIN_AREA
    params.maxArea = MAX_AREA
    params.filterByCircularity = MIN_CIRCULARITY > 0
    params.minCircularity = MIN_CIRCULARITY
    params.filterByConvexity = False
    params.filterByInertia = False
    return cv2.SimpleBlobDetector_create(params)


_detector = _make_detector()


def _get_gray(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (BLUR_KERNEL, BLUR_KERNEL), 0)


def _mean_intensity(gray: np.ndarray, cx: int, cy: int, r: int) -> float:
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.circle(mask, (cx, cy), r, 255, -1)
    return float(cv2.mean(gray, mask=mask)[0])


def _ring_intensity(gray: np.ndarray, cx: int, cy: int, r_inner: int, r_outer: int) -> float:
    """Intensidad promedio solo en el anillo entre r_inner y r_outer."""
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.circle(mask, (cx, cy), r_outer, 255, -1)
    cv2.circle(mask, (cx, cy), r_inner, 0, -1)
    return float(cv2.mean(gray, mask=mask)[0])


def _classify(gray: np.ndarray, cx: int, cy: int, r: int) -> tuple[bool, float, float]:
    """Devuelve (es_burbuja, intensidad_centro, intensidad_halo).

    Una burbuja tiene borde oscuro y halo brillante alrededor.
    Un microplástico es oscuro también en la zona exterior.
    """
    inner = _mean_intensity(gray, cx, cy, r)
    halo_r = max(r + 1, int(r * BUBBLE_HALO_MULT))
    halo = _ring_intensity(gray, cx, cy, r, halo_r)
    # Halo de burbuja: moderado (luz interior de la burbuja).
    # Halo muy alto = microplástico cerca del fondo brillante → no es burbuja.
    halo_triggered = (inner >= BUBBLE_RING_CENTER_MIN
                      and BUBBLE_HALO_THRESHOLD < halo <= BUBBLE_HALO_MAX)
    is_bubble = inner > BUBBLE_THRESHOLD or halo_triggered
    return is_bubble, inner, halo


def _draw_debug(gray: np.ndarray, keypoints, detections: list["Detection"]) -> None:
    """Muestra 3 ventanas: gris+blur, clasificación detallada, resultado final."""
    cv2.imshow("DEBUG 1: Gris + Blur", gray)

    blobs_vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for kp in keypoints:
        cx, cy = int(kp.pt[0]), int(kp.pt[1])
        r = max(1, int(kp.size / 2))
        halo_r = max(r + 1, int(r * BUBBLE_HALO_MULT))
        is_bubble, inner, halo = _classify(gray, cx, cy, r)

        color = (0, 0, 255) if is_bubble else (0, 255, 0)
        # Círculo interior (zona muestreada como centro)
        cv2.circle(blobs_vis, (cx, cy), r, color, 2)
        # Círculo exterior (zona de halo)
        cv2.circle(blobs_vis, (cx, cy), halo_r, color, 1)
        # Punto en el centroide
        cv2.circle(blobs_vis, (cx, cy), 3, (255, 255, 0), -1)

        reason = ""
        if inner > BUBBLE_THRESHOLD:
            reason = "centro"
        elif inner >= BUBBLE_RING_CENTER_MIN and BUBBLE_HALO_THRESHOLD < halo <= BUBBLE_HALO_MAX:
            reason = "anillo"

        # Amarillo: pasó como microplástico pero halo era alto (fondo brillante)
        halo_too_high = (not is_bubble and halo > BUBBLE_HALO_MAX)
        display_color = (0, 200, 255) if halo_too_high else color

        tag = f"B({reason}) c:{inner:.0f} h:{halo:.0f}" if is_bubble else f"P c:{inner:.0f} h:{halo:.0f}"
        if halo_too_high:
            tag += " [fondo]"
        cv2.putText(blobs_vis, tag, (cx + halo_r + 2, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, display_color, 1)

    cv2.imshow("DEBUG 2: Blobs — c=centro h=halo (rojo=burbuja verde=plastico)", blobs_vis)

    final_vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for d in detections:
        cx, cy = d.centroid
        r = max(1, int(math.sqrt(d.area / math.pi)))
        cv2.circle(final_vis, (cx, cy), r, (0, 255, 0), 2)
        cv2.putText(final_vis, f"A:{d.area:.0f}", (cx + r + 2, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    cv2.putText(final_vis, f"Detectados: {len(detections)}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    cv2.imshow("DEBUG 3: Resultado final", final_vis)


def detect_particles(frame: np.ndarray) -> list[Detection]:
    """Detecta partículas oscuras sobre fondo claro (contraluz).
    Filtra arena por área y burbujas por brillo central o halo exterior."""
    gray = _get_gray(frame)
    keypoints = _detector.detect(gray)
    detections = []
    for kp in keypoints:
        cx, cy = int(kp.pt[0]), int(kp.pt[1])
        r = max(1, int(kp.size / 2))
        is_bubble, _, _ = _classify(gray, cx, cy, r)
        if is_bubble:
            continue
        detections.append(Detection(
            centroid=(cx, cy),
            area=math.pi * (kp.size / 2) ** 2,
            bbox=(cx - r, cy - r, 2 * r, 2 * r),
        ))

    if DEBUG:
        _draw_debug(gray, keypoints, detections)

    return detections
