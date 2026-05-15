import cv2
import numpy as np

# --- Parámetros ajustables ---
BLUR_KERNEL = 5        # Tamaño del kernel Gaussiano (impar)
MIN_AREA = 500        # Área mínima en px² — excluye arena fina
MAX_AREA = 200000      # Área máxima en px² — excluye objetos grandes
MIN_CIRCULARITY = 0.1  # 0.0 = desactivado, 1.0 = círculo perfecto

# Filtro de burbujas: las burbujas tienen centro brillante en contraluz.
# Si el brillo medio dentro del blob supera este umbral, se descarta como burbuja.
# Rango: 0–255. Bajar si se descartan microplásticos reales, subir si pasan burbujas.
BUBBLE_THRESHOLD = 80


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


def get_debug_gray(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (BLUR_KERNEL, BLUR_KERNEL), 0)


def _mean_intensity(gray: np.ndarray, cx: int, cy: int, r: int) -> float:
    """Brillo medio dentro del círculo centrado en (cx, cy) con radio r."""
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.circle(mask, (cx, cy), r, 255, -1)
    return float(cv2.mean(gray, mask=mask)[0])


def detect_particles(frame: np.ndarray) -> list[dict]:
    """
    Detecta partículas oscuras sobre fondo claro (contraluz).
    Filtra arena por área y burbujas de aire por brillo interno.
    """
    gray = get_debug_gray(frame)
    keypoints = _detector.detect(gray)
    detections = []
    for kp in keypoints:
        cx, cy = int(kp.pt[0]), int(kp.pt[1])
        r = max(1, int(kp.size / 2))

        # Descarta burbujas: su centro brillante eleva el brillo medio
        if _mean_intensity(gray, cx, cy, r) > BUBBLE_THRESHOLD:
            continue

        detections.append({
            "bbox": (cx - r, cy - r, 2 * r, 2 * r),
            "centroid": (cx, cy),
            "area": kp.size ** 2,
        })
    return detections
