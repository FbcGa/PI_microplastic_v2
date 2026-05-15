# Detector de Microplásticos en Video

Sistema de visión por computadora para detectar y rastrear partículas de microplásticos en videos, usando OpenCV y Python.

## Requisitos

- Python 3.11+
- [UV](https://docs.astral.sh/uv/) (recomendado) o pip

## Instalación

### Con UV (recomendado)

```bash
uv sync
```

### Con pip

```bash
pip install "numpy>=2.4.4" "opencv-python>=4.13.0.92"
```

## Uso

1. Coloca el archivo de video en la raíz del proyecto.
2. Edita las constantes al inicio de `main.py` según tu video:

```python
VIDEO_PATH = "tu_video.mp4"   # Ruta al video
START_SEC  = 30               # Segundo de inicio del análisis
STOP_SEC   = 60               # Segundo de fin del análisis
CSV_PATH   = "resultados.csv" # Archivo de salida
```

3. Ejecuta el script principal:

```bash
python main.py
```

Durante la ejecución se abrirán tres ventanas de OpenCV:

| Ventana | Descripción |
|---|---|
| `Microplasticos` | Video principal con partículas detectadas y sus IDs |
| `Debug: Gris` | Frame en escala de grises usado para la detección |
| `Debug: Keypoints` | Keypoints crudos detectados por el blob detector |

Presiona **Q** para detener el procesamiento en cualquier momento.

## Salida

Los resultados se guardan en `resultados.csv` con las siguientes columnas:

| Columna | Descripción |
|---|---|
| `frame` | Número de frame |
| `tiempo_seg` | Tiempo en segundos |
| `id` | ID único de la partícula rastreada |
| `cx`, `cy` | Coordenadas del centroide (píxeles) |
| `velocidad_px_frame` | Velocidad en píxeles por frame |
| `area_px2` | Área de la partícula en píxeles cuadrados |

## Estructura del proyecto

```
code_microplasticos/
├── main.py        # Punto de entrada principal
├── detector.py    # Detección de partículas con SimpleBlobDetector
├── tracker.py     # Rastreo de centroides entre frames
├── pyproject.toml # Configuración del proyecto y dependencias
└── resultados.csv # Resultado generado tras la ejecución
```

## Ajuste de parámetros

### Detección (`detector.py`)

| Parámetro | Valor por defecto | Descripción |
|---|---|---|
| `MIN_AREA` | `500` px² | Tamaño mínimo de partícula |
| `MAX_AREA` | `200000` px² | Tamaño máximo de partícula |
| `MIN_CIRCULARITY` | `0.1` | Circularidad mínima (0 = desactivado) |
| `BUBBLE_THRESHOLD` | `80` | Umbral para filtrar burbujas de aire |

### Rastreo (`tracker.py`)

| Parámetro | Valor por defecto | Descripción |
|---|---|---|
| `MAX_DISTANCE_MIN` | `350` px | Radio máximo de búsqueda entre frames |
| `VELOCITY_MARGIN` | `4.0` | Multiplicador del umbral basado en velocidad |
| `MAX_DISAPPEARED` | `60` frames | Frames antes de eliminar un track |
| `TRAIL_LEN` | `20` frames | Longitud del historial de posiciones |
