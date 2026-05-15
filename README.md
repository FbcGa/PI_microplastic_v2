# Detector de Microplásticos en Video

Sistema de visión por computadora para detectar y rastrear partículas de microplásticos en videos, usando OpenCV y Python.

## Requisitos

- Python 3.11+
- [UV](https://docs.astral.sh/uv/) (recomendado) o pip

## Instalación

### En Raspberry Pi con Thonny (recomendado)

Abre una terminal en la Pi e instala las dependencias:

```bash
pip install numpy opencv-python
```

Luego abre `main.py` en Thonny y presiona **Run (F5)**.

> Alternativa desde Thonny: Herramientas → Gestionar paquetes → busca e instala `numpy` y `opencv-python`.

### Con UV (PC/desarrollo)

```bash
uv sync
```

### Con pip (PC)

```bash
pip install -r requirements.txt
```

## Uso

1. Coloca el archivo de video en la misma carpeta que `main.py`.
2. Edita las constantes al inicio de `main.py` según tu video:

```python
VIDEO_PATH = os.path.join(_DIR, "tu_video.mp4")  # nombre del archivo de video
START_SEC  = 30                                   # segundo de inicio del análisis
STOP_SEC   = 60                                   # segundo de fin del análisis
```

3. Ejecuta el script principal:

```bash
python main.py
# o desde Thonny: presiona F5
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
