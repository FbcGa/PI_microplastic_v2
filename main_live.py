import csv
import math
import os
import subprocess
import time

import cv2
import numpy as np

from detector import detect_particles
from tracker import CentroidTracker
from arduino import ArduinoController
from overlay import draw_overlay, draw_trails

_DIR = os.path.dirname(os.path.abspath(__file__))

CAMERA_WIDTH  = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS    = 30
AWB_GAINS     = "3.0,1.8"
DISPLAY_SCALE = 0.5
CSV_PATH      = os.path.join(_DIR, "resultados.csv")
CSV_COLUMNS   = ["frame", "tiempo_seg", "id", "cx", "cy", "velocidad_px_frame", "area_px2"]


def _speed(trail):
    pts = list(trail)
    if len(pts) < 2:
        return 0.0
    return math.hypot(pts[-1][0] - pts[-2][0], pts[-1][1] - pts[-2][1])


def main():
    bomba = ArduinoController()
    bomba.conectar()
    bomba.iniciar_hilo_lectura()

    frame_bytes = CAMERA_WIDTH * CAMERA_HEIGHT * 3 // 2

    proc = subprocess.Popen([
        "libcamera-vid",
        "--timeout", "0",
        "--awb", "custom",
        "--awbgains", AWB_GAINS,
        "--framerate", str(CAMERA_FPS),
        "--width",  str(CAMERA_WIDTH),
        "--height", str(CAMERA_HEIGHT),
        "--codec", "yuv420",
        "--output", "-",
    ], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    tracker   = CentroidTracker()
    frame_num = 0
    t0        = time.time()

    print("\n========================================")
    print("SISTEMA INTEGRADO - UPC Lima 2025")
    print("Teclas en la ventana de video:")
    print("  S - iniciar bomba")
    print("  X - detener bomba")
    print("  + - subir caudal 10 ml/min")
    print("  - - bajar caudal 10 ml/min")
    print("  Q - salir")
    print("========================================\n")

    key_actions = {
        ord('s'): lambda: (bomba.iniciar(),  print("[Bomba] Iniciada")),
        ord('x'): lambda: (bomba.detener(),  print("[Bomba] Detenida")),
        ord('+'): lambda: print(f"[Bomba] Caudal: {bomba.set_caudal(bomba.caudal_actual + 10)} ml/min"),
        ord('='): lambda: print(f"[Bomba] Caudal: {bomba.set_caudal(bomba.caudal_actual + 10)} ml/min"),
        ord('-'): lambda: print(f"[Bomba] Caudal: {bomba.set_caudal(bomba.caudal_actual - 10)} ml/min"),
    }

    with open(CSV_PATH, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        while True:
            raw = proc.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break

            time_sec = round(time.time() - t0, 3)
            yuv   = np.frombuffer(raw, dtype=np.uint8).reshape((CAMERA_HEIGHT * 3 // 2, CAMERA_WIDTH))
            frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)

            detections = detect_particles(frame)
            area_map   = {d["centroid"]: d["area"] for d in detections}

            tracker.update(detections)
            active = tracker.active_objects()

            for obj_id, (cx, cy) in active.items():
                speed = _speed(tracker.trails[obj_id])
                area  = area_map.get((cx, cy), 0)
                writer.writerow({
                    "frame":               frame_num,
                    "tiempo_seg":          round(time_sec, 3),
                    "id":                  obj_id,
                    "cx":                  cx,
                    "cy":                  cy,
                    "velocidad_px_frame":  round(speed, 2),
                    "area_px2":            round(area, 1),
                })

            draw_overlay(frame, active, tracker.total_count, len(active), frame_num,
                         estado_bomba=bomba.estado())
            draw_trails(frame, tracker)

            cv2.imshow("Microplasticos - UPC Lima 2026", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q')):
                break
            action = key_actions.get(key) or key_actions.get(key | 0x20)
            if action:
                action()

            frame_num += 1

    bomba.cerrar()
    proc.terminate()
    cv2.destroyAllWindows()

    estado_final = bomba.estado()
    print(f"\nTotal microplasticos detectados: {tracker.total_count}")
    print(f"Volumen total procesado: {estado_final['volumen_ml']:.1f} ml / {estado_final['volumen_litros']:.4f} L")
    print(f"CSV guardado en: {CSV_PATH}")


if __name__ == "__main__":
    main()
