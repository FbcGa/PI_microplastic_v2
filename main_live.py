import csv
import math
import os
import subprocess
import time
import cv2
import numpy as np
from detector import detect_particles
from tracker import CentroidTracker

_DIR = os.path.dirname(os.path.abspath(__file__))

CAMERA_WIDTH  = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS    = 30
AWB_GAINS     = "3.0,1.8"   # ganancias AWB para la iluminación del experimento
DISPLAY_SCALE = 0.5          # escala de visualización (0.5 = mitad del tamaño original)
CSV_PATH = os.path.join(_DIR, "resultados.csv")

CSV_COLUMNS = ["frame", "tiempo_seg", "id", "cx", "cy", "velocidad_px_frame", "area_px2"]


def draw_overlay(frame: np.ndarray, objects: dict, total: int, current: int, frame_num: int) -> None:
    for obj_id, (cx, cy) in objects.items():
        cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
        cv2.putText(frame, f"#{obj_id}", (cx + 6, cy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(frame, f"En frame: {current}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
    cv2.putText(frame, f"Total acumulado: {total}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
    cv2.putText(frame, f"Frame: {frame_num}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)


def draw_trails(frame: np.ndarray, tracker: CentroidTracker) -> None:
    active_ids = set(tracker.active_objects().keys())
    for obj_id, trail in tracker.trails.items():
        if obj_id not in active_ids:
            continue
        pts = list(trail)
        for i in range(1, len(pts)):
            cv2.line(frame, pts[i - 1], pts[i], (0, 165, 255), 1)
        if len(pts) >= 2:
            speed = math.hypot(pts[-1][0] - pts[-2][0], pts[-1][1] - pts[-2][1])
            cx, cy = pts[-1]
            cv2.putText(frame, f"{speed:.0f}px", (cx + 6, cy + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)


def _speed(trail) -> float:
    pts = list(trail)
    if len(pts) < 2:
        return 0.0
    return math.hypot(pts[-1][0] - pts[-2][0], pts[-1][1] - pts[-2][1])


def main():
    frame_bytes = CAMERA_WIDTH * CAMERA_HEIGHT * 3 // 2  # YUV420

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

    tracker = CentroidTracker()
    frame_num = 0
    t0 = time.time()


    with open(CSV_PATH, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        while True:
            raw = proc.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break

            time_sec = round(time.time() - t0, 3)
            yuv = np.frombuffer(raw, dtype=np.uint8).reshape((CAMERA_HEIGHT * 3 // 2, CAMERA_WIDTH))
            frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)

            detections = detect_particles(frame)
            area_map = {d["centroid"]: d["area"] for d in detections}

            tracker.update(detections)
            active = tracker.active_objects()

            for obj_id, (cx, cy) in active.items():
                speed = _speed(tracker.trails[obj_id])
                area = area_map.get((cx, cy), 0)
                writer.writerow({
                    "frame": frame_num,
                    "tiempo_seg": round(time_sec, 3),
                    "id": obj_id,
                    "cx": cx,
                    "cy": cy,
                    "velocidad_px_frame": round(speed, 2),
                    "area_px2": round(area, 1),
                })

            draw_overlay(frame, active, tracker.total_count, len(active), frame_num)
            draw_trails(frame, tracker)

            cv2.imshow("Microplasticos", frame)

            frame_num += 1
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    proc.terminate()
    cv2.destroyAllWindows()
    print(f"Total de microplasticos detectados: {tracker.total_count}")
    print(f"CSV guardado en: {CSV_PATH}")


if __name__ == "__main__":
    main()
