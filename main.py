import csv
import math
import cv2
import numpy as np
from detector import detect_particles, get_debug_gray
from tracker import CentroidTracker

VIDEO_PATH = "12Ama_20Azu__20Rojas_20Blancas20260514_021507.mp4"
DISPLAY_SCALE = 0.5   # escala de visualización (0.5 = mitad del tamaño original)
START_SEC = 30
STOP_SEC = 60
CSV_PATH = "resultados.csv"

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
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"No se pudo abrir: {VIDEO_PATH}")
        return

    cap.set(cv2.CAP_PROP_POS_MSEC, START_SEC * 1000)
    tracker = CentroidTracker()
    frame_num = 0

    with open(CSV_PATH, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        while True:
            time_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if time_sec >= STOP_SEC:
                break

            ret, frame = cap.read()
            if not ret:
                break

            detections = detect_particles(frame)
            # Mapa centroide → área para enriquecer el CSV
            area_map = {d["centroid"]: d["area"] for d in detections}

            tracker.update(detections)
            active = tracker.active_objects()

            # Escribir una fila por partícula activa
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

            def show(name, img):
                h, w = img.shape[:2]
                small = cv2.resize(img, (int(w * DISPLAY_SCALE), int(h * DISPLAY_SCALE)))
                cv2.imshow(name, small)

            show("Microplasticos", frame)
            show("Debug: Gris", get_debug_gray(frame))

            debug_kp = frame.copy()
            for d in detections:
                cv2.circle(debug_kp, d["centroid"], 6, (0, 0, 255), 2)
            show("Debug: Keypoints", debug_kp)

            frame_num += 1
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Total de microplasticos detectados: {tracker.total_count}")
    print(f"CSV guardado en: {CSV_PATH}")


if __name__ == "__main__":
    main()
