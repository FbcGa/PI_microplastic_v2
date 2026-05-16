import csv
import math
import cv2
import numpy as np
from detector import detect_particles, Detection, DEBUG
from tracker import CentroidTracker

VIDEO_PATH = "test.mp4"
START_SEC = 15
STOP_SEC = 200
CSV_PATH = "resultados.csv"

CSV_COLUMNS = ["frame", "tiempo_seg", "id", "cx", "cy", "velocidad_px_frame", "area_px2"]


def _speed(trail) -> float:
    pts = list(trail)
    if len(pts) < 2:
        return 0.0
    return math.hypot(pts[-1][0] - pts[-2][0], pts[-1][1] - pts[-2][1])


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


def draw_trails(frame: np.ndarray, tracker: CentroidTracker, active_ids: set) -> None:
    for obj_id, trail in tracker.trails.items():
        if obj_id not in active_ids:
            continue
        pts = list(trail)
        for i in range(1, len(pts)):
            cv2.line(frame, pts[i - 1], pts[i], (0, 165, 255), 1)
        if len(pts) >= 2:
            cx, cy = pts[-1]
            cv2.putText(frame, f"{_speed(trail):.0f}px", (cx + 6, cy + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)


def process_frame(
    frame: np.ndarray, tracker: CentroidTracker
) -> tuple[dict, list[Detection]]:
    detections = detect_particles(frame)
    tracker.update(detections)
    return tracker.active_objects(), detections


_SKIP_KEYS = {ord("f"): 49, ord("g"): 99}


def _write_rows(writer, frame_num: int, time_sec: float, active: dict,
                tracker: CentroidTracker, area_map: dict) -> None:
    for obj_id, (cx, cy) in active.items():
        writer.writerow({
            "frame": frame_num,
            "tiempo_seg": round(time_sec, 3),
            "id": obj_id,
            "cx": cx,
            "cy": cy,
            "velocidad_px_frame": round(_speed(tracker.trails[obj_id]), 2),
            "area_px2": round(area_map.get((cx, cy), 0), 1),
        })


def _skip_frames(cap: cv2.VideoCapture, n: int, frame_num: int) -> int:
    for _ in range(n):
        if not cap.read()[0]:
            break
        frame_num += 1
    return frame_num


def _handle_key(cap: cv2.VideoCapture, frame_num: int) -> tuple[bool, int]:
    """Devuelve (salir, frame_num actualizado)."""
    wait_ms = 0 if DEBUG else 1
    key = cv2.waitKey(wait_ms) & 0xFF
    if key == ord("q"):
        return True, frame_num
    if DEBUG and key in _SKIP_KEYS:
        frame_num = _skip_frames(cap, _SKIP_KEYS[key], frame_num)
    return False, frame_num


def _run_loop(cap: cv2.VideoCapture, writer) -> CentroidTracker:
    tracker = CentroidTracker()
    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        time_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        if time_sec >= STOP_SEC:
            break

        active, detections = process_frame(frame, tracker)
        area_map = {d.centroid: d.area for d in detections}
        _write_rows(writer, frame_num, time_sec, active, tracker, area_map)

        draw_overlay(frame, active, tracker.total_count, len(active), frame_num)
        draw_trails(frame, tracker, set(active.keys()))
        if DEBUG:
            cv2.putText(frame, "SPACE: +1 frame | f: +50 | g: +100 | q: salir",
                        (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)
        cv2.imshow("Microplasticos", frame)

        frame_num += 1
        quit_requested, frame_num = _handle_key(cap, frame_num)
        if quit_requested:
            break
    return tracker


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"No se pudo abrir: {VIDEO_PATH}")
        return

    cap.set(cv2.CAP_PROP_POS_MSEC, START_SEC * 1000)
    with open(CSV_PATH, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        tracker = _run_loop(cap, writer)

    cap.release()
    cv2.destroyAllWindows()
    print(f"Total de microplasticos detectados: {tracker.total_count}")
    print(f"CSV guardado en: {CSV_PATH}")


if __name__ == "__main__":
    main()
