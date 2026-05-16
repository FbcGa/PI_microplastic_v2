import math
import cv2


def draw_overlay(frame, objects, total, current, frame_num, estado_bomba=None):
    for obj_id, (cx, cy) in objects.items():
        cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
        cv2.putText(frame, f"#{obj_id}", (cx + 6, cy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    # Panel izquierdo - deteccion
    cv2.putText(frame, f"En frame: {current}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
    cv2.putText(frame, f"Total: {total}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
    cv2.putText(frame, f"Frame: {frame_num}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

    # Panel bomba y sensor
    if estado_bomba:
        activo = estado_bomba['activo']
        estado = "ACTIVO" if activo else "DETENIDO"
        color_estado = (0, 255, 0) if activo else (0, 0, 255)

        cv2.putText(frame, f"Bomba: {estado}", (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_estado, 2)
        cv2.putText(frame, f"Caudal fijado: {estado_bomba['caudal_actual']:.0f} ml/min", (10, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Caudal sensor: {estado_bomba['caudal_sensor']:.1f} ml/min", (10, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Volumen: {estado_bomba['volumen_ml']:.1f} ml", (10, 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
        cv2.putText(frame, f"Litros: {estado_bomba['volumen_litros']:.4f} L", (10, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)

    cv2.putText(frame, "S=start  X=stop  +/-=caudal  Q=salir", (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)


def draw_trails(frame, tracker):
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
