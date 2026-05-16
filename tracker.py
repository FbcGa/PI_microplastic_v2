import math
from collections import OrderedDict, deque

from detector import Detection

MAX_DISTANCE_MIN = 350   # radio mínimo de búsqueda para tracks sin velocidad histórica
VELOCITY_MARGIN = 4.0    # multiplicador sobre la última velocidad observada
MAX_DISAPPEARED = 60     # frames sin detección antes de eliminar un track
RIGHTWARD_TOLERANCE = 100  # px — tolerancia de movimiento hacia la derecha (ruido de detección)
TRAIL_LEN = 20           # cantidad de posiciones anteriores que se guardan por track


class CentroidTracker:
    def __init__(self):
        self.next_id = 0
        self.objects: OrderedDict[int, tuple] = OrderedDict()
        self.disappeared: dict[int, int] = {}
        self.trails: dict[int, deque] = {}
        self.velocities: dict[int, float] = {}
        self.total_count = 0

    def _register(self, centroid: tuple) -> None:
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.trails[self.next_id] = deque([centroid], maxlen=TRAIL_LEN)
        self.velocities[self.next_id] = 0.0
        self.next_id += 1
        self.total_count += 1

    def _deregister(self, obj_id: int) -> None:
        del self.objects[obj_id]
        del self.disappeared[obj_id]
        del self.trails[obj_id]
        del self.velocities[obj_id]

    def _age_all(self) -> None:
        for obj_id in dict(self.disappeared):
            self.disappeared[obj_id] += 1
            if self.disappeared[obj_id] > MAX_DISAPPEARED:
                self._deregister(obj_id)

    def _match(self, obj_ids, obj_centroids, centroids) -> tuple[set, set]:
        D = [
            [_dist(oc, dc) if dc[0] <= oc[0] + RIGHTWARD_TOLERANCE else float("inf")
             for dc in centroids]
            for oc in obj_centroids
        ]
        pairs = sorted(
            ((D[r][c], r, c) for r in range(len(obj_centroids)) for c in range(len(centroids))),
            key=lambda x: x[0]
        )
        used_rows, used_cols = set(), set()
        for dist_val, row, col in pairs:
            if row in used_rows or col in used_cols:
                continue
            obj_id = obj_ids[row]
            max_d = max(MAX_DISTANCE_MIN, self.velocities[obj_id] * VELOCITY_MARGIN)
            if dist_val > max_d:
                continue
            self.objects[obj_id] = centroids[col]
            self.trails[obj_id].append(centroids[col])
            self.velocities[obj_id] = dist_val
            self.disappeared[obj_id] = 0
            used_rows.add(row)
            used_cols.add(col)
        return used_rows, used_cols

    def active_objects(self) -> dict[int, tuple]:
        return {oid: c for oid, c in self.objects.items() if self.disappeared[oid] == 0}

    def update(self, detections: list[Detection]) -> dict[int, tuple]:
        centroids = [d.centroid for d in detections]

        if not centroids:
            self._age_all()
            return dict(self.objects)

        if not self.objects:
            for c in centroids:
                self._register(c)
            return dict(self.objects)

        obj_ids = list(self.objects.keys())
        obj_centroids = list(self.objects.values())
        used_rows, used_cols = self._match(obj_ids, obj_centroids, centroids)

        for row in range(len(obj_centroids)):
            if row not in used_rows:
                obj_id = obj_ids[row]
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > MAX_DISAPPEARED:
                    self._deregister(obj_id)

        for col in range(len(centroids)):
            if col not in used_cols:
                self._register(centroids[col])

        return dict(self.objects)


def _dist(a: tuple, b: tuple) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])
