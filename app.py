import streamlit as st
import cv2
import numpy as np
import pandas as pd

import os
import json
import time
import math
import queue
import hashlib
import tempfile
import threading
import traceback
import csv

from collections import deque, defaultdict
from datetime import datetime


# -----------------------------------------------------------------------------
# Optional heavy dependencies. The app must still run immediately in simulation
# mode if these are unavailable.
# -----------------------------------------------------------------------------
try:
    import torch
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False

try:
    from ultralytics import YOLO
    HAVE_ULTRALYTICS = True
except Exception:
    HAVE_ULTRALYTICS = False


st.set_page_config(
    page_title="Real-Time Multi-Object Detection & Tracking Studio",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------------------------------------
# Constants and small helpers
# -----------------------------------------------------------------------------
CLASS_COLORS = {
    "car": (255, 120, 30),
    "bus": (0, 200, 0),
    "truck": (0, 80, 255),
    "motorcycle": (180, 0, 200),
    "person": (0, 165, 255),
    "worker": (0, 165, 255),
    "forklift": (0, 255, 255),
    "pallet": (90, 60, 30),
    "cow": (230, 230, 230),
    "sheep": (210, 210, 210),
    "horse": (120, 80, 40),
    "tractor": (0, 128, 0),
    "machine": (120, 120, 120),
    "helmet": (0, 215, 255),
    "no_helmet": (0, 0, 255),
    "tool": (255, 0, 255),
}

VEHICLE_CLASSES = {"car", "bus", "truck", "motorcycle", "tractor", "forklift"}
PERSON_CLASSES = {"person", "worker", "helmet", "no_helmet"}
ANIMAL_CLASSES = {"cow", "sheep", "horse"}


def class_color(cls_name: str):
    if cls_name in CLASS_COLORS:
        return CLASS_COLORS[cls_name]
    digest = hashlib.md5(str(cls_name).encode("utf-8")).digest()
    return (int(digest[0]), int(digest[1]), int(digest[2]))


def centroid(bbox):
    x1, y1, x2, y2 = bbox
    return np.array([(float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0], dtype=float)


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = map(float, box_a)
    bx1, by1, bx2, by2 = map(float, box_b)
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 1e-9 else 0.0


def cosine_similarity(a, b):
    if a is None or b is None:
        return 0.0
    a = np.asarray(a, dtype=float).flatten()
    b = np.asarray(b, dtype=float).flatten()
    if a.size != b.size or a.size == 0:
        return 0.0
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _orient(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a, b, c, eps=1e-6):
    return (
        min(a[0], c[0]) - eps <= b[0] <= max(a[0], c[0]) + eps
        and min(a[1], c[1]) - eps <= b[1] <= max(a[1], c[1]) + eps
    )


def segments_intersect(a, b, c, d):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    d = np.asarray(d, dtype=float)

    o1 = _orient(a, b, c)
    o2 = _orient(a, b, d)
    o3 = _orient(c, d, a)
    o4 = _orient(c, d, b)

    if abs(o1) < 1e-9 and _on_segment(a, c, b):
        return True
    if abs(o2) < 1e-9 and _on_segment(a, d, b):
        return True
    if abs(o3) < 1e-9 and _on_segment(c, a, d):
        return True
    if abs(o4) < 1e-9 and _on_segment(c, b, d):
        return True

    return (o1 * o2 < 0) and (o3 * o4 < 0)


def line_cross_sign(line_pts, prev_pt, curr_pt):
    line_pts = np.asarray(line_pts, dtype=float)
    prev_pt = np.asarray(prev_pt, dtype=float)
    curr_pt = np.asarray(curr_pt, dtype=float)
    lv = line_pts[1] - line_pts[0]
    mv = curr_pt - prev_pt
    val = float(lv[0] * mv[1] - lv[1] * mv[0])
    if abs(val) < 1e-6:
        return 0
    return 1 if val > 0 else -1


def point_in_polygon(pt, poly):
    pt = (float(pt[0]), float(pt[1]))
    poly = np.asarray(poly, dtype=np.float32).reshape((-1, 1, 2))
    val = cv2.pointPolygonTest(poly, pt, False)
    return bool(val >= -1e-6)


def to_py(obj):
    if isinstance(obj, dict):
        return {str(k): to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_py(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (bool, int, float, str)) or obj is None:
        return obj
    return str(obj)


def stable_upload(file_obj, kind: str):
    if file_obj is None:
        return None
    buf = file_obj.getbuffer()
    digest = hashlib.md5(buf).hexdigest()
    key = f"uploaded_{kind}_{file_obj.name}_{file_obj.size}_{digest}"
    if key not in st.session_state:
        tmp_dir = tempfile.mkdtemp(prefix=f"cv_upload_{kind}_")
        path = os.path.join(tmp_dir, file_obj.name)
        with open(path, "wb") as f:
            f.write(buf)
        st.session_state[key] = path
    return st.session_state[key]


# -----------------------------------------------------------------------------
# Default scenario configurations
# -----------------------------------------------------------------------------
def default_geometry(scenario: str):
    if scenario == "traffic":
        lines = [
            {
                "name": "left_lanes_count",
                "points": [[0.50, 0.12], [0.50, 0.45]],
                "forbidden_sign": 1,
            },
            {
                "name": "right_lanes_count",
                "points": [[0.50, 0.55], [0.50, 0.88]],
                "forbidden_sign": -1,
            },
        ]
        zones = [
            {
                "name": "restricted_shoulder",
                "points": [[0.03, 0.72], [0.28, 0.72], [0.28, 0.96], [0.03, 0.96]],
                "loiter_seconds": 5.0,
                "trigger_missing_helmet": False,
            }
        ]
    elif scenario == "warehouse":
        lines = [
            {
                "name": "entry_line",
                "points": [[0.15, 0.10], [0.15, 0.90]],
                "forbidden_sign": None,
            }
        ]
        zones = [
            {
                "name": "restricted_rack",
                "points": [[0.55, 0.20], [0.92, 0.20], [0.92, 0.78], [0.55, 0.78]],
                "loiter_seconds": 7.0,
                "trigger_missing_helmet": False,
            }
        ]
    elif scenario == "agriculture":
        lines = [
            {
                "name": "gate_line",
                "points": [[0.45, 0.08], [0.45, 0.92]],
                "forbidden_sign": None,
            }
        ]
        zones = [
            {
                "name": "crop_protected_zone",
                "points": [[0.10, 0.55], [0.45, 0.55], [0.45, 0.92], [0.10, 0.92]],
                "loiter_seconds": 8.0,
                "trigger_missing_helmet": False,
            }
        ]
    else:  # safety
        lines = [
            {
                "name": "machine_boundary_line",
                "points": [[0.35, 0.15], [0.35, 0.85]],
                "forbidden_sign": None,
            }
        ]
        zones = [
            {
                "name": "machine_danger_zone",
                "points": [[0.45, 0.20], [0.88, 0.20], [0.88, 0.80], [0.45, 0.80]],
                "loiter_seconds": 4.0,
                "trigger_missing_helmet": True,
            }
        ]
    return lines, zones


def default_class_limits(scenario: str):
    if scenario == "traffic":
        return {"car": 60.0, "bus": 50.0, "truck": 45.0, "motorcycle": 70.0, "default": 50.0}
    if scenario == "warehouse":
        return {"person": 5.0, "worker": 5.0, "forklift": 10.0, "pallet": 1.0, "default": 8.0}
    if scenario == "agriculture":
        return {"person": 6.0, "tractor": 25.0, "cow": 5.0, "sheep": 4.0, "horse": 12.0, "default": 10.0}
    return {
        "person": 5.0,
        "worker": 5.0,
        "helmet": 5.0,
        "no_helmet": 5.0,
        "machine": 1.0,
        "tool": 1.0,
        "default": 6.0,
    }


def sanitize_geometry(raw, defaults, kind: str):
    warnings = []
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        return defaults, [f"Invalid {kind} JSON. Using defaults."]

    if not isinstance(parsed, list):
        return defaults, [f"{kind.upper()} JSON must be a list. Using defaults."]

    out = []
    for idx, item in enumerate(parsed):
        if not isinstance(item, dict) or "points" not in item:
            continue
        try:
            pts = np.asarray(item["points"], dtype=float)
        except Exception:
            continue
        if pts.ndim != 2 or pts.shape[1] != 2:
            continue
        if kind == "line" and len(pts) < 2:
            continue
        if kind == "zone" and len(pts) < 3:
            continue

        clean = {
            "name": str(item.get("name", f"{kind}_{idx + 1}")),
            "points": pts.tolist(),
        }
        if kind == "line":
            clean["forbidden_sign"] = item.get("forbidden_sign", None)
        else:
            clean["loiter_seconds"] = item.get("loiter_seconds", None)
            clean["trigger_missing_helmet"] = bool(item.get("trigger_missing_helmet", False))
        out.append(clean)

    if not out:
        warnings.append(f"No valid {kind} definitions found. Using defaults.")
        return defaults, warnings
    return out, warnings


def prepare_geometry(lines, zones, width, height, default_loiter):
    def to_pixels(points):
        arr = np.asarray(points, dtype=float)
        if arr.size == 0:
            return arr.astype(int)
        if float(arr.max()) <= 1.5:
            arr[:, 0] *= float(width)
            arr[:, 1] *= float(height)
        return arr.astype(int)

    prepared_lines = []
    for line in lines:
        pts = to_pixels(line["points"])
        if len(pts) < 2:
            continue

        forbidden = line.get("forbidden_sign", None)
        if forbidden is not None:
            try:
                forbidden = int(forbidden)
            except Exception:
                forbidden = None

        prepared_lines.append(
            {
                "name": line["name"],
                "pts": pts,
                "forbidden_sign": forbidden,
            }
        )

    prepared_zones = []
    for zone in zones:
        pts = to_pixels(zone["points"])
        if len(pts) < 3:
            continue

        loiter = zone.get("loiter_seconds", None)
        try:
            loiter = float(default_loiter if loiter is None else loiter)
        except Exception:
            loiter = float(default_loiter)

        prepared_zones.append(
            {
                "name": zone["name"],
                "pts": pts,
                "loiter_seconds": loiter,
                "trigger_missing_helmet": bool(zone.get("trigger_missing_helmet", False)),
            }
        )
    return prepared_lines, prepared_zones


def parse_homography(text):
    if not text or not str(text).strip():
        return None
    try:
        arr = np.asarray(json.loads(text), dtype=float)
        if arr.shape == (3, 3):
            return arr.tolist()
    except Exception:
        return None
    return None


# -----------------------------------------------------------------------------
# Track and tracker implementations
# -----------------------------------------------------------------------------
class Track:
    def __init__(self, track_id, cls_name, bbox, conf, embedding, frame_idx, ts):
        self.id = int(track_id)
        self.cls = str(cls_name)
        self.bbox = tuple(int(v) for v in bbox)
        self.conf = float(conf)
        self.embedding = embedding

        self.centroid = centroid(self.bbox)
        self.history = deque(maxlen=60)
        self.timestamps = deque(maxlen=60)
        self.metric_history = deque(maxlen=60)

        self.history.append(self.centroid.copy())
        self.timestamps.append(float(ts))

        self.velocity = np.zeros(2, dtype=float)
        self.prediction = self.centroid.copy()

        self.last_frame = int(frame_idx)
        self.first_ts = float(ts)
        self.last_ts = float(ts)

        self.hits = 1
        self.misses = 0
        self.age = 0
        self.confirmed = False

        self.speed = 0.0
        self.smoothed_speed = 0.0
        self.max_speed = 0.0
        self.speed_sum = 0.0
        self.speed_count = 0
        self.heading = 0.0
        self.state = "stationary"
        self.speed_limit = 30.0
        self.distance_m = 0.0

        self.zone_enter = {}
        self.dwell = {}
        self.loitered_zones = set()
        self.prev_zone_point = None

        self.true_id = None


def _smooth_bbox(old_bbox, new_bbox, alpha=0.55):
    return tuple(int(alpha * n + (1.0 - alpha) * o) for o, n in zip(old_bbox, new_bbox))


def _update_track_geometry(tr: Track, det: dict, frame_idx: int, ts: float):
    old_centroid = tr.centroid.copy()
    dt_frames = max(1, int(frame_idx) - int(tr.last_frame))

    tr.bbox = _smooth_bbox(tr.bbox, det["bbox"])
    tr.centroid = centroid(tr.bbox)
    tr.conf = float(det.get("conf", tr.conf))

    emb = det.get("embedding", None)
    if emb is not None:
        emb = np.asarray(emb, dtype=float)
        n = float(np.linalg.norm(emb))
        if n > 1e-9:
            emb = emb / n

        if tr.embedding is None:
            tr.embedding = emb
        else:
            tr.embedding = 0.60 * tr.embedding + 0.40 * emb
            n = float(np.linalg.norm(tr.embedding))
            if n > 1e-9:
                tr.embedding = tr.embedding / n

    instant_v = (tr.centroid - old_centroid) / float(dt_frames)
    tr.velocity = 0.55 * tr.velocity + 0.45 * instant_v

    tr.history.append(tr.centroid.copy())
    tr.timestamps.append(float(ts))
    tr.prediction = tr.centroid.copy()

    tr.last_frame = int(frame_idx)
    tr.last_ts = float(ts)
    tr.hits += 1
    tr.misses = 0

    if det.get("true_id", None) is not None:
        tr.true_id = int(det["true_id"])


class SimpleTracker:
    """
    Lightweight fallback tracker with:
      - constant-velocity prediction
      - IoU + center-distance matching
      - optional appearance embedding for BoT-SORT-like behavior
    """

    def __init__(self, frame_shape, tracker_name="ByteTrack", max_age=30, min_hits=2):
        h, w = frame_shape[:2]
        self.diag = math.hypot(float(w), float(h))
        self.tracker_name = tracker_name
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self.use_reid = "bot" in str(tracker_name).lower()
        self.next_id = 1
        self.tracks = {}
        self.finished = []

    def _extract_embedding(self, frame, bbox):
        if frame is None:
            return None
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        crop = cv2.resize(crop, (32, 32), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        parts = []
        for ch in range(3):
            hist = cv2.calcHist([hsv], [ch], None, [16], [0, 256]).flatten()
            n = float(np.linalg.norm(hist))
            if n > 1e-9:
                hist = hist / n
            parts.append(hist)
        emb = np.concatenate(parts)
        n = float(np.linalg.norm(emb))
        if n > 1e-9:
            emb = emb / n
        return emb

    def update(self, frame, dets, frame_idx, ts):
        track_list = list(self.tracks.values())

        for tr in track_list:
            dt_frames = max(1, int(frame_idx) - int(tr.last_frame))
            tr.prediction = tr.centroid + tr.velocity * float(dt_frames)
            tr.age += 1

        for d in dets:
            if d.get("embedding", None) is None:
                d["embedding"] = self._extract_embedding(frame, d["bbox"])

        costs = []
        for i, d in enumerate(dets):
            dc = centroid(d["bbox"])
            for j, tr in enumerate(track_list):
                if str(d.get("cls", "")) != str(tr.cls):
                    continue
                dist = float(np.linalg.norm(dc - tr.prediction))
                geom = dist / (0.22 * self.diag + 1.0)
                iou_value = iou(d["bbox"], tr.bbox)
                app = 0.0
                if self.use_reid:
                    app = 1.0 - cosine_similarity(d.get("embedding"), tr.embedding)
                cost = 0.62 * geom + 0.18 * (1.0 - iou_value) + 0.20 * app
                if cost < 0.75:
                    costs.append((cost, i, j))

        costs.sort(key=lambda x: x[0])
        matched_d = set()
        matched_t = set()
        matches = []
        for _, i, j in costs:
            if i in matched_d or j in matched_t:
                continue
            matches.append((i, j))
            matched_d.add(i)
            matched_t.add(j)

        for i, j in matches:
            tr = track_list[j]
            _update_track_geometry(tr, dets[i], frame_idx, ts)
            tr.confirmed = tr.hits >= self.min_hits

        for i, d in enumerate(dets):
            if i in matched_d:
                continue
            if float(d.get("conf", 0.0)) < 0.18:
                continue
            tr = Track(
                track_id=self.next_id,
                cls_name=d.get("cls", "object"),
                bbox=d["bbox"],
                conf=d.get("conf", 0.5),
                embedding=d.get("embedding", None),
                frame_idx=frame_idx,
                ts=ts,
            )
            tr.true_id = d.get("true_id", None)
            tr.confirmed = tr.hits >= self.min_hits
            self.tracks[self.next_id] = tr
            self.next_id += 1

        for j, tr in enumerate(track_list):
            if j in matched_t:
                continue
            tr.misses += 1
            if tr.misses > self.max_age:
                self.finished.append(tr)
                self.tracks.pop(tr.id, None)

        return list(self.tracks.values()), len(matches)


class NativeTrackManager:
    """
    Uses track IDs supplied by Ultralytics ByteTrack / BoT-SORT.
    """

    def __init__(self, max_age=45, min_hits=2):
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self.tracks = {}
        self.finished = []

    def update(self, dets, frame_idx, ts):
        seen = set()
        for d in dets:
            tid = int(d["track_id"])
            seen.add(tid)
            tr = self.tracks.get(tid)
            if tr is None:
                tr = Track(
                    track_id=tid,
                    cls_name=d.get("cls", "object"),
                    bbox=d["bbox"],
                    conf=d.get("conf", 0.5),
                    embedding=d.get("embedding", None),
                    frame_idx=frame_idx,
                    ts=ts,
                )
                tr.true_id = d.get("true_id", None)
                self.tracks[tid] = tr
            else:
                _update_track_geometry(tr, d, frame_idx, ts)
            tr.confirmed = tr.hits >= self.min_hits

        for tid, tr in list(self.tracks.items()):
            if tid in seen:
                continue
            dt_frames = max(1, int(frame_idx) - int(tr.last_frame))
            tr.prediction = tr.centroid + tr.velocity * float(dt_frames)
            tr.misses += 1
            tr.age += 1
            if tr.misses > self.max_age:
                self.finished.append(tr)
                self.tracks.pop(tid, None)

        return list(self.tracks.values()), len(dets)


# -----------------------------------------------------------------------------
# Motion estimation
# -----------------------------------------------------------------------------
class MotionEstimator:
    def __init__(self, pixels_per_meter, homography, class_limits, default_limit):
        self.pixels_per_meter = float(pixels_per_meter)
        self.H = None
        if homography is not None:
            try:
                H = np.asarray(homography, dtype=float)
                if H.shape == (3, 3):
                    self.H = H
            except Exception:
                self.H = None
        self.class_limits = dict(class_limits or {})
        self.default_limit = float(default_limit)

    def pixel_to_metric(self, pt):
        pt = np.asarray(pt, dtype=float)
        if self.H is not None:
            p = self.H @ np.array([pt[0], pt[1], 1.0], dtype=float)
            if abs(float(p[2])) > 1e-9:
                return np.array([p[0] / p[2], p[1] / p[2]], dtype=float)
        return np.array([pt[0] * self.pixels_per_meter, pt[1] * self.pixels_per_meter], dtype=float)

    def update(self, tr: Track):
        current_metric = self.pixel_to_metric(tr.centroid)

        if len(tr.metric_history) > 0:
            last_metric = tr.metric_history[-1]
            tr.distance_m += float(np.linalg.norm(current_metric - last_metric))
        tr.metric_history.append(current_metric)

        inst_speed = tr.smoothed_speed
        if len(tr.metric_history) >= 2 and len(tr.timestamps) >= 2:
            window = min(5, len(tr.metric_history), len(tr.timestamps))
            p0 = tr.metric_history[-window]
            p1 = tr.metric_history[-1]
            t0 = tr.timestamps[-window]
            t1 = tr.timestamps[-1]
            dt = float(t1 - t0)
            if dt > 1e-3:
                inst_speed = float(np.linalg.norm(p1 - p0) / dt * 3.6)

        tr.smoothed_speed = 0.35 * inst_speed + 0.65 * tr.smoothed_speed
        tr.speed = float(tr.smoothed_speed)
        tr.max_speed = max(tr.max_speed, tr.speed)
        tr.speed_sum += tr.speed
        tr.speed_count += 1

        if len(tr.metric_history) >= 2:
            delta = tr.metric_history[-1] - tr.metric_history[-2]
            if float(np.linalg.norm(delta)) > 1e-4:
                tr.heading = (math.degrees(math.atan2(float(delta[1]), float(delta[0]))) + 360.0) % 360.0

        limit = float(self.class_limits.get(tr.cls, self.default_limit))
        limit = max(1.0, limit)
        tr.speed_limit = limit

        if tr.speed < 0.7:
            tr.state = "stationary"
        elif tr.speed >= limit * 1.25:
            tr.state = "fast"
        else:
            tr.state = "moving"


# -----------------------------------------------------------------------------
# Events
# -----------------------------------------------------------------------------
class EventEngine:
    def __init__(self, snapshot_dir, max_snapshots=30):
        self.snapshot_dir = snapshot_dir
        self.max_snapshots = int(max_snapshots)
        self.events = []
        self.snapshots = []
        self.cooldown = {}
        self.cooldown_s = {
            "speeding": 2.0,
            "wrong_way": 2.0,
            "loitering": 5.0,
            "zone_intrusion": 0.0,
            "missing_helmet": 3.0,
            "tool_misuse": 3.0,
        }

    def add(self, event_type, tr: Track, ts, frame, message="", **extra):
        key = (event_type, tr.id, extra.get("zone", ""), extra.get("line", ""))
        cd = float(self.cooldown_s.get(event_type, 0.0))
        if cd > 0.0:
            last = self.cooldown.get(key, -1e9)
            if float(ts) - float(last) < cd:
                return
        self.cooldown[key] = float(ts)

        snapshot_path = None
        if frame is not None and len(self.snapshots) < self.max_snapshots:
            snapshot_path = os.path.join(
                self.snapshot_dir,
                f"{len(self.events) + 1:03d}_{event_type}_id{tr.id}.jpg",
            )
            snap = frame.copy()
            x1, y1, x2, y2 = map(int, tr.bbox)
            cv2.rectangle(snap, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(
                snap,
                f"{event_type} ID:{tr.id}",
                (x1, max(18, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            ok = cv2.imwrite(snapshot_path, snap)
            if ok and os.path.exists(snapshot_path):
                self.snapshots.append(snapshot_path)
            else:
                snapshot_path = None

        event = {
            "timestamp": float(ts),
            "iso": datetime.now().isoformat(timespec="seconds"),
            "type": str(event_type),
            "track_id": int(tr.id),
            "class": str(tr.cls),
            "confidence": round(float(tr.conf), 3),
            "speed_kmh": round(float(tr.speed), 2),
            "heading_deg": round(float(tr.heading), 1),
            "state": str(tr.state),
            "message": str(message),
            "snapshot": snapshot_path,
        }
        event.update(extra)
        self.events.append(event)


# -----------------------------------------------------------------------------
# Zone and line analytics
# -----------------------------------------------------------------------------
class ZoneAnalytics:
    def __init__(self, lines, zones, event_engine: EventEngine):
        self.lines = lines
        self.zones = zones
        self.events = event_engine
        self.line_counts = {
            line["name"]: {"forward": 0, "backward": 0, "total": 0}
            for line in lines
        }
        self.zone_max_dwell = {zone["name"]: 0.0 for zone in zones}

    def update(self, tracks, ts, frame):
        for tr in tracks:
            curr = tr.centroid.copy() if tr.misses == 0 else tr.prediction.copy()
            curr = np.asarray(curr, dtype=float)

            if tr.prev_zone_point is None:
                tr.prev_zone_point = curr.copy()
                continue

            prev = np.asarray(tr.prev_zone_point, dtype=float)

            if float(np.linalg.norm(curr - prev)) > 0.5:
                for line in self.lines:
                    pts = line["pts"]
                    if len(pts) < 2:
                        continue
                    if segments_intersect(prev, curr, pts[0], pts[1]):
                        sign = line_cross_sign(pts, prev, curr)
                        if sign != 0:
                            direction = "forward" if sign < 0 else "backward"
                            self.line_counts[line["name"]][direction] += 1
                            self.line_counts[line["name"]]["total"] += 1
                            forbidden = line.get("forbidden_sign", None)
                            if forbidden is not None and sign == int(forbidden):
                                self.events.add(
                                    "wrong_way",
                                    tr,
                                    ts,
                                    frame,
                                    message=f"Crossed {line['name']} in forbidden direction",
                                    line=line["name"],
                                    direction=direction,
                                    sign=sign,
                                )

            tr.prev_zone_point = curr.copy()

            for zone in self.zones:
                inside = point_in_polygon(curr, zone["pts"])
                name = zone["name"]
                if inside:
                    enter = tr.zone_enter.get(name, None)
                    if enter is None:
                        tr.zone_enter[name] = float(ts)
                        self.events.add(
                            "zone_intrusion",
                            tr,
                            ts,
                            frame,
                            message=f"Entered {name}",
                            zone=name,
                        )
                    else:
                        dwell = float(ts) - float(enter)
                        tr.dwell[name] = max(float(tr.dwell.get(name, 0.0)), dwell)
                        self.zone_max_dwell[name] = max(
                            float(self.zone_max_dwell.get(name, 0.0)), dwell
                        )
                        if dwell > float(zone["loiter_seconds"]) and name not in tr.loitered_zones:
                            tr.loitered_zones.add(name)
                            self.events.add(
                                "loitering",
                                tr,
                                ts,
                                frame,
                                message=f"Loitering in {name} for {dwell:.1f}s",
                                zone=name,
                                dwell_s=round(dwell, 2),
                            )
                        if zone.get("trigger_missing_helmet", False) and "no_helmet" in str(tr.cls).lower():
                            self.events.add(
                                "missing_helmet",
                                tr,
                                ts,
                                frame,
                                message=f"Person without helmet in {name}",
                                zone=name,
                            )
                else:
                    if tr.zone_enter.get(name, None) is not None:
                        tr.zone_enter[name] = None
                        tr.loitered_zones.discard(name)

            if tr.confirmed and tr.misses == 0 and tr.speed > tr.speed_limit and tr.speed > 1.0:
                self.events.add(
                    "speeding",
                    tr,
                    ts,
                    frame,
                    message=f"Speed {tr.speed:.1f} km/h exceeds limit {tr.speed_limit:.1f} km/h",
                    limit_kmh=round(float(tr.speed_limit), 2),
                )


# -----------------------------------------------------------------------------
# Renderer
# -----------------------------------------------------------------------------
def blur_privacy(frame, tracks):
    out = frame.copy()
    h, w = out.shape[:2]
    for tr in tracks:
        if tr.misses > 0:
            continue
        x1, y1, x2, y2 = map(int, tr.bbox)
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h, y2))
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        cls = str(tr.cls).lower()

        regions = []
        if any(s in cls for s in ["person", "worker", "helmet"]):
            fx1 = max(0, x1 + int(bw * 0.25))
            fx2 = min(w, x1 + int(bw * 0.75))
            fy1 = max(0, y1)
            fy2 = min(h, y1 + int(bh * 0.24))
            if fx2 > fx1 and fy2 > fy1:
                regions.append((fx1, fy1, fx2, fy2))

        if any(s in cls for s in ["car", "bus", "truck", "motorcycle", "vehicle"]):
            px1 = max(0, x1 + int(bw * 0.25))
            px2 = min(w, x1 + int(bw * 0.75))
            py1 = max(0, y2 - int(bh * 0.18))
            py2 = min(h, y2)
            if px2 > px1 and py2 > py1:
                regions.append((px1, py1, px2, py2))

        for rx1, ry1, rx2, ry2 in regions:
            roi = out[ry1:ry2, rx1:rx2]
            if roi.size > 0:
                ks = min(31, roi.shape[0], roi.shape[1])
                if ks % 2 == 0:
                    ks -= 1
                if ks >= 3:
                    out[ry1:ry2, rx1:rx2] = cv2.GaussianBlur(roi, (ks, ks), 0)
    return out


class Renderer:
    def __init__(self, lines, zones, privacy_blur=False, trail_length=45):
        self.lines = lines
        self.zones = zones
        self.privacy_blur = bool(privacy_blur)
        self.trail_length = int(trail_length)

    @staticmethod
    def _zone_color(zone):
        name = str(zone.get("name", "")).lower()
        if zone.get("trigger_missing_helmet", False) or "danger" in name or "restrict" in name:
            return (0, 0, 255)
        return (255, 150, 0)

    def draw(self, frame, tracks, metrics):
        out = frame.copy()
        if self.privacy_blur:
            out = blur_privacy(out, tracks)

        zone_overlay = out.copy()
        for zone in self.zones:
            pts = zone["pts"].astype(np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(zone_overlay, [pts], self._zone_color(zone))
        out = cv2.addWeighted(zone_overlay, 0.18, out, 0.82, 0.0, out)

        for zone in self.zones:
            pts = zone["pts"].astype(np.int32)
            color = self._zone_color(zone)
            cv2.polylines(out, [pts.reshape((-1, 1, 2))], True, color, 2)
            if len(pts) > 0:
                cv2.putText(
                    out,
                    zone["name"],
                    (int(pts[0][0]) + 4, int(pts[0][1]) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv2.LINE_AA,
                )

        for line in self.lines:
            pts = line["pts"].astype(np.int32)
            if len(pts) >= 2:
                cv2.line(out, tuple(pts[0]), tuple(pts[1]), (0, 255, 255), 2)
                cv2.putText(
                    out,
                    line["name"],
                    (int(pts[0][0]) + 4, int(pts[0][1]) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        trail_layer = np.zeros_like(out)
        for tr in tracks:
            if not tr.confirmed:
                continue
            hist = list(tr.history)[-self.trail_length :]
            if len(hist) == 0:
                continue
            col = class_color(tr.cls)
            for i, pt in enumerate(hist):
                alpha = float(i + 1) / float(max(1, len(hist)))
                c = tuple(int(min(255, max(0, x * alpha))) for x in col)
                radius = max(1, int(1.0 + 2.5 * alpha))
                cv2.circle(trail_layer, (int(pt[0]), int(pt[1])), radius, c, -1)
        out = cv2.addWeighted(out, 1.0, trail_layer, 0.75, 0.0, out)

        for tr in tracks:
            if tr.misses > 8:
                continue
            x1, y1, x2, y2 = map(int, tr.bbox)
            color = (160, 160, 160) if tr.misses > 0 else class_color(tr.cls)
            thickness = 1 if tr.misses > 0 else 2
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

            label = (
                f"{tr.cls} ID:{tr.id} {tr.conf:.2f} "
                f"{tr.speed:.1f}km/h {tr.state}"
            )
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            ty = max(th + 4, y1 - 4)
            cv2.rectangle(out, (x1, ty - th - 4), (x1 + tw + 4, ty + baseline), color, -1)
            cv2.putText(
                out,
                label,
                (x1 + 2, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        hud = [
            f"Scenario: {metrics.get('scenario', 'n/a')} | Tracker: {metrics.get('tracker', 'n/a')}",
            f"FPS: {metrics.get('avg_fps', 0.0):.1f} | Frame latency: {metrics.get('latency_ms', 0.0):.1f} ms",
            f"Active tracks: {metrics.get('active_tracks', 0)} | Events: {metrics.get('events', 0)}",
        ]
        line_total = 0
        for counts in metrics.get("line_counts", {}).values():
            line_total += int(counts.get("total", 0))
        hud.append(f"Line crossings: {line_total}")

        hud_w = min(420, max(220, out.shape[1] - 10))
        hud_h = 24 + 20 * len(hud)
        cv2.rectangle(out, (0, 0), (hud_w, hud_h), (0, 0, 0), -1)
        for i, text in enumerate(hud):
            cv2.putText(
                out,
                text,
                (6, 20 + i * 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return out


# -----------------------------------------------------------------------------
# Synthetic realistic demo scene
# -----------------------------------------------------------------------------
class SyntheticScene:
    def __init__(self, scenario, width, height, fps, duration_s, seed):
        self.scenario = scenario
        self.w = int(width)
        self.h = int(height)
        self.fps = int(fps)
        self.frames = max(15, int(float(duration_s) * float(fps)))
        self.rng = np.random.default_rng(int(seed))
        self.t = 0

        # IMPORTANT: create the list before _spawn() appends to it.
        self.objects = []
        self._spawn()

    def _add_obj(self, cls, x, y, ow, oh, vx, vy):
        if not hasattr(self, "objects"):
            self.objects = []

        self.objects.append(
            {
                "id": len(self.objects) + 1,
                "cls": cls,
                "x": float(x),
                "y": float(y),
                "w": int(max(8, ow)),
                "h": int(max(8, oh)),
                "vx": float(vx),
                "vy": float(vy),
                "occlude_until": 0,
            }
        )

    def _spawn(self):
        w, h = self.w, self.h
        rng = self.rng

        if self.scenario == "traffic":
            lanes = [(0.20, 1), (0.38, 1), (0.62, -1), (0.80, -1)]
            for lane_y, direction in lanes:
                for _ in range(3):
                    cls = str(rng.choice(["car", "car", "car", "bus", "truck", "motorcycle"]))
                    if cls in ("bus", "truck"):
                        ow = int(w * 0.11)
                        oh = int(h * 0.13)
                        base_speed = rng.normal(6.0, 0.8)
                    elif cls == "car":
                        ow = int(w * 0.075)
                        oh = int(h * 0.09)
                        base_speed = rng.normal(9.5, 1.2)
                    else:
                        ow = int(w * 0.045)
                        oh = int(h * 0.06)
                        base_speed = rng.normal(12.0, 1.5)
                    x = rng.uniform(0, w)
                    y = lane_y * h + rng.normal(0, 4)
                    vx = direction * abs(float(base_speed))
                    vy = float(rng.normal(0, 0.06))
                    self._add_obj(cls, x, y, ow, oh, vx, vy)

            # wrong-way vehicles
            self._add_obj("car", 0.20 * w, 0.20 * h, int(w * 0.075), int(h * 0.09), -9.0, 0.0)
            self._add_obj("motorcycle", 0.80 * w, 0.80 * h, int(w * 0.045), int(h * 0.06), 13.0, 0.0)

            # slow/stopped vehicle in restricted shoulder
            self._add_obj("car", 0.12 * w, 0.84 * h, int(w * 0.075), int(h * 0.09), 0.15, 0.02)

        elif self.scenario == "warehouse":
            for _ in range(6):
                cls = "person"
                ow = int(rng.uniform(24, 34))
                oh = int(rng.uniform(55, 75))
                x = rng.uniform(0.10, 0.90) * w
                y = rng.uniform(0.15, 0.85) * h
                vx = rng.normal(0, 0.9)
                vy = rng.normal(0, 0.9)
                self._add_obj(cls, x, y, ow, oh, vx, vy)

            for _ in range(2):
                cls = "forklift"
                ow = int(w * 0.09)
                oh = int(h * 0.12)
                x = rng.uniform(0.2, 0.8) * w
                y = rng.choice([0.30, 0.65]) * h
                vx = rng.choice([-1, 1]) * rng.normal(4.0, 0.7)
                vy = rng.normal(0, 0.15)
                self._add_obj(cls, x, y, ow, oh, vx, vy)

            for _ in range(4):
                cls = "pallet"
                ow = int(rng.uniform(38, 58))
                oh = int(rng.uniform(28, 42))
                x = rng.uniform(0.58, 0.88) * w
                y = rng.uniform(0.25, 0.75) * h
                self._add_obj(cls, x, y, ow, oh, 0.0, 0.0)

            # loitering person in restricted rack
            self._add_obj("person", 0.72 * w, 0.48 * h, 30, 68, 0.05, 0.03)

        elif self.scenario == "agriculture":
            for _ in range(5):
                cls = str(rng.choice(["cow", "sheep", "sheep", "horse"]))
                ow = int(w * (0.08 if cls == "horse" else 0.065))
                oh = int(h * (0.10 if cls == "horse" else 0.08))
                x = rng.uniform(0.10, 0.90) * w
                y = rng.uniform(0.20, 0.85) * h
                vx = rng.normal(0, 0.55)
                vy = rng.normal(0, 0.45)
                self._add_obj(cls, x, y, ow, oh, vx, vy)

            self._add_obj("tractor", 0.25 * w, 0.35 * h, int(w * 0.13), int(h * 0.14), 5.5, 0.15)
            self._add_obj("person", 0.55 * w, 0.70 * h, 30, 70, 0.8, -0.4)
            self._add_obj("cow", 0.22 * w, 0.72 * h, int(w * 0.065), int(h * 0.08), 0.08, 0.04)

        else:  # safety
            self._add_obj("machine", 0.66 * w, 0.50 * h, int(w * 0.24), int(h * 0.34), 0.0, 0.0)
            self._add_obj("tool", 0.54 * w, 0.76 * h, 34, 22, 0.25, 0.08)

            for i in range(6):
                cls = "no_helmet" if i < 2 else "helmet"
                ow = int(rng.uniform(26, 34))
                oh = int(rng.uniform(58, 76))
                x = rng.uniform(0.12, 0.88) * w
                y = rng.uniform(0.18, 0.82) * h
                vx = rng.normal(0, 0.85)
                vy = rng.normal(0, 0.75)
                self._add_obj(cls, x, y, ow, oh, vx, vy)

            # person without helmet entering danger zone
            self._add_obj("no_helmet", 0.58 * w, 0.42 * h, 30, 70, 0.35, 0.12)

        return self.objects

    def _bbox(self, obj):
        x1 = int(obj["x"] - obj["w"] / 2.0)
        y1 = int(obj["y"] - obj["h"] / 2.0)
        x2 = int(obj["x"] + obj["w"] / 2.0)
        y2 = int(obj["y"] + obj["h"] / 2.0)
        return x1, y1, x2, y2

    def _background(self):
        rng = self.rng
        if self.scenario == "traffic":
            frame = np.full((self.h, self.w, 3), 65, dtype=np.uint8)
            cv2.rectangle(frame, (0, int(0.10 * self.h)), (self.w, int(0.90 * self.h)), (55, 55, 55), -1)
            for lane_y in [0.29, 0.50, 0.71]:
                y = int(lane_y * self.h)
                for x in range(0, self.w, 80):
                    cv2.rectangle(frame, (x, y - 2), (x + 38, y + 2), (220, 220, 220), -1)
            cv2.line(frame, (0, int(0.10 * self.h)), (self.w, int(0.10 * self.h)), (0, 255, 255), 3)
            cv2.line(frame, (0, int(0.90 * self.h)), (self.w, int(0.90 * self.h)), (0, 255, 255), 3)
        elif self.scenario == "warehouse":
            frame = np.full((self.h, self.w, 3), 175, dtype=np.uint8)
            for x in range(0, self.w, 90):
                cv2.line(frame, (x, 0), (x, self.h), (150, 150, 150), 1)
            for y in range(0, self.h, 90):
                cv2.line(frame, (0, y), (self.w, y), (150, 150, 150), 1)
            cv2.rectangle(
                frame,
                (int(0.55 * self.w), int(0.18 * self.h)),
                (int(0.94 * self.w), int(0.82 * self.h)),
                (90, 90, 90),
                8,
            )
        elif self.scenario == "agriculture":
            frame = np.full((self.h, self.w, 3), (55, 110, 55), dtype=np.uint8)
            for y in range(0, self.h, 55):
                cv2.line(frame, (0, y), (self.w, y), (45, 95, 45), 8)
            cv2.rectangle(frame, (0, int(0.88 * self.h)), (self.w, self.h), (80, 60, 40), -1)
            for x in range(20, self.w, 80):
                cv2.rectangle(frame, (x, int(0.05 * self.h)), (x + 8, int(0.15 * self.h)), (120, 90, 60), -1)
        else:
            frame = np.full((self.h, self.w, 3), 55, dtype=np.uint8)
            for x in range(0, self.w, 100):
                cv2.line(frame, (x, 0), (x, self.h), (70, 70, 70), 1)
            for y in range(0, self.h, 100):
                cv2.line(frame, (0, y), (self.w, y), (70, 70, 70), 1)
            cv2.rectangle(
                frame,
                (int(0.45 * self.w), int(0.18 * self.h)),
                (int(0.90 * self.w), int(0.82 * self.h)),
                (40, 40, 40),
                -1,
            )
            for x in range(int(0.45 * self.w), int(0.90 * self.w), 35):
                cv2.line(frame, (x, int(0.18 * self.h)), (x + 20, int(0.82 * self.h)), (0, 0, 255), 2)
        return frame

    def _draw_object(self, frame, obj):
        x1, y1, x2, y2 = self._bbox(obj)
        cls = obj["cls"]
        color = class_color(cls)

        # shadow
        cx = int((x1 + x2) / 2)
        cy = int(y2 - 2)
        axes = (max(5, int((x2 - x1) * 0.45)), max(2, int((y2 - y1) * 0.10)))
        cv2.ellipse(frame, (cx, cy), axes, 0, 0, 360, (30, 30, 30), -1)

        if cls in VEHICLE_CLASSES:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (20, 20, 20), 2)
            wx1 = int(x1 + (x2 - x1) * 0.18)
            wx2 = int(x1 + (x2 - x1) * 0.82)
            wy1 = int(y1 + (y2 - y1) * 0.12)
            wy2 = int(y1 + (y2 - y1) * 0.42)
            if wx2 > wx1 and wy2 > wy1:
                cv2.rectangle(frame, (wx1, wy1), (wx2, wy2), (220, 220, 220), -1)
            wheel_y = int(y2 - 3)
            for wx in [int(x1 + (x2 - x1) * 0.22), int(x1 + (x2 - x1) * 0.78)]:
                cv2.circle(frame, (wx, wheel_y), max(3, int((x2 - x1) * 0.08)), (10, 10, 10), -1)
            if cls == "forklift":
                cv2.rectangle(frame, (x2 - 8, y1), (x2 + 6, y2), (70, 70, 70), -1)
            if cls == "tractor":
                cv2.circle(
                    frame,
                    (int(x1 + (x2 - x1) * 0.25), y2 - 4),
                    max(6, int((y2 - y1) * 0.18)),
                    (20, 20, 20),
                    -1,
                )

        elif cls in PERSON_CLASSES:
            body_x1 = int(x1 + (x2 - x1) * 0.25)
            body_x2 = int(x1 + (x2 - x1) * 0.75)
            body_y1 = int(y1 + (y2 - y1) * 0.28)
            cv2.rectangle(frame, (body_x1, body_y1), (body_x2, y2), color, -1)
            head_c = (int((x1 + x2) / 2), int(y1 + (y2 - y1) * 0.16))
            head_r = max(4, int((x2 - x1) * 0.18))
            cv2.circle(frame, head_c, head_r, (180, 180, 180), -1)
            if cls == "helmet":
                cv2.circle(frame, head_c, head_r + 2, (0, 215, 255), 3)
            elif cls == "no_helmet":
                cv2.circle(frame, head_c, head_r + 2, (0, 0, 255), 2)

        elif cls in ANIMAL_CLASSES:
            cv2.ellipse(
                frame,
                (int((x1 + x2) / 2), int((y1 + y2) / 2 + 4)),
                (int((x2 - x1) * 0.42), int((y2 - y1) * 0.28)),
                0,
                0,
                360,
                color,
                -1,
            )
            head_c = (int(x1 + (x2 - x1) * 0.18), int(y1 + (y2 - y1) * 0.32))
            cv2.circle(frame, head_c, max(4, int((x2 - x1) * 0.13)), color, -1)
            for fx in [0.28, 0.45, 0.62, 0.78]:
                px = int(x1 + (x2 - x1) * fx)
                cv2.line(frame, (px, int(y1 + (y2 - y1) * 0.62)), (px, y2), (60, 60, 60), 2)

        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (20, 20, 20), 2)
            if cls == "machine":
                for xx in range(x1 + 8, x2, 18):
                    cv2.line(frame, (xx, y1 + 6), (xx, y2 - 6), (40, 40, 40), 2)
            elif cls == "tool":
                cv2.circle(
                    frame,
                    (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                    max(3, int((x2 - x1) * 0.18)),
                    (255, 255, 255),
                    -1,
                )

    def step(self):
        frame = self._background()
        dets = []
        gt_count = 0
        rng = self.rng

        for obj in self.objects:
            obj["x"] += obj["vx"]
            obj["y"] += obj["vy"]

            # horizontal wrap / respawn
            if obj["x"] < -obj["w"]:
                obj["x"] = self.w + obj["w"] / 2.0
                if self.scenario == "traffic":
                    obj["y"] = rng.uniform(0.15, 0.85) * self.h
            elif obj["x"] > self.w + obj["w"]:
                obj["x"] = -obj["w"] / 2.0
                if self.scenario == "traffic":
                    obj["y"] = rng.uniform(0.15, 0.85) * self.h

            # vertical bounce
            margin_y = 0.08 * self.h
            if obj["y"] < margin_y:
                obj["y"] = margin_y
                obj["vy"] = abs(obj["vy"])
            elif obj["y"] > self.h - margin_y:
                obj["y"] = self.h - margin_y
                obj["vy"] = -abs(obj["vy"])

            # random occlusion
            if self.t >= int(obj.get("occlude_until", 0)) and rng.random() < 0.004:
                obj["occlude_until"] = self.t + int(rng.integers(6, 24))

            visible = self.t >= int(obj.get("occlude_until", 0))
            if visible:
                gt_count += 1
                self._draw_object(frame, obj)

                if rng.random() > 0.07:
                    x1, y1, x2, y2 = self._bbox(obj)
                    jx = int(rng.normal(0, 2.0))
                    jy = int(rng.normal(0, 2.0))
                    bbox = (
                        max(0, x1 + jx),
                        max(0, y1 + jy),
                        min(self.w, x2 + jx),
                        min(self.h, y2 + jy),
                    )
                    conf = float(np.clip(rng.normal(0.84, 0.08), 0.25, 0.99))
                    dets.append(
                        {
                            "bbox": bbox,
                            "conf": conf,
                            "cls": obj["cls"],
                            "true_id": int(obj["id"]),
                        }
                    )

        # occasional false positive
        if rng.random() < 0.035:
            ow = int(rng.integers(25, 70))
            oh = int(rng.integers(25, 70))
            x = int(rng.integers(0, max(1, self.w - ow)))
            y = int(rng.integers(0, max(1, self.h - oh)))
            cls = str(rng.choice(list(VEHICLE_CLASSES | PERSON_CLASSES | ANIMAL_CLASSES)))
            dets.append(
                {
                    "bbox": (x, y, x + ow, y + oh),
                    "conf": float(rng.uniform(0.25, 0.45)),
                    "cls": cls,
                    "true_id": None,
                }
            )

        self.t += 1
        return frame, dets, gt_count


# -----------------------------------------------------------------------------
# Video output with browser-friendly fallbacks
# -----------------------------------------------------------------------------
class VideoOutput:
    def __init__(self, base_path, fps, size):
        self.base_path = base_path
        self.fps = float(fps)
        self.size = size
        self.active = []
        self.fallback = []
        candidates = [
            (0, ".webm", "VP80", "video/webm"),
            (1, ".mp4", "avc1", "video/mp4"),
            (2, ".mp4", "mp4v", "video/mp4"),
            (3, ".avi", "MJPG", "video/x-msvideo"),
            (4, ".avi", "XVID", "video/x-msvideo"),
        ]
        for priority, ext, cc, mime in candidates:
            path = self.base_path + ext
            fourcc = cv2.VideoWriter_fourcc(*cc)
            writer = cv2.VideoWriter(path, fourcc, self.fps, self.size)
            if writer.isOpened():
                self.active.append((priority, writer, path, mime))
        self.active.sort(key=lambda x: x[0])

    def write(self, frame):
        if self.active:
            for _, writer, _, _ in self.active:
                writer.write(frame)
        else:
            self.fallback.append(frame.copy())
            if len(self.fallback) > 60:
                self.fallback.pop(0)

    def finalize(self):
        for _, writer, _, _ in self.active:
            writer.release()
        for _, _, path, mime in self.active:
            if os.path.exists(path) and os.path.getsize(path) > 5000:
                return path, mime, []
        return None, None, self.fallback


# -----------------------------------------------------------------------------
# Real source support: threaded reader with bounded queue and frame dropping
# -----------------------------------------------------------------------------
class ThreadedFrameSource:
    def __init__(self, source_type, source_value, fps):
        self.source_type = source_type
        self.source_value = source_value
        self.fps = max(1.0, float(fps))
        self.q = queue.Queue(maxsize=3)
        self.stop_event = threading.Event()
        self.cap = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _open_cap(self):
        if self.source_type == "webcam":
            cap = cv2.VideoCapture(0)
        elif self.source_type == "rtsp":
            cap = cv2.VideoCapture(str(self.source_value), cv2.CAP_FFMPEG)
        elif self.source_type == "video":
            cap = cv2.VideoCapture(str(self.source_value))
        else:
            raise ValueError(f"Unsupported source type: {self.source_type}")
        if not cap.isOpened():
            raise RuntimeError("Could not open video source.")
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        return cap

    def _put_frame(self, frame):
        if frame is None:
            return
        if self.q.full():
            try:
                self.q.get_nowait()
            except Exception:
                pass
        try:
            self.q.put_nowait(frame)
        except Exception:
            pass

    def _run(self):
        try:
            if self.source_type == "images":
                paths = list(self.source_value or [])
                if not paths:
                    return
                idx = 0
                while not self.stop_event.is_set():
                    path = paths[idx % len(paths)]
                    frame = cv2.imread(path)
                    self._put_frame(frame)
                    idx += 1
                    time.sleep(1.0 / self.fps)
                return

            while not self.stop_event.is_set():
                try:
                    self.cap = self._open_cap()
                except Exception:
                    time.sleep(1.0)
                    continue

                while not self.stop_event.is_set():
                    ok, frame = self.cap.read()
                    if not ok or frame is None:
                        if self.source_type == "video":
                            self.stop_event.set()
                            break
                        # reconnect for webcam / rtsp
                        try:
                            self.cap.release()
                        except Exception:
                            pass
                        time.sleep(0.5)
                        break
                    self._put_frame(frame)

                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None
                if self.source_type != "video":
                    time.sleep(0.5)
        except Exception:
            pass

    def read(self):
        try:
            return self.q.get(timeout=3.0)
        except Exception:
            return None

    def release(self):
        self.stop_event.set()
        try:
            self.thread.join(timeout=1.0)
        except Exception:
            pass
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass


# -----------------------------------------------------------------------------
# Ultralytics YOLO detector / native tracker wrapper
# -----------------------------------------------------------------------------
class UltralyticsDetector:
    def __init__(self, params):
        if not HAVE_ULTRALYTICS:
            raise RuntimeError("Ultralytics is not available.")
        self.params = params
        self.weights = params.get("weights", "yolo11n.pt")
        self.model = YOLO(self.weights)
        self.device = "cuda:0" if HAVE_TORCH and torch.cuda.is_available() else "cpu"
        self.conf = float(params.get("conf", 0.25))
        self.iou = float(params.get("iou", 0.45))
        self.classes = params.get("classes", []) or None
        self.tile = bool(params.get("tile", False))
        self.native = bool(params.get("use_ultralytics_tracker", True)) and not self.tile
        tracker_name = str(params.get("tracker", "ByteTrack")).lower()
        self.tracker_yaml = "bytetrack.yaml" if "byte" in tracker_name else "botsort.yaml"

    def _parse_boxes(self, result, with_id=False):
        dets = []
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return dets
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        ids = None
        if with_id and getattr(boxes, "id", None) is not None:
            ids = boxes.id.cpu().numpy().astype(int)

        names = getattr(self.model, "names", {})
        for i in range(len(xyxy)):
            cls_id = int(clss[i])
            if isinstance(names, dict):
                cls_name = names.get(cls_id, str(cls_id))
            else:
                try:
                    cls_name = str(names[cls_id])
                except Exception:
                    cls_name = str(cls_id)

            det = {
                "bbox": tuple(int(v) for v in xyxy[i]),
                "conf": float(confs[i]),
                "cls": str(cls_name),
                "embedding": None,
                "true_id": None,
            }
            if ids is not None:
                det["track_id"] = int(ids[i])
            dets.append(det)
        return dets

    def _nms_dets(self, dets):
        if not dets:
            return []
        boxes = []
        scores = []
        kept = []
        for d in dets:
            x1, y1, x2, y2 = d["bbox"]
            w = float(x2 - x1)
            h = float(y2 - y1)
            if w <= 1 or h <= 1:
                continue
            boxes.append([float(x1), float(y1), w, h])
            scores.append(float(d["conf"]))
            kept.append(d)
        if not boxes:
            return []
        idxs = cv2.dnn.NMSBoxes(boxes, scores, 0.10, self.iou)
        if idxs is None or len(idxs) == 0:
            return []
        idxs = np.asarray(idxs).flatten().tolist()
        return [kept[i] for i in idxs]

    def _tiled_predict(self, frame):
        h, w = frame.shape[:2]
        tile = 640
        overlap = 0.25
        step = max(64, int(tile * (1.0 - overlap)))
        all_dets = []
        for y in range(0, h, step):
            for x in range(0, w, step):
                y2 = min(h, y + tile)
                x2 = min(w, x + tile)
                crop = frame[y:y2, x:x2]
                if crop.size == 0 or crop.shape[0] < 32 or crop.shape[1] < 32:
                    continue
                results = self.model.predict(
                    crop,
                    conf=self.conf,
                    iou=self.iou,
                    classes=self.classes,
                    device=self.device,
                    verbose=False,
                )
                dets = self._parse_boxes(results[0], with_id=False)
                for d in dets:
                    x1, y1, x2b, y2b = d["bbox"]
                    d["bbox"] = (x1 + x, y1 + y, x2b + x, y2b + y)
                    all_dets.append(d)
        return self._nms_dets(all_dets)

    def process(self, frame, frame_idx, ts):
        if self.native:
            results = self.model.track(
                frame,
                persist=True,
                tracker=self.tracker_yaml,
                classes=self.classes,
                conf=self.conf,
                iou=self.iou,
                device=self.device,
                verbose=False,
            )
            return self._parse_boxes(results[0], with_id=True)

        if self.tile:
            return self._tiled_predict(frame)

        results = self.model.predict(
            frame,
            conf=self.conf,
            iou=self.iou,
            classes=self.classes,
            device=self.device,
            verbose=False,
        )
        return self._parse_boxes(results[0], with_id=False)


# -----------------------------------------------------------------------------
# Approximate tracking evaluator for synthetic ground truth
# -----------------------------------------------------------------------------
class TrackingEvaluator:
    def __init__(self):
        self.total_gt = 0
        self.total_det = 0
        self.matched = 0
        self.true_seq = defaultdict(list)
        self.pair_counts = defaultdict(int)

    def update(self, dets, tracks, match_count, gt_count):
        self.total_gt += int(gt_count or 0)
        self.total_det += int(len(dets))
        self.matched += int(match_count or 0)

        for tr in tracks:
            if tr.misses == 0 and tr.confirmed and tr.true_id is not None:
                self.true_seq[int(tr.true_id)].append(int(tr.id))
                self.pair_counts[(int(tr.true_id), int(tr.id))] += 1

    def summary(self):
        switches = 0
        fragmentation = 0
        for seq in self.true_seq.values():
            comp = []
            for x in seq:
                if not comp or comp[-1] != x:
                    comp.append(x)
            switches += max(0, len(comp) - 1)
            fragmentation += len(comp)

        fp = max(0, self.total_det - self.matched)
        misses = max(0, self.total_gt - self.matched)
        mota = 0.0
        if self.total_gt > 0:
            mota = 1.0 - float(misses + fp + switches) / float(self.total_gt)

        # Approximate IDF1 using greedy one-to-one association.
        total_true_frames = sum(self.pair_counts.values())
        items = sorted(self.pair_counts.items(), key=lambda kv: kv[1], reverse=True)
        used_true = set()
        used_track = set()
        idtp = 0
        for (tid, track_id), count in items:
            if tid in used_true or track_id in used_track:
                continue
            used_true.add(tid)
            used_track.add(track_id)
            idtp += count

        idfn = max(0, self.total_gt - idtp)
        idfp = max(0, total_true_frames - idtp)
        denom = 2 * idtp + idfp + idfn
        idf1 = float(2 * idtp / denom) if denom > 0 else 0.0

        return {
            "mota_approx": round(float(mota), 4),
            "idf1_approx": round(float(idf1), 4),
            "id_switches": int(switches),
            "fragmentation": int(fragmentation),
            "misses": int(misses),
            "false_positives": int(fp),
            "total_ground_truth_detections": int(self.total_gt),
            "note": "Approximate metrics from synthetic ground truth. Use TrackEval/py-motmetrics on MOTChallenge-format logs for official MOTA/IDF1.",
        }


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------
def run_pipeline(params):
    warnings = []
    scenario = params.get("scenario", "traffic")
    width = int(params.get("width", 960))
    height = int(params.get("height", 540))
    fps = int(params.get("fps", 15))
    duration = float(params.get("duration", 8.0))
    seed = int(params.get("seed", 42))
    tracker_name = params.get("tracker", "ByteTrack")
    privacy = bool(params.get("privacy_blur", False))
    trail_length = int(params.get("trail_length", 45))
    pixels_per_meter = float(params.get("pixels_per_meter", 0.03))
    homography = params.get("homography", None)
    class_limits = dict(params.get("class_limits", default_class_limits(scenario)))
    default_limit = float(class_limits.get("default", 30.0))
    loiter_seconds = float(params.get("loiter_seconds", 6.0))

    out_dir = tempfile.mkdtemp(prefix="cv_tracker_demo_")
    snapshot_dir = os.path.join(out_dir, "snapshots")
    os.makedirs(snapshot_dir, exist_ok=True)

    lines, zones = prepare_geometry(
        params.get("lines", []),
        params.get("zones", []),
        width,
        height,
        loiter_seconds,
    )

    motion = MotionEstimator(pixels_per_meter, homography, class_limits, default_limit)
    event_engine = EventEngine(snapshot_dir, max_snapshots=30)
    analytics = ZoneAnalytics(lines, zones, event_engine)
    renderer = Renderer(lines, zones, privacy_blur=privacy, trail_length=trail_length)

    max_age = 45 if "bot" in str(tracker_name).lower() else 30
    simple_tracker = SimpleTracker((height, width), tracker_name=tracker_name, max_age=max_age, min_hits=2)
    native_tracker = NativeTrackManager(max_age=max_age, min_hits=2)
    evaluator = TrackingEvaluator()

    mode = params.get("mode", "simulation")
    use_real = mode != "simulation"
    scene = None
    source = None
    detector = None
    used_native = False
    real_deadline = float(duration)

    if use_real:
        try:
            detector = UltralyticsDetector(params)
            source = ThreadedFrameSource(params.get("source_type", "video"), params.get("source_value", None), fps)
        except Exception as exc:
            warnings.append(f"Real-time source/detector unavailable: {exc}. Falling back to synthetic simulation.")
            use_real = False

    if not use_real:
        scene = SyntheticScene(scenario, width, height, fps, duration, seed)
        total_frames = scene.frames
    else:
        total_frames = None

    video = VideoOutput(os.path.join(out_dir, "annotated"), fps, (width, height))

    frame_idx = 0
    start = time.perf_counter()
    stage_times = defaultdict(float)
    last_frame = None
    peak_active = 0

    try:
        while True:
            loop_start = time.perf_counter()

            if not use_real:
                if frame_idx >= total_frames:
                    break
                t0 = time.perf_counter()
                frame, dets, gt_count = scene.step()
                t1 = time.perf_counter()
                stage_times["detect"] += t1 - t0
                ts = frame_idx / float(fps)
            else:
                t0 = time.perf_counter()
                frame = source.read()
                t1 = time.perf_counter()
                stage_times["source"] += t1 - t0
                if frame is None:
                    break
                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height))
                ts = time.time() - start
                if ts >= real_deadline:
                    break
                t2 = time.perf_counter()
                dets = detector.process(frame, frame_idx, ts)
                t3 = time.perf_counter()
                stage_times["detect"] += t3 - t2
                gt_count = len([d for d in dets if d.get("true_id", None) is not None])

            t4 = time.perf_counter()
            if (
                use_real
                and detector is not None
                and detector.native
                and (len(dets) == 0 or all(d.get("track_id", None) is not None for d in dets))
            ):
                tracks, match_count = native_tracker.update(dets, frame_idx, ts)
                used_native = True
            else:
                tracks, match_count = simple_tracker.update(frame, dets, frame_idx, ts)
            t5 = time.perf_counter()
            stage_times["track"] += t5 - t4

            t5 = time.perf_counter()
            for tr in tracks:
                if tr.misses == 0:
                    motion.update(tr)
            t6 = time.perf_counter()
            stage_times["motion"] += t6 - t5

            t6 = time.perf_counter()
            analytics.update(tracks, ts, frame)
            t7 = time.perf_counter()
            stage_times["analytics"] += t7 - t6

            evaluator.update(dets, tracks, match_count, gt_count)

            t7 = time.perf_counter()
            active_count = sum(1 for tr in tracks if tr.confirmed and tr.misses == 0)
            peak_active = max(peak_active, active_count)
            metrics = {
                "scenario": scenario,
                "tracker": tracker_name,
                "avg_fps": float((frame_idx + 1) / max(1e-6, time.perf_counter() - start)),
                "latency_ms": float((time.perf_counter() - loop_start) * 1000.0),
                "active_tracks": int(active_count),
                "events": int(len(event_engine.events)),
                "line_counts": analytics.line_counts,
            }
            annotated = renderer.draw(frame, tracks, metrics)
            t8 = time.perf_counter()
            stage_times["render"] += t8 - t7

            video.write(annotated)
            last_frame = annotated
            frame_idx += 1
    finally:
        if source is not None:
            source.release()

    if frame_idx == 0 and use_real:
        fallback_params = dict(params)
        fallback_params["mode"] = "simulation"
        fallback_params["source_type"] = "simulation"
        fallback_params["source_value"] = None
        fallback_result = run_pipeline(fallback_params)
        fallback_result["warnings"] = warnings + fallback_result.get("warnings", [])
        return fallback_result

    video_path, video_mime, fallback_frames = video.finalize()

    all_tracks = []
    if used_native:
        all_tracks = list(native_tracker.tracks.values()) + list(native_tracker.finished)
        unique_tracks = len(set([t.id for t in all_tracks]))
    else:
        all_tracks = list(simple_tracker.tracks.values()) + list(simple_tracker.finished)
        unique_tracks = len(set([t.id for t in all_tracks]))

    events_csv = os.path.join(out_dir, "events.csv")
    event_fields = [
        "timestamp",
        "iso",
        "type",
        "track_id",
        "class",
        "confidence",
        "speed_kmh",
        "heading_deg",
        "state",
        "zone",
        "line",
        "direction",
        "limit_kmh",
        "dwell_s",
        "message",
        "snapshot",
    ]
    with open(events_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=event_fields, extrasaction="ignore")
        writer.writeheader()
        for ev in event_engine.events:
            row = {k: ev.get(k, "") for k in event_fields}
            writer.writerow(row)

    tracks_csv = os.path.join(out_dir, "tracks.csv")
    track_fields = [
        "track_id",
        "class",
        "first_seen_s",
        "last_seen_s",
        "hits",
        "confirmed",
        "avg_speed_kmh",
        "max_speed_kmh",
        "distance_m",
        "dwell_zones_json",
    ]
    with open(tracks_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=track_fields, extrasaction="ignore")
        writer.writeheader()
        for tr in all_tracks:
            avg_speed = float(tr.speed_sum / tr.speed_count) if tr.speed_count > 0 else 0.0
            writer.writerow(
                {
                    "track_id": tr.id,
                    "class": tr.cls,
                    "first_seen_s": round(float(tr.first_ts), 3),
                    "last_seen_s": round(float(tr.last_ts), 3),
                    "hits": int(tr.hits),
                    "confirmed": bool(tr.confirmed),
                    "avg_speed_kmh": round(avg_speed, 3),
                    "max_speed_kmh": round(float(tr.max_speed), 3),
                    "distance_m": round(float(tr.distance_m), 3),
                    "dwell_zones_json": json.dumps(to_py(tr.dwell)),
                }
            )

    elapsed = max(1e-6, time.perf_counter() - start)
    stage_ms = {k: float(v / max(1, frame_idx) * 1000.0) for k, v in stage_times.items()}
    event_counts = defaultdict(int)
    for ev in event_engine.events:
        event_counts[str(ev.get("type", "unknown"))] += 1

    summary_json_path = os.path.join(out_dir, "summary.json")

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "configuration": to_py({k: v for k, v in params.items() if k != "source_value"}),
        "performance": {
            "frames_processed": int(frame_idx),
            "duration_s": float(frame_idx / max(1, fps)),
            "average_fps": float(frame_idx / elapsed),
            "wall_clock_s": float(elapsed),
            "stage_latency_ms": stage_ms,
            "total_pipeline_latency_ms": float(sum(stage_ms.values())),
            "peak_active_tracks": int(peak_active),
        },
        "tracking": {
            "unique_tracks": int(unique_tracks),
            "confirmed_tracks": int(sum(1 for t in all_tracks if t.confirmed)),
            "quality": evaluator.summary(),
        },
        "analytics": {
            "line_counts": to_py(analytics.line_counts),
            "zone_max_dwell_s": to_py(analytics.zone_max_dwell),
            "event_counts": to_py(dict(event_counts)),
            "total_events": int(len(event_engine.events)),
        },
        "outputs": {
            "video_path": video_path,
            "events_csv": events_csv,
            "tracks_csv": tracks_csv,
            "snapshot_dir": snapshot_dir,
            "summary_json": summary_json_path,
        },
    }

    summary = to_py(summary)
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    events_out = to_py(event_engine.events)

    last_frame_bytes = None
    if last_frame is not None:
        last_frame_rgb = cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB)
        ok, buf = cv2.imencode(".png", last_frame_rgb)
        if ok:
            last_frame_bytes = buf.tobytes()

    fallback_bytes = []
    for fb in fallback_frames:
        fb_rgb = cv2.cvtColor(fb, cv2.COLOR_BGR2RGB)
        ok, buf = cv2.imencode(".png", fb_rgb)
        if ok:
            fallback_bytes.append(buf.tobytes())

    snapshots = [p for p in event_engine.snapshots if os.path.exists(p)]

    return {
        "video_path": video_path,
        "video_mime": video_mime,
        "last_frame": last_frame_bytes,
        "fallback_frames": fallback_bytes,
        "events": events_out,
        "events_csv": events_csv,
        "tracks_csv": tracks_csv,
        "summary": summary,
        "summary_json": summary_json_path,
        "snapshots": snapshots,
        "warnings": warnings,
    }


# -----------------------------------------------------------------------------
# Self tests
# -----------------------------------------------------------------------------
def run_self_tests():
    results = []

    def check(name, fn):
        try:
            fn()
            results.append({"test": name, "status": "PASS", "detail": ""})
        except AssertionError as exc:
            results.append({"test": name, "status": "FAIL", "detail": str(exc)})
        except Exception as exc:
            results.append({"test": name, "status": "ERROR", "detail": repr(exc)})

    def test_iou():
        assert abs(iou([0, 0, 10, 10], [0, 0, 10, 10]) - 1.0) < 1e-6
        assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0

    def test_segments():
        assert segments_intersect([0, 0], [10, 10], [0, 10], [10, 0])
        assert not segments_intersect([0, 0], [1, 1], [5, 5], [6, 6])

    def test_point_poly():
        poly = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=int)
        assert point_in_polygon([5, 5], poly)
        assert not point_in_polygon([15, 5], poly)

    def test_tracker():
        tr = SimpleTracker((100, 100, 3), "ByteTrack", max_age=5, min_hits=2)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        det = [{"bbox": (10, 10, 30, 30), "conf": 0.9, "cls": "car", "true_id": 1}]
        tracks, m0 = tr.update(frame, det, 0, 0.0)
        tracks, m1 = tr.update(frame, det, 1, 0.1)
        assert m1 == 1
        assert any(t.confirmed and t.cls == "car" for t in tracks)

    def test_motion():
        me = MotionEstimator(0.05, None, {"car": 30.0}, 30.0)
        tr = Track(1, "car", (0, 0, 20, 20), 0.9, None, 0, 0.0)
        me.update(tr)
        tr.centroid = np.array([20.0, 0.0])
        tr.history.append(tr.centroid.copy())
        tr.timestamps.append(0.1)
        me.update(tr)
        assert tr.speed >= 0.0

    check("IoU", test_iou)
    check("Segment intersection", test_segments)
    check("Polygon containment", test_point_poly)
    check("Simple tracker persistence", test_tracker)
    check("Motion estimator", test_motion)

    return pd.DataFrame(results)


# -----------------------------------------------------------------------------
# Sidebar configuration and UI
# -----------------------------------------------------------------------------
st.sidebar.title("⚙️ Configuration")

if st.sidebar.button("🔄 Regenerate demo"):
    st.session_state.pop("result", None)
    st.session_state.pop("_params_json", None)

mode_label = st.sidebar.selectbox(
    "Input mode",
    ["Simulation", "Webcam", "Video file", "RTSP stream", "Image folder"],
    index=0,
)
scenario = st.sidebar.selectbox(
    "Use case",
    ["traffic", "warehouse", "agriculture", "safety"],
    index=0,
)

if st.session_state.get("_last_scenario") != scenario:
    st.session_state.pop("computed_homography", None)
    st.session_state["_last_scenario"] = scenario

tracker = st.sidebar.selectbox("Tracker", ["ByteTrack", "BoT-SORT"], index=0)

with st.sidebar.expander("YOLO model / custom weights"):
    model_select = st.selectbox(
        "Base model",
        [
            "yolo11n.pt",
            "yolo11s.pt",
            "yolo11m.pt",
            "yolo11l.pt",
            "yolo11x.pt",
            "yolov8n.pt",
            "yolov8s.pt",
            "yolov8m.pt",
            "yolov8l.pt",
            "yolov8x.pt",
        ],
        index=0,
    )
    custom_weights = st.file_uploader("Drop custom trained weights (.pt)", type=["pt"])
    conf = st.slider("Detection confidence", 0.05, 0.95, 0.25, 0.01)
    iou_th = st.slider("NMS IoU", 0.10, 0.90, 0.45, 0.01)
    use_native_tracker = st.checkbox("Use Ultralytics native tracker", value=True)
    tile = st.checkbox("Tiled inference for small/distant objects", value=False)
    classes_text = st.text_input("Class ID filter, e.g. 0,2,5 (empty = all)", "")

with st.sidebar.expander("Motion calibration"):
    default_ppm = {
        "traffic": 0.120,
        "warehouse": 0.050,
        "agriculture": 0.080,
        "safety": 0.040,
    }.get(scenario, 0.030)
    pixels_per_meter = st.number_input(
        "Pixel-to-meter scale (m/px)",
        min_value=0.001,
        max_value=1.0,
        value=float(default_ppm),
        step=0.001,
        format="%.4f",
    )
    homography_text = st.text_area("Optional homography 3x3 JSON", "", height=100)

    st.markdown("**Four-point homography helper**")
    src_text = st.text_area(
        "Image points [[x,y],[x,y],[x,y],[x,y]]",
        "[[120,90],[820,90],[820,420],[120,420]]",
        height=80,
    )
    dst_text = st.text_area(
        "Metric points [[x,y],[x,y],[x,y],[x,y]]",
        "[[0,0],[10,0],[10,5],[0,5]]",
        height=80,
    )
    colh1, colh2 = st.columns(2)
    with colh1:
        if st.button("Compute homography"):
            try:
                src = np.asarray(json.loads(src_text), dtype=float)
                dst = np.asarray(json.loads(dst_text), dtype=float)
                if src.shape != (4, 2) or dst.shape != (4, 2):
                    raise ValueError("Need exactly four 2D points for source and destination.")
                H = cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32))
                st.session_state["computed_homography"] = H.tolist()
                st.success("Homography computed and stored for this session.")
            except Exception as exc:
                st.error(f"Could not compute homography: {exc}")
    with colh2:
        if st.button("Clear homography"):
            st.session_state.pop("computed_homography", None)
            st.info("Homography cleared.")

    if "computed_homography" in st.session_state:
        st.success("Using calibrated homography from session state.")

with st.sidebar.expander("Zones and counting lines"):
    default_lines, default_zones = default_geometry(scenario)
    lines_text = st.text_area(
        "Lines JSON",
        json.dumps(default_lines, indent=2),
        key=f"lines_json_{scenario}",
        height=180,
    )
    zones_text = st.text_area(
        "Zones JSON",
        json.dumps(default_zones, indent=2),
        key=f"zones_json_{scenario}",
        height=220,
    )
    st.caption("Points are normalized to [0,1]. You may also use pixel coordinates.")

with st.sidebar.expander("Events and rendering"):
    default_limits = default_class_limits(scenario)
    default_speed_limit = st.number_input(
        "Default speed limit (km/h)",
        min_value=1.0,
        max_value=200.0,
        value=float(default_limits.get("default", 30.0)),
        step=1.0,
        key=f"speed_limit_{scenario}",
    )
    loiter_seconds = st.number_input(
        "Loitering threshold (s)",
        min_value=1.0,
        max_value=120.0,
        value=6.0,
        step=0.5,
    )
    privacy_blur = st.checkbox("Privacy blur (faces / plates)", value=False)
    trail_length = st.slider("Motion trail length", 10, 60, 45, 1)

with st.sidebar.expander("Simulation / capture settings"):
    duration = st.slider("Demo duration (s)", 2, 20, 8, 1)
    fps = st.select_slider("FPS", options=[10, 12, 15, 20, 24, 25, 30], value=15)
    resolution = st.selectbox("Resolution", ["640x360", "960x540", "1280x720"], index=1)
    seed = st.number_input("Random seed", min_value=0, max_value=100000, value=42, step=1)

source_type = "simulation"
source_value = None
config_warnings = []

raw_mode = mode_label.lower().replace(" ", "_")
mode_map = {
    "simulation": "simulation",
    "webcam": "webcam",
    "video_file": "video",
    "rtsp_stream": "rtsp",
    "image_folder": "images",
}
mode = mode_map.get(raw_mode, "simulation")

if mode != "simulation":
    if mode == "webcam":
        source_type = "webcam"
        source_value = 0
    elif mode == "video":
        uploaded_video = st.sidebar.file_uploader("Upload video file", type=["mp4", "avi", "mov", "mkv"])
        source_type = "video"
        source_value = stable_upload(uploaded_video, "video") if uploaded_video is not None else None
    elif mode == "rtsp":
        rtsp_url = st.sidebar.text_input("RTSP URL", "rtsp://username:password@ip:port/stream")
        url = rtsp_url.strip()
        source_type = "rtsp"
        if not url or "username" in url or "password" in url or "ip" in url:
            source_value = None
        else:
            source_value = url
    elif mode == "images":
        image_files = st.sidebar.file_uploader(
            "Upload images",
            type=["jpg", "jpeg", "png", "bmp"],
            accept_multiple_files=True,
        )
        source_type = "images"
        if image_files:
            key = "images_" + hashlib.md5(
                json.dumps([(f.name, f.size) for f in image_files], sort_keys=True).encode("utf-8")
            ).hexdigest()
            if key not in st.session_state:
                st.session_state[key] = [stable_upload(f, "image") for f in image_files]
            source_value = st.session_state[key]
        else:
            source_value = None

    if source_value is None:
        config_warnings.append("No valid real source selected. Using simulation.")
        mode = "simulation"
        source_type = "simulation"
        source_value = None
    elif not HAVE_ULTRALYTICS:
        config_warnings.append("Ultralytics is not available in this environment. Using simulation.")
        mode = "simulation"
        source_type = "simulation"
        source_value = None

weights_path = model_select
if custom_weights is not None:
    weights_path = stable_upload(custom_weights, "weights")

parsed_classes = []
if classes_text.strip():
    try:
        parsed_classes = [int(x.strip()) for x in classes_text.split(",") if x.strip()]
    except Exception:
        config_warnings.append("Invalid class filter. Using all classes.")
        parsed_classes = []

clean_lines, line_warn = sanitize_geometry(lines_text, default_lines, "line")
clean_zones, zone_warn = sanitize_geometry(zones_text, default_zones, "zone")
config_warnings.extend(line_warn)
config_warnings.extend(zone_warn)

computed_H = st.session_state.get("computed_homography", None)
parsed_H = parse_homography(homography_text)
homography = computed_H if computed_H is not None else parsed_H
if computed_H is None and homography_text.strip() and parsed_H is None:
    config_warnings.append("Invalid homography JSON. Using pixel-to-meter scale only.")

class_limits = default_class_limits(scenario)
class_limits["default"] = float(default_speed_limit)

width, height = map(int, resolution.split("x"))

params = {
    "mode": mode,
    "scenario": scenario,
    "width": width,
    "height": height,
    "fps": int(fps),
    "duration": float(duration),
    "seed": int(seed),
    "tracker": tracker,
    "weights": weights_path,
    "conf": float(conf),
    "iou": float(iou_th),
    "classes": parsed_classes,
    "tile": bool(tile),
    "use_ultralytics_tracker": bool(use_native_tracker),
    "pixels_per_meter": float(pixels_per_meter),
    "homography": homography,
    "lines": clean_lines,
    "zones": clean_zones,
    "class_limits": class_limits,
    "loiter_seconds": float(loiter_seconds),
    "privacy_blur": bool(privacy_blur),
    "trail_length": int(trail_length),
    "source_type": source_type,
    "source_value": source_value,
}

params_json = json.dumps(params, sort_keys=True)

# Auto-run demo immediately. No blank dashboard.
result = None
if st.session_state.get("_params_json") != params_json or "result" not in st.session_state:
    with st.spinner("Generating live demo video, events, and analytics..."):
        try:
            result = run_pipeline(params)
        except Exception as exc:
            st.error(f"Pipeline failed: {exc}")
            st.code(traceback.format_exc())
            fallback_params = dict(params)
            fallback_params.update(
                {
                    "mode": "simulation",
                    "source_type": "simulation",
                    "source_value": None,
                }
            )
            try:
                result = run_pipeline(fallback_params)
            except Exception as exc2:
                st.error(f"Fallback simulation also failed: {exc2}")
                st.code(traceback.format_exc())
                st.stop()

    if result is not None:
        st.session_state["result"] = result
        st.session_state["_params_json"] = params_json
else:
    result = st.session_state.get("result")

if result is None:
    st.stop()


# -----------------------------------------------------------------------------
# Main page
# -----------------------------------------------------------------------------
st.title("🎥 Real-Time Multi-Object Detection & Tracking Studio")
st.markdown(
    """
This app automatically starts a realistic synthetic simulation and generates an annotated video,
event log, track log, and analytics summary. Switch the sidebar to webcam, video file, RTSP, or images
to use YOLO + ByteTrack / BoT-SORT when available.
"""
)

all_warnings = config_warnings + result.get("warnings", [])
if all_warnings:
    for warn in all_warnings:
        st.warning(warn)

perf = result["summary"]["performance"]
tracking = result["summary"]["tracking"]
analytics = result["summary"]["analytics"]

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Frames processed", f"{perf['frames_processed']}")
m2.metric("Average FPS", f"{perf['average_fps']:.1f}")
m3.metric("Unique tracks", f"{tracking['unique_tracks']}")
m4.metric("Events", f"{analytics['total_events']}")
m5.metric("Pipeline latency", f"{perf['total_pipeline_latency_ms']:.1f} ms")

st.subheader("Generated live demo video")
if result.get("video_path") and os.path.exists(result["video_path"]):
    with open(result["video_path"], "rb") as f:
        video_bytes = f.read()
    st.video(video_bytes, format=result.get("video_mime", "video/mp4"))
else:
    st.warning("No compatible video codec was available. Showing fallback frames from the simulation.")
    fallback = result.get("fallback_frames", [])
    if fallback:
        cols = st.columns(min(4, len(fallback)))
        for i, img in enumerate(fallback[:12]):
            with cols[i % len(cols)]:
                st.image(img, use_container_width=True)

st.subheader("Latest annotated frame")
if result.get("last_frame"):
    st.image(result["last_frame"], use_container_width=True)
else:
    st.info("No frame available.")

tab_events, tab_analytics, tab_downloads, tab_tests, tab_help = st.tabs(
    ["Events", "Analytics", "Downloads", "Tests & Benchmark", "How to extend"]
)

with tab_events:
    st.subheader("Event log")
    events = result.get("events", [])
    if events:
        df_events = pd.DataFrame(events)
        st.dataframe(df_events, use_container_width=True, height=320)

        st.subheader("Saved snapshots")
        snapshots = result.get("snapshots", [])
        if snapshots:
            cols = st.columns(min(4, len(snapshots)))
            for i, path in enumerate(snapshots[:16]):
                with cols[i % len(cols)]:
                    st.image(path, caption=os.path.basename(path), use_container_width=True)
        else:
            st.info("No snapshot files were written.")
    else:
        st.info("No events were triggered in this run. Try lowering speed limit, enabling wrong-way line, or increasing duration.")

with tab_analytics:
    st.subheader("Structured analytics summary")
    st.json(result["summary"])

    st.subheader("Per-stage latency")
    stage_df = pd.DataFrame.from_dict(perf["stage_latency_ms"], orient="index", columns=["milliseconds"])
    stage_df.index.name = "stage"
    st.bar_chart(stage_df)

    st.subheader("Line crossing counts")
    line_df = pd.DataFrame(result["summary"]["analytics"]["line_counts"]).T
    if not line_df.empty:
        st.dataframe(line_df, use_container_width=True)
    else:
        st.info("No lines configured.")

    st.subheader("Zone maximum dwell time")
    zone_df = pd.DataFrame([result["summary"]["analytics"]["zone_max_dwell_s"]]).T
    if not zone_df.empty:
        zone_df.columns = ["max_dwell_s"]
    st.dataframe(zone_df, use_container_width=True)

with tab_downloads:
    st.subheader("Download artifacts")

    if result.get("video_path") and os.path.exists(result["video_path"]):
        with open(result["video_path"], "rb") as f:
            st.download_button(
                "Download annotated video",
                f.read(),
                file_name=os.path.basename(result["video_path"]),
                mime=result.get("video_mime", "video/mp4"),
            )

    if result.get("events_csv") and os.path.exists(result["events_csv"]):
        with open(result["events_csv"], "rb") as f:
            st.download_button(
                "Download events CSV",
                f.read(),
                file_name="events.csv",
                mime="text/csv",
            )

    if result.get("tracks_csv") and os.path.exists(result["tracks_csv"]):
        with open(result["tracks_csv"], "rb") as f:
            st.download_button(
                "Download tracks CSV",
                f.read(),
                file_name="tracks.csv",
                mime="text/csv",
            )

    summary_json_path = result.get("summary_json")
    if summary_json_path and os.path.exists(summary_json_path):
        with open(summary_json_path, "rb") as f:
            st.download_button(
                "Download analytics JSON",
                f.read(),
                file_name="analytics_summary.json",
                mime="application/json",
            )

with tab_tests:
    st.subheader("Built-in unit tests")
    test_df = run_self_tests()
    st.dataframe(test_df, use_container_width=True)

    st.subheader("Mini benchmark: ByteTrack vs BoT-SORT fallback")
    st.caption("Runs two short 3-second simulations using the current scenario and settings.")
    if st.button("Run mini benchmark"):
        rows = []
        with st.spinner("Benchmarking..."):
            for tr_name in ["ByteTrack", "BoT-SORT"]:
                bench_params = dict(params)
                bench_params["tracker"] = tr_name
                bench_params["mode"] = "simulation"
                bench_params["source_type"] = "simulation"
                bench_params["source_value"] = None
                bench_params["duration"] = 3.0
                bench_result = run_pipeline(bench_params)
                bp = bench_result["summary"]["performance"]
                rows.append(
                    {
                        "tracker": tr_name,
                        "frames": bp["frames_processed"],
                        "avg_fps": round(bp["average_fps"], 2),
                        "detect_ms": round(bp["stage_latency_ms"].get("detect", 0.0), 2),
                        "track_ms": round(bp["stage_latency_ms"].get("track", 0.0), 2),
                        "motion_ms": round(bp["stage_latency_ms"].get("motion", 0.0), 2),
                        "analytics_ms": round(bp["stage_latency_ms"].get("analytics", 0.0), 2),
                        "render_ms": round(bp["stage_latency_ms"].get("render", 0.0), 2),
                        "total_ms": round(bp["total_pipeline_latency_ms"], 2),
                        "events": bench_result["summary"]["analytics"]["total_events"],
                    }
                )
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

with tab_help:
    st.subheader("Architecture")
    st.markdown(
        """
The implementation follows a modular pipeline:

`VideoSource -> Detector -> Tracker -> MotionEstimator -> ZoneAnalytics -> EventEngine -> Renderer / Logger`

- **Simulation mode** generates realistic moving objects, occlusions, noise, and false positives.
- **Detector wrapper** supports Ultralytics YOLO weights, custom `.pt` drop-in weights, confidence/NMS controls, class filtering, and optional tiled inference.
- **Tracker stage** uses Ultralytics ByteTrack / BoT-SORT when real detection is active. In simulation or fallback mode, it uses a built-in geometry + appearance tracker with persistent IDs and track history.
- **Motion estimation** supports pixel-to-meter scaling or a four-point homography. Speed is smoothed over recent track history.
- **Zone analytics** counts line crossings by direction, measures dwell time, and triggers intrusion, loitering, wrong-way, speeding, and missing-helmet events.
- **Renderer** draws fading trails, bounding boxes, IDs, confidence, speed, state, HUD, and optional privacy blur.
- **Logger** writes annotated video, event CSV, track CSV, JSON summary, and snapshot images.
"""
    )

    st.subheader("How to add a new class or camera")
    st.markdown(
        """
### New class
1. Add the class name and color in `CLASS_COLORS`.
2. Add speed limits in `default_class_limits()` for the relevant scenario.
3. If using simulation, spawn the class in `SyntheticScene._spawn()`.
4. If using a custom YOLO model, make sure the trained model emits that class name or ID.

### New camera
1. Select **RTSP stream** and enter the URL, or select **Video file** / **Webcam**.
2. The threaded source keeps capture independent from inference and drops old frames if inference is slow.
3. If the stream drops, the source attempts to reconnect automatically.
4. Tune `conf`, `iou`, tracker, and homography calibration for the new viewpoint.
"""
    )

st.caption(
    "Privacy note: enable privacy blur for public demos. This app stores temporary artifacts in system temp directories."
)