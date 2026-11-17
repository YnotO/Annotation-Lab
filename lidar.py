"""Deepen-style cuboid parsing and full oriented 3D detection evaluation."""
from dataclasses import dataclass
from itertools import product
import json

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.transform import Rotation

from evaluation import ValidationError

SIGNS = np.array(list(product([-1, 1], repeat=3)))
EDGES = [(i, j) for i in range(8) for j in range(i + 1, 8)
         if np.count_nonzero(SIGNS[i] != SIGNS[j]) == 1]


@dataclass
class Box:
    frame: str
    object_id: str
    label: str
    center: np.ndarray
    size: np.ndarray  # local x=length, y=width, z=height
    rotation: np.ndarray  # active local-to-world matrix

    def corners(self, origin=None):
        origin = self.center if origin is None else origin
        return (SIGNS * self.size / 2) @ self.rotation.T + (self.center - origin)


def load_export(data: bytes) -> list:
    try:
        document = json.loads(data.decode("utf-8-sig"))
    except (ValueError, UnicodeError) as exc:
        raise ValidationError("LiDAR upload must be valid UTF-8 JSON.") from exc
    if not isinstance(document, dict) or not isinstance(document.get("labels"), list):
        raise ValidationError('LiDAR JSON must contain a top-level "labels" array.')
    if not all(isinstance(row, dict) for row in document["labels"]):
        raise ValidationError("Each entry in labels must be an object.")
    return document["labels"]


def scope_options(records, field):
    return sorted({str(row.get(field) or "").strip() for row in records})


def parse_boxes(records, *, stage=None, sensor=None, label_set=None, ignore_case=False):
    selected = records
    for field, choice in [("stage_id", stage), ("sensor_id", sensor), ("label_set_id", label_set)]:
        options = scope_options(selected, field)
        if choice is None and len(options) > 1:
            raise ValidationError(f"Select one {field} before evaluation; do not combine annotation scopes.")
        if choice is not None:
            if options and choice not in options:
                raise ValidationError(f"Selected {field} does not exist in this export.")
            selected = [row for row in selected if str(row.get(field) or "").strip() == choice]
    boxes, seen = [], set()
    for index, row in enumerate(selected, 1):
        prefix = f"LiDAR label {index}"
        if row.get("label_type") != "3d_bbox":
            raise ValidationError(f"{prefix}: only 3d_bbox labels are supported. Select a cuboid-only export.")
        fields = []
        for field in ["file_id", "label_id", "label_category_id"]:
            value = row.get(field)
            if not isinstance(value, (str, int)) or not str(value).strip():
                raise ValidationError(f"{prefix}: {field} must be a nonblank ID or name.")
            fields.append(str(value).strip())
        frame, object_id, label = fields
        if (frame, object_id) in seen:
            raise ValidationError(f"Duplicate object ID {object_id!r} in frame {frame!r}. Select a single stage, sensor and label set.")
        seen.add((frame, object_id))
        try:
            cuboid = row["three_d_bbox"]
            center = np.array([cuboid[k] for k in ["cx", "cy", "cz"]], dtype=float)
            size = np.array([cuboid[k] for k in ["l", "w", "h"]], dtype=float)
            quat = np.array([cuboid["quaternion"][k] for k in ["x", "y", "z", "w"]], dtype=float)
            if center.shape != (3,) or size.shape != (3,) or quat.shape != (4,):
                raise ValueError("geometry components must be scalars")
            if not all(np.isfinite(v).all() for v in [center, size, quat]):
                raise ValueError("nonfinite geometry")
            if np.any(size <= 0) or np.linalg.norm(quat) < 1e-12:
                raise ValueError("nonpositive size or zero quaternion")
            rotation = Rotation.from_quat(quat).as_matrix()
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValidationError(f"{prefix}: provide finite cx/cy/cz, positive l/w/h and a nonzero x/y/z/w quaternion.") from exc
        boxes.append(Box(frame, object_id, label.casefold() if ignore_case else label, center, size, rotation))
    return boxes


def iou_3d(a: Box, b: Box) -> float:
    """Intersect two convex cuboids, retaining roll, pitch and yaw.

    Intersection vertices are contained box corners and edge/face intersections.
    Translate and scale before Qhull to avoid precision loss at global coordinates.
    """
    scale = max(float(a.size.max()), float(b.size.max()))
    ca, cb = a.corners(a.center) / scale, b.corners(a.center) / scale
    if np.any(np.minimum(ca.max(0), cb.max(0)) - np.maximum(ca.min(0), cb.min(0)) <= 0):
        return 0.0
    normals_a = np.concatenate([a.rotation.T, -a.rotation.T])
    normals_b = np.concatenate([b.rotation.T, -b.rotation.T])
    offsets_a = np.tile(a.size / (2 * scale), 2)
    offsets_b = np.tile(b.size / (2 * scale), 2) + normals_b @ ((b.center - a.center) / scale)
    normals = np.concatenate([normals_a, normals_b])
    offsets = np.concatenate([offsets_a, offsets_b])
    candidates = [ca, cb]
    for corners, face_normals, face_offsets in [(ca, normals_b, offsets_b), (cb, normals_a, offsets_a)]:
        for start, end in EDGES:
            direction = corners[end] - corners[start]
            denominator = face_normals @ direction
            valid = np.abs(denominator) > 1e-12
            t = (face_offsets[valid] - face_normals[valid] @ corners[start]) / denominator[valid]
            t = t[(t >= -1e-10) & (t <= 1 + 1e-10)]
            if len(t):
                candidates.append(corners[start] + t[:, None] * direction)
    points = np.concatenate(candidates)
    points = points[np.all(points @ normals.T <= offsets + 1e-9, axis=1)]
    points = np.unique(np.round(points, 12), axis=0)
    if len(points) < 4:
        return 0.0
    try:
        intersection = ConvexHull(points).volume
    except QhullError:
        if np.linalg.matrix_rank(points - points[0], tol=1e-9) < 3:
            return 0.0
        raise ValidationError("Could not reliably intersect these cuboids. Check box geometry and coordinate scales.")
    va, vb = np.prod(a.size / scale), np.prod(b.size / scale)
    intersection = np.clip(intersection, 0, min(va, vb))
    return float(np.clip(intersection / (va + vb - intersection), 0, 1))


def match_boxes(truth, predictions, threshold):
    matrix = np.zeros((len(truth), len(predictions)))
    for i, gold in enumerate(truth):
        for j, predicted in enumerate(predictions):
            if gold.label == predicted.label:
                matrix[i, j] = iou_3d(gold, predicted)
    if not matrix.size:
        return [], set(range(len(truth))), set(range(len(predictions)))
    # Treat roundoff at the threshold as equality, including identical rotated
    # boxes whose computed IoU can be a few trillionths below one.
    eligible = (matrix >= threshold) | np.isclose(matrix, threshold, rtol=1e-9, atol=0)
    # One more eligible match always outweighs the total possible IoU tie-break.
    reward = eligible * (min(matrix.shape) + 1 + matrix)
    rows, cols = linear_sum_assignment(-reward)
    pairs = [(int(i), int(j), float(matrix[i, j])) for i, j in zip(rows, cols) if eligible[i, j]]
    return pairs, set(range(len(truth))) - {i for i, _, _ in pairs}, set(range(len(predictions))) - {j for _, j, _ in pairs}


DETAIL_COLUMNS = ["annotator", "frame", "status", "expected_id", "submitted_id", "label", "iou_3d",
                  "center_error", "size_mae", "rotation_error_deg"]


def detection_scores(tp, fp, fn):
    return {"matched": tp, "extra": fp, "missed": fn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}


def evaluate_lidar(truth, submissions, threshold=0.5):
    if not 0 < threshold <= 1:
        raise ValidationError("IoU threshold must be greater than 0 and at most 1.")
    if not truth:
        raise ValidationError("Selected ground-truth scope contains no cuboids.")
    if not submissions:
        raise ValidationError("Add at least one annotator export.")
    summaries, details, classes = [], [], []
    for annotator, boxes in submissions.items():
        annotator_rows = []
        frames = sorted({b.frame for b in truth} | {b.frame for b in boxes})
        for frame in frames:
            gold = [b for b in truth if b.frame == frame]
            predicted = [b for b in boxes if b.frame == frame]
            pairs, missed, extra = match_boxes(gold, predicted, threshold)
            for i, j, iou in pairs:
                a, b = gold[i], predicted[j]
                annotator_rows.append({"annotator": annotator, "frame": frame, "status": "matched",
                    "expected_id": a.object_id, "submitted_id": b.object_id, "label": a.label, "iou_3d": iou,
                    "center_error": float(np.linalg.norm(a.center - b.center)),
                    "size_mae": float(np.abs(a.size - b.size).mean()),
                    "rotation_error_deg": float(np.degrees(Rotation.from_matrix(a.rotation.T @ b.rotation).magnitude()))})
            for indices, items, status in [(missed, gold, "missed"), (extra, predicted, "extra")]:
                for i in sorted(indices):
                    box = items[i]
                    annotator_rows.append({"annotator": annotator, "frame": frame, "status": status,
                        "expected_id": box.object_id if status == "missed" else "",
                        "submitted_id": box.object_id if status == "extra" else "", "label": box.label})
        report = pd.DataFrame(annotator_rows, columns=DETAIL_COLUMNS)
        details.extend(annotator_rows)
        tp, fp, fn = (int(report.status.eq(status).sum()) for status in ["matched", "extra", "missed"])
        good = report[report.status.eq("matched")]
        summaries.append({"annotator": annotator, **detection_scores(tp, fp, fn),
                          "mean_iou": good.iou_3d.mean(), "mean_center_error": good.center_error.mean(),
                          "ground_truth_boxes": len(truth), "submitted_boxes": len(boxes)})
        for label, rows in report.groupby("label", sort=True):
            counts = [int(rows.status.eq(status).sum()) for status in ["matched", "extra", "missed"]]
            classes.append({"annotator": annotator, "label": label, **detection_scores(*counts)})
    return (pd.DataFrame(summaries).sort_values("f1", ascending=False).reset_index(drop=True),
            pd.DataFrame(details, columns=DETAIL_COLUMNS), pd.DataFrame(classes))
