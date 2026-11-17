import json
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from evaluation import ValidationError
from lidar import Box, evaluate_lidar, iou_3d, load_export, match_boxes, parse_boxes


def box(x=0, y=0, z=0, size=(2, 2, 2), angles=(0, 0, 0), label="car", ident="one", frame="0.pcd"):
    return Box(frame, ident, label, np.array([x, y, z], dtype=float), np.array(size, dtype=float),
               Rotation.from_euler("xyz", angles, degrees=True).as_matrix())


def record(stage="Labelling"):
    return {"file_id": "0.pcd", "label_id": "car:1", "label_category_id": "car", "label_type": "3d_bbox", "stage_id": stage,
            "three_d_bbox": {"cx": 0, "cy": 0, "cz": 0, "l": 4, "w": 2, "h": 1,
                             "quaternion": {"x": 0, "y": 0, "z": 0, "w": 1}}}


class GeometryTests(unittest.TestCase):
    def test_identical_arbitrary_rotation_at_global_coordinates(self):
        a = box(x=66000, y=41000, z=60, size=(4.6, 1.8, 1.5), angles=(17, 23, 82))
        self.assertAlmostEqual(iou_3d(a, a), 1, places=9)

    def test_disjoint_and_touching(self):
        self.assertEqual(iou_3d(box(), box(x=3)), 0)
        self.assertEqual(iou_3d(box(), box(x=2)), 0)

    def test_analytic_overlap_and_containment(self):
        self.assertAlmostEqual(iou_3d(box(), box(x=1)), 1 / 3, places=9)
        self.assertAlmostEqual(iou_3d(box(size=(4, 4, 4)), box(size=(1, 1, 1), angles=(20, 30, 40))), 1 / 64, places=9)

    def test_yaw_intersection_without_contained_corners(self):
        self.assertAlmostEqual(iou_3d(box(), box(angles=(0, 0, 45))), 1 / np.sqrt(2), places=9)

    def test_roll_is_not_ignored(self):
        a = box(size=(4, 2, 1))
        b = box(size=(4, 2, 1), angles=(90, 0, 0))
        self.assertAlmostEqual(iou_3d(a, b), 1 / 3, places=9)

    def test_translation_and_rotation_invariance(self):
        a, b = box(), box(x=1)
        rotation = Rotation.from_euler("xyz", [23, -34, 75], degrees=True).as_matrix()
        for item in [a, b]:
            item.center = rotation @ item.center + [66000, 41000, 60]
            item.rotation = rotation @ item.rotation
        self.assertAlmostEqual(iou_3d(a, b), 1 / 3, places=8)
        self.assertAlmostEqual(iou_3d(a, b), iou_3d(b, a), places=8)

    def test_quaternion_sign_equivalence(self):
        a, b = record(), record()
        b["three_d_bbox"]["quaternion"]["w"] = -1
        self.assertAlmostEqual(iou_3d(parse_boxes([a])[0], parse_boxes([b])[0]), 1)


class ParserTests(unittest.TestCase):
    def test_top_level_format(self):
        self.assertEqual(len(load_export(json.dumps({"labels": [record()]}).encode())), 1)
        with self.assertRaises(ValidationError):
            load_export(b"[]")

    def test_stages_must_be_selected(self):
        rows = [record(), record("QA")]
        with self.assertRaisesRegex(ValidationError, "stage_id"):
            parse_boxes(rows)
        self.assertEqual(len(parse_boxes(rows, stage="QA")), 1)

    def test_sensor_and_label_set_selection(self):
        a, b = record(), record()
        a["sensor_id"], b["sensor_id"] = "front", "rear"
        with self.assertRaisesRegex(ValidationError, "sensor_id"):
            parse_boxes([a, b])
        self.assertEqual(len(parse_boxes([a, b], sensor="front")), 1)
        a, b = record(), record()
        a["label_set_id"], b["label_set_id"] = "one", "two"
        with self.assertRaisesRegex(ValidationError, "label_set_id"):
            parse_boxes([a, b])
        self.assertEqual(len(parse_boxes([a, b], label_set="one")), 1)

    def test_duplicate_ids_rejected(self):
        with self.assertRaisesRegex(ValidationError, "Duplicate"):
            parse_boxes([record(), record()])

    def test_invalid_geometry(self):
        for field, value in [("h", 0), ("w", -1), ("cx", float("nan")), ("cx", [0])]:
            row = record()
            row["three_d_bbox"][field] = value
            with self.assertRaises(ValidationError):
                parse_boxes([row])
        row = record()
        row["three_d_bbox"]["quaternion"] = dict(x=0, y=0, z=0, w=0)
        with self.assertRaises(ValidationError):
            parse_boxes([row])

    def test_bad_types_and_missing_ids(self):
        row = record()
        row["label_type"] = "segmentation"
        with self.assertRaises(ValidationError):
            parse_boxes([row])
        row = record()
        del row["label_id"]
        with self.assertRaises(ValidationError):
            parse_boxes([row])


class MatchingTests(unittest.TestCase):
    def test_maximizes_match_count_before_iou(self):
        gold = [box(ident="g0"), box(ident="g1")]
        predicted = [box(ident="p0"), box(ident="p1")]
        overlaps = {("g0", "p0"): .99, ("g0", "p1"): .51, ("g1", "p0"): .51, ("g1", "p1"): .49}
        with patch("lidar.iou_3d", side_effect=lambda a, b: overlaps[a.object_id, b.object_id]):
            pairs, missing, extra = match_boxes(gold, predicted, .5)
        self.assertEqual(len(pairs), 2)
        self.assertFalse(missing or extra)

    def test_one_to_one_extra_duplicates(self):
        summary, _, _ = evaluate_lidar([box()], {"Ann": [box(ident="a"), box(ident="b")]})
        self.assertEqual(summary.iloc[0].matched, 1)
        self.assertEqual(summary.iloc[0].extra, 1)
        self.assertEqual(summary.iloc[0].precision, .5)

    def test_wrong_class_is_missed_and_extra(self):
        summary, _, _ = evaluate_lidar([box()], {"Ann": [box(label="truck")]})
        self.assertEqual(summary.iloc[0].missed, 1)
        self.assertEqual(summary.iloc[0].extra, 1)
        self.assertEqual(summary.iloc[0].f1, 0)

    def test_wrong_frame_does_not_match(self):
        summary, _, _ = evaluate_lidar([box()], {"Ann": [box(frame="other.pcd")]})
        self.assertEqual(summary.iloc[0].matched, 0)
        self.assertEqual(summary.iloc[0].extra, 1)
        self.assertEqual(summary.iloc[0].missed, 1)

    def test_empty_submission_and_geometry_errors(self):
        summary, _, _ = evaluate_lidar([box()], {"Ann": []})
        self.assertEqual(summary.iloc[0].recall, 0)
        self.assertTrue(np.isnan(summary.iloc[0].mean_iou))
        summary, details, _ = evaluate_lidar([box()], {"Ann": [box(x=.2)]})
        self.assertAlmostEqual(details.iloc[0].center_error, .2)
        self.assertAlmostEqual(details.iloc[0].rotation_error_deg, 0)

    def test_empty_ground_truth_and_invalid_threshold_rejected(self):
        with self.assertRaises(ValidationError):
            evaluate_lidar([], {"Ann": []})
        with self.assertRaises(ValidationError):
            evaluate_lidar([box()], {"Ann": []}, 0)

    def test_threshold_controls_matching(self):
        gold, predicted = [box()], {"Ann": [box(x=.5)]}  # IoU = .6
        self.assertEqual(evaluate_lidar(gold, predicted, .5)[0].iloc[0].matched, 1)
        self.assertEqual(evaluate_lidar(gold, predicted, .7)[0].iloc[0].matched, 0)

    def test_perfect_rotated_box_at_threshold_one(self):
        a = box(x=66000, y=41000, z=60, size=(4.5664394802, 1.7795400223, 1.549835478), angles=(2, -3, 158))
        self.assertEqual(evaluate_lidar([a], {"Ann": [a]}, 1)[0].iloc[0].matched, 1)
        self.assertEqual(evaluate_lidar([a], {"Ann": [box(label="truck")]}, 1e-12)[0].iloc[0].matched, 0)


if __name__ == "__main__":
    unittest.main()
