import json
from io import BytesIO
import unittest

import pandas as pd

from evaluation import ValidationError, csv_bytes, evaluate, prepare, read_table


class EvaluationTests(unittest.TestCase):
    def run_evaluation(self, truths, predictions):
        gold = pd.DataFrame(truths, columns=["item_id", "label"])
        submissions = pd.DataFrame(predictions, columns=["item_id", "annotator", "label"])
        return evaluate(prepare(gold, {"item_id": "item_id", "label": "label"}, "Ground truth"),
                        prepare(submissions, {"item_id": "item_id", "annotator": "annotator", "label": "label"}, "Annotations"))

    def test_perfect_annotations(self):
        summary, details, _ = self.run_evaluation([("1", "a"), ("2", "b")], [("2", "Ann", "b"), ("1", "Ann", "a")])
        for metric in ["accuracy", "precision", "recall", "f1", "coverage"]:
            self.assertEqual(summary.iloc[0][metric], 1)
        self.assertTrue(details.status.eq("correct").all())

    def test_missing_is_false_negative_unknown_id_is_unscored(self):
        summary, details, labels = self.run_evaluation([("1", "a"), ("2", "a")], [("1", "Ann", "a"), ("3", "Ann", "b")])
        result = summary.iloc[0]
        self.assertEqual(result.accuracy, 0.5)
        self.assertEqual(result.coverage, 0.5)
        self.assertEqual(result.precision, 1)
        self.assertEqual(result.recall, 0.5)
        self.assertAlmostEqual(result.f1, 2 / 3)
        self.assertEqual(result.unknown_ids, 1)
        self.assertEqual(set(details.status), {"correct", "missing", "unknown ID"})
        self.assertEqual(labels.label.tolist(), ["a"])

    def test_known_confusion_has_expected_macro_metrics(self):
        summary, _, _ = self.run_evaluation([("1", "a"), ("2", "a"), ("3", "b")],
                                             [("1", "Ann", "a"), ("2", "Ann", "b"), ("3", "Ann", "b")])
        result = summary.iloc[0]
        self.assertAlmostEqual(result.accuracy, 2 / 3)
        self.assertAlmostEqual(result.precision, 0.75)
        self.assertAlmostEqual(result.recall, 0.75)
        self.assertAlmostEqual(result.f1, 2 / 3)

    def test_no_matching_ids_scores_zero(self):
        summary, _, _ = self.run_evaluation([("1", "a")], [("other", "Ann", "a")])
        self.assertEqual(summary.iloc[0].f1, 0)
        self.assertEqual(summary.iloc[0].accuracy, 0)
        self.assertEqual(summary.iloc[0].missing, 1)

    def test_extra_predicted_label_penalizes_macro_precision(self):
        summary, _, labels = self.run_evaluation([("1", "a"), ("2", "a")], [("1", "Ann", "a"), ("2", "Ann", "typo")])
        self.assertEqual(set(labels.label), {"a", "typo"})
        self.assertEqual(summary.iloc[0].precision, 0.5)

    def test_reserved_looking_labels_are_real_labels(self):
        summary, _, _ = self.run_evaluation([("1", "__MISSING__"), ("2", "__MISSING__")], [("1", "Ann", "__MISSING__")])
        self.assertEqual(summary.iloc[0].accuracy, 0.5)
        self.assertEqual(summary.iloc[0].recall, 0.5)

    def test_duplicate_annotator_item_rejected_after_trimming(self):
        with self.assertRaisesRegex(ValidationError, "duplicate"):
            self.run_evaluation([("1", "a")], [("1", "Ann", "a"), (" 1 ", "Ann", "a")])

    def test_duplicate_truth_rejected(self):
        with self.assertRaisesRegex(ValidationError, "duplicate"):
            self.run_evaluation([("1", "a"), ("1", "b")], [("1", "Ann", "a")])

    def test_blank_required_value_rejected(self):
        with self.assertRaisesRegex(ValidationError, "blank"):
            self.run_evaluation([("1", "a")], [("1", "Ann", "  ")])

    def test_csv_preserves_ids_and_na_labels(self):
        frame = read_table(b"item_id,label\n001,NA\n", "test.csv")
        self.assertEqual(frame.iloc[0].item_id, "001")
        self.assertEqual(frame.iloc[0].label, "NA")

    def test_excel_and_json(self):
        frame = pd.DataFrame({"item_id": ["001"], "label": ["NA"]})
        buffer = BytesIO()
        frame.to_excel(buffer, index=False)
        self.assertEqual(read_table(buffer.getvalue(), "test.xlsx").iloc[0].item_id, "001")
        self.assertEqual(read_table(frame.to_json(orient="records").encode(), "test.json").iloc[0].label, "NA")
        with self.assertRaisesRegex(ValidationError, "array"):
            read_table(json.dumps({"rows": []}).encode(), "test.json")

    def test_label_case_is_optional_ids_unchanged(self):
        frame = pd.DataFrame({"id": ["ABC"], "class": [" Positive "]})
        mapping = {"item_id": "id", "label": "class"}
        self.assertEqual(prepare(frame, mapping, "Truth").iloc[0].label, "Positive")
        folded = prepare(frame, mapping, "Truth", True)
        self.assertEqual(folded.iloc[0].label, "positive")
        self.assertEqual(folded.iloc[0].item_id, "ABC")

    def test_spreadsheet_formulas_escaped_on_export(self):
        exported = csv_bytes(pd.DataFrame({"label": ["=1+1", "@SUM(A1)", "normal"]})).decode("utf-8-sig")
        self.assertIn("'=1+1", exported)
        self.assertIn("'@SUM(A1)", exported)

    def test_annotators_are_evaluated_independently(self):
        summary, _, _ = self.run_evaluation([("1", "a"), ("2", "b")],
                                             [("1", "Ann", "a"), ("2", "Ann", "b"), ("1", "Bob", "b")])
        scores = summary.set_index("annotator")
        self.assertEqual(scores.loc["Ann", "accuracy"], 1)
        self.assertEqual(scores.loc["Bob", "accuracy"], 0)
        self.assertEqual(scores.loc["Bob", "coverage"], 0.5)


if __name__ == "__main__":
    unittest.main()
