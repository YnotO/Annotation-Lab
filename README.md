# Annotation Lab

A local Streamlit MVP for evaluating annotators against ground truth. Supports single-label classification and oriented LiDAR 3D cuboid detection.

## Start

Requires Python 3.11 or newer. From this folder in PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open http://127.0.0.1:8501. Start with **Sample project**, then select **Upload your data**. Stop the server with Ctrl+C. An environment already prepared in this workspace can be started with only the last command.

Alternatively, run `.\run.ps1` from PowerShell after installing the dependencies. The helper starts the app from its own folder.

## Classification files

Use CSV (UTF-8), XLSX (first worksheet), or JSON (array of objects). Combine all annotators in a single submission file. Map your own column names in the app.

Ground truth:

```csv
item_id,label
001,positive
002,negative
```

Annotations:

```csv
item_id,annotator,label
001,Amina,positive
002,Amina,negative
001,Brian,negative
002,Brian,negative
```

An optional ground-truth column named `text`, `content`, or `description` appears in item review. Keep IDs as text in Excel to retain leading zeros. Leading/trailing whitespace is trimmed. Label case can optionally be ignored. IDs and annotator names are case-sensitive. Nulls, blanks, duplicate ground-truth IDs and duplicate annotator/item pairs are rejected.

## Classification evaluation

- Every ground-truth item is expected from each annotator appearing in the submissions. Use a ground-truth file restricted to the shared assignment when evaluating a subset. Annotators who submit no rows are not discoverable in this MVP.
- Accuracy = correct / ground-truth item count. Coverage = known submitted item count / ground-truth item count.
- Missing items are incorrect for accuracy and false negatives for their expected class. They create no positive predictions.
- Unknown IDs are excluded from scores and shown separately for review.
- Precision, recall, and F1 use an unweighted macro average over all ground-truth labels plus additional labels predicted on known items. Undefined label metrics are zero. Per-label support is the count in ground truth.
- Dashboard summary metrics average over annotators equally. A low-coverage annotator can have high precision: compare recall, F1, and coverage as well.
- Confusion matrix rows are expected labels; columns are submitted labels, with a separate missing annotation column.

## Reports and privacy

Download a filtered review CSV or a ZIP containing annotator scores, all item comparisons, per-label metrics, and evaluation rules. CSV metrics are ratios from 0 to 1; the interface displays percentages. Potential spreadsheet formulas in text cells are prefixed with an apostrophe on export.

The application binds to localhost by default, processes uploads in memory, and does not write uploaded data or results to disk. There is no database, sign-in, saved history, or assignment roster. Do not expose it as a public service without adding access controls and a persistence policy. Streamlit upload limit: 25 MB per file. Large files or many annotators can require significant memory because evaluation generates one row per annotator and ground-truth item.

LiDAR evaluation results are cached in server memory (up to five input configurations) to keep review filters responsive. They are not written to disk. Restart the server to clear the cache.

## LiDAR 3D boxes

Choose **LiDAR 3D boxes** under **Evaluation type** in the sidebar. Start with the synthetic sample project, or choose **Upload your data** and:

1. Upload a ground-truth JSON and one JSON per annotator.
2. Select the stage, sensor and label set for each export where choices exist. For example, Vega exports containing both `Labelling` and `QA` must be scoped to one stage. The stages can be different between ground truth and submissions (for example, QA ground truth versus Labelling submissions).
3. Give each submission a unique annotator name. Embedded `user_id` and `labeller_email` fields are not used to group objects; one selected export represents one annotator.
4. Set the minimum 3D IoU (default 0.5). Inspect scores, box comparisons and the 3D frame overlay, then download the ZIP report.

Supported format: Deepen-style JSON with a top-level `labels` array:

```json
{
  "labels": [{
    "file_id": "0.pcd",
    "label_id": "car:1",
    "label_category_id": "car",
    "label_type": "3d_bbox",
    "stage_id": "Labelling",
    "sensor_id": "lidar",
    "label_set_id": "default",
    "three_d_bbox": {
      "cx": 0, "cy": 0, "cz": 1,
      "l": 4, "w": 2, "h": 2,
      "quaternion": {"x": 0, "y": 0, "z": 0, "w": 1}
    }
  }]
}
```

Geometry conventions: center coordinates `cx/cy/cz`; local X=length, Y=width, Z=height; active local-to-world quaternion in `(x,y,z,w)` order. Non-unit quaternions are normalized. Both exports must use the same coordinate system, units and calibration. Conversion between coordinate systems is not performed. Positive finite dimensions and finite nonzero quaternions are required. Duplicate frame/object pairs within a selected scope are rejected.

### LiDAR scoring

- Match only within the same frame and class. IDs may differ across exports. Each cuboid can match only once.
- Calculate oriented 3D IoU using actual cuboid intersection volume, including roll, pitch and yaw. Geometry is translated and scaled before convex hull calculations to reduce numerical error at large global coordinates.
- At the selected IoU threshold, choose the largest number of eligible one-to-one matches, breaking ties by maximum total IoU. This custom annotation comparison policy is not a benchmark AP/mAP implementation.
- Threshold comparisons allow a relative tolerance of 1e-9 for floating-point geometry calculations.
- A match is a true positive. Unmatched ground-truth boxes are false negatives; unmatched submitted boxes are false positives. Wrong classes and insufficient overlaps produce one missed box and one extra box. Boxes in additional submitted frames are also false positives.
- Precision = TP/(TP+FP); recall = TP/(TP+FN); F1 = 2TP/(2TP+FP+FN). Scores aggregate object counts across frames per annotator. Undefined scores are zero. The dashboard mean gives annotators equal weight.
- Report mean IoU and center distance on matched pairs only. Per-pair review also includes mean absolute dimension error and the shortest full rotation difference in degrees. Distance and dimension errors use source coordinate units. Rotation error is sensitive to object-axis direction even for geometrically symmetric boxes.
- Entire selected exports are scored. Restrict input files to the same assigned frame set before evaluation if annotators have different assignments. Missing ground-truth frames count as missed work. An empty submission (`{"labels": []}`) is supported; empty ground truth is rejected. Frames with no boxes cannot be inferred without a separate frame manifest.

The 3D viewer overlays ground-truth and submitted wireframe cuboids; it does not load `.pcd` point clouds. Raw point-cloud rendering, point-level segmentation, track identity scoring, text spans, multi-label selections and free-text similarity are not included.

Geometry uses SciPy [quaternion rotations](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_quat.html) and [convex hull volumes](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.ConvexHull.html).

## Verification

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The tests cover classification scoring, file formats, validation and exports; LiDAR analytical intersections (including roll), large-coordinate geometry, scope selection, one-to-one matching, incorrect classes, missing frames, and empty submissions. Streamlit AppTest also checks both dashboard modes, sample metrics, filters, threshold changes, and upload screens.
