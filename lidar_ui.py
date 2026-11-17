"""Streamlit interface for evaluating LiDAR cuboid exports."""
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from evaluation import ValidationError, csv_bytes
from lidar import EDGES, evaluate_lidar, load_export, parse_boxes, scope_options

ROOT = Path(__file__).resolve().parent


def select_scope(records, key):
    scope = {}
    for field, argument, title in [("stage_id", "stage", "Stage"), ("sensor_id", "sensor", "Sensor"),
                                    ("label_set_id", "label_set", "Label set")]:
        options = scope_options(records, field)
        if len(options) > 1:
            value = st.selectbox(title, options, key=f"{key}_{field}", format_func=lambda x: x or "Unspecified")
        else:
            value = options[0] if options else None
            if value:
                st.caption(f"{title}: {value}")
        scope[argument] = value
        if value is not None:
            records = [r for r in records if str(r.get(field) or "").strip() == value]
    return scope


@st.cache_data(show_spinner=False, max_entries=5)
def calculate(truth_data, truth_scope, sources, threshold, ignore_case):
    truth = parse_boxes(load_export(truth_data), **truth_scope, ignore_case=ignore_case)
    submissions = {name: parse_boxes(load_export(data), **scope, ignore_case=ignore_case)
                   for name, data, scope in sources}
    return evaluate_lidar(truth, submissions, threshold), truth, submissions


def wireframes(figure, boxes, origin, name, color):
    x, y, z = [], [], []
    for box in boxes:
        corners = box.corners(origin)
        for start, end in EDGES:
            x.extend([corners[start, 0], corners[end, 0], None])
            y.extend([corners[start, 1], corners[end, 1], None])
            z.extend([corners[start, 2], corners[end, 2], None])
    figure.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="lines", name=name,
                                  line=dict(color=color, width=4), hoverinfo="name"))


def render_lidar(mode, ignore_case):
    st.subheader("LiDAR · 3D cuboid evaluation")
    st.caption("Deepen-style JSON • Full quaternion rotation • One-to-one, same-class matching")
    threshold = st.slider("Minimum 3D IoU for a match", 0.05, 1.0, 0.5, 0.05, key="lidar_threshold")
    st.caption("IoU measures overlapping volume: 1 means identical occupied volume; 0 means no overlap. "
               "Boxes must have the same class and IoU at or above this threshold to match.")
    try:
        if mode == "Sample project":
            st.info("Synthetic LiDAR sample · 2 frames · 4 ground-truth boxes · 2 annotators")
            truth_data = (ROOT / "samples/lidar_ground_truth.json").read_bytes()
            truth_scope = {}
            sources = [(name, (ROOT / f"samples/{filename}").read_bytes(), {}) for name, filename in [
                ("Amina", "lidar_amina.json"), ("Brian", "lidar_brian.json")]]
        else:
            st.write("Upload ground truth and one JSON export per annotator. Select the intended stage for each file.")
            st.info("For exports containing both Labelling and QA, choose one stage per file to avoid counting the same object twice.")
            gt = st.file_uploader("LiDAR ground truth JSON", type="json", key="lidar_gt")
            files = st.file_uploader("LiDAR annotator JSON exports", type="json", accept_multiple_files=True, key="lidar_submissions")
            if gt is None:
                st.info("Add a ground-truth export to continue, or use Sample project.")
                return
            truth_data = gt.getvalue()
            st.markdown("**Ground-truth scope**")
            truth_scope = select_scope(load_export(truth_data), "lidar_truth")
            if not files:
                st.info("Add at least one annotator export to compare with ground truth.")
                return
            sources = []
            for index, uploaded in enumerate(files):
                with st.expander(f"Submission {index + 1}: {uploaded.name}", expanded=True):
                    name = st.text_input("Annotator name", value=Path(uploaded.name).stem, key=f"lidar_name_{index}").strip()
                    if not name:
                        raise ValidationError("Give each submission a nonblank annotator name.")
                    scope = select_scope(load_export(uploaded.getvalue()), f"lidar_submission_{index}")
                    sources.append((name, uploaded.getvalue(), scope))
            if len({name for name, _, _ in sources}) != len(sources):
                raise ValidationError("Annotator names must be unique. One export represents one annotator.")
        st.caption("The entire selected export is scored. Use the same assigned frames, coordinate system, units, "
                   "and sensor calibration in ground truth and submissions. Extra boxes in additional frames count as false positives. "
                   "Local box axes are length=X, width=Y, height=Z, with an active local-to-world quaternion (x, y, z, w).")
        with st.spinner("Matching oriented 3D boxes…"):
            (summary, details, classes), truth, submissions = calculate(truth_data, truth_scope, sources, threshold, ignore_case)
    except ValidationError as exc:
        st.error(str(exc))
        return

    cols = st.columns(4)
    cols[0].metric("Annotators evaluated", len(summary))
    cols[1].metric("Mean detection F1", f"{summary.f1.mean():.1%}")
    cols[2].metric("Missed boxes", int(summary.missed.sum()))
    cols[3].metric("Extra boxes", int(summary.extra.sum()))
    st.caption("F1 uses object counts across frames for each annotator. The displayed mean weights annotators equally.")
    overview, review, scene, exports = st.tabs(["LiDAR overview", "Box review", "3D frame viewer", "LiDAR reports"])
    with overview:
        st.subheader("Detection quality by annotator")
        display = summary.copy()
        display[["precision", "recall", "f1", "mean_iou"]] *= 100
        config = {col: st.column_config.NumberColumn(col.replace("_", " ").title(), format="%.1f%%")
                  for col in ["precision", "recall", "f1", "mean_iou"]}
        st.dataframe(display, hide_index=True, width="stretch", column_config=config)
        st.caption("Mean IoU and center error include matched boxes only; a blank means there were no matches. Center errors use source coordinate units.")
        chart = summary.melt(id_vars="annotator", value_vars=["precision", "recall", "f1"], var_name="Metric", value_name="Score")
        fig = px.bar(chart, x="annotator", y="Score", color="Metric", barmode="group",
                     color_discrete_sequence=["#167D7F", "#5064BA", "#B7C9DC"], template="plotly_white")
        fig.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 1.05], margin=dict(t=20, b=20))
        st.plotly_chart(fig, width="stretch")
        st.markdown("**Per-class results**")
        st.dataframe(classes, hide_index=True, width="stretch")
    with review:
        left, middle, right = st.columns(3)
        who = left.selectbox("Review annotator", summary.annotator.tolist(), key="lidar_review_who")
        statuses = middle.multiselect("Box status", ["matched", "missed", "extra"], default=["matched", "missed", "extra"])
        frame = right.selectbox("Review frame", [None] + sorted(details.frame.unique()), format_func=lambda x: "All frames" if x is None else x)
        filtered = details[details.annotator.eq(who) & details.status.isin(statuses)]
        if frame is not None:
            filtered = filtered[filtered.frame.eq(frame)]
        st.caption(f"{len(filtered)} rows. A wrong class or insufficient overlap produces a missed ground-truth box and an extra submitted box.")
        st.dataframe(filtered, hide_index=True, width="stretch")
        st.download_button("Download box review", csv_bytes(filtered), "lidar_box_review.csv", "text/csv")
    with scene:
        st.caption("Cuboid overlay only; raw point clouds are not loaded. Drag to rotate, scroll to zoom.")
        viewer_who = st.selectbox("Viewer annotator", summary.annotator.tolist())
        frames = sorted({box.frame for box in truth} | {box.frame for box in submissions[viewer_who]})
        viewer_frame = st.selectbox("Viewer frame", frames)
        gold = [b for b in truth if b.frame == viewer_frame]
        predicted = [b for b in submissions[viewer_who] if b.frame == viewer_frame]
        origin = np.mean([b.center for b in gold + predicted], axis=0)
        fig = go.Figure()
        wireframes(fig, gold, origin, "Ground truth", "#167D7F")
        wireframes(fig, predicted, origin, viewer_who, "#D77932")
        fig.update_layout(scene=dict(aspectmode="data", xaxis_title="Relative X", yaxis_title="Relative Y", zaxis_title="Relative Z"),
                          height=550, margin=dict(t=10, b=10, l=0, r=0), legend=dict(orientation="h"))
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Display origin subtracted from all boxes: {origin.round(3).tolist()}. Scores use original coordinates.")
    with exports:
        rules = {"evaluation": "oriented_3d_cuboid_detection", "minimum_iou": threshold,
                 "ignore_label_case": ignore_case, "ground_truth_scope": truth_scope,
                 "submission_scopes": {name: scope for name, _, scope in sources},
                 "matching": "same frame and class; maximize number of eligible one-to-one pairs, then total IoU",
                 "unmatched": "ground truth = false negative; submission = false positive, including additional frames",
                 "axes": "local X=length, Y=width, Z=height; active local-to-world quaternion x,y,z,w",
                 "units": "source coordinate units", "mean_geometry_metrics": "matched pairs only",
                 "undefined_precision_recall_f1": 0, "export_ratios": "0 to 1"}
        archive = BytesIO()
        with ZipFile(archive, "w") as bundle:
            for filename, table in [("lidar_scores.csv", summary), ("box_comparisons.csv", details), ("class_scores.csv", classes)]:
                bundle.writestr(filename, csv_bytes(table))
            bundle.writestr("evaluation_rules.json", json.dumps(rules, indent=2))
        st.download_button("Download LiDAR evaluation (.zip)", archive.getvalue(), "lidar_evaluation.zip", "application/zip", type="primary")
        st.caption("Contains annotator scores, per-class scores, individual box comparisons, and the selected evaluation settings.")
    with st.expander("LiDAR scoring rules and supported format"):
        st.markdown("""
        **Precision = matched / (matched + extra)** · **Recall = matched / (matched + missed)**

        **F1 = 2 × matched / (2 × matched + extra + missed)**. Undefined scores are zero.
        This is detection F1 at the selected threshold, rather than classification macro F1 or benchmark AP/mAP.

        Matching is within the same frame and class. Object IDs do not need to agree across exports.
        Each cuboid can match at most once. Roll, pitch, and yaw are retained when intersecting box volumes.
        Matched pairs include 3D IoU, center distance, mean absolute dimension error, and rotation difference in degrees.
        Low-overlap and wrong-class boxes remain unmatched.

        Supported JSON: a top-level `labels` array with `file_id`, `label_id`, `label_category_id`,
        `label_type: "3d_bbox"`, and `three_d_bbox` containing `cx`, `cy`, `cz`, `l`, `w`, `h`,
        plus a quaternion object with `x`, `y`, `z`, `w`. Positive dimensions and finite geometry are required.
        Select a single stage, sensor, and label set where multiple exist. One submitted file represents one annotator;
        embedded user and interpolation metadata are not used to assign authorship.

        Empty `labels` arrays are allowed for a submission, giving zero recall when ground truth has boxes.
        Empty ground truth is not supported. Every export must cover the intended shared assignment:
        omitted ground-truth frames count as missed work. Frame coverage cannot be inferred for frames with no boxes.
        Point-level segmentation, track identity scoring, and raw point-cloud rendering are not included.
        """)
