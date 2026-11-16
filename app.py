"""Run with: python -m streamlit run app.py"""

from pathlib import Path
from io import BytesIO
from zipfile import ZipFile

import pandas as pd
import plotly.express as px
import streamlit as st

from evaluation import ValidationError, csv_bytes, evaluate, prepare, read_table

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Annotation Lab · Evaluation", page_icon="🔎", layout="wide")
st.markdown("""<style>
    .block-container {padding-top:2.4rem; max-width:1400px;}
    [data-testid="stMetric"] {background:white; border:1px solid #e2e8f0;
        border-radius:12px; padding:18px;}
    h1 {letter-spacing:-1.4px;}
    [data-testid="stSidebar"] {border-right:1px solid #e2e8f0;}
</style>""", unsafe_allow_html=True)


def mapping(frame, fields, prefix):
    result = {}
    for field, label, aliases in fields:
        options = list(frame.columns)
        default = next((i for i, col in enumerate(options) if str(col).lower() in aliases), 0)
        result[field] = st.selectbox(label, options, index=default, key=f"{prefix}_{field}")
    return result


def percent_columns(names):
    return {name: st.column_config.NumberColumn(name.replace("_", " ").title(), format="%.1f%%")
            for name in names}


with st.sidebar:
    st.markdown("### ◈ Annotation Lab")
    st.caption("QUALITY EVALUATION WORKSPACE")
    st.divider()
    mode = st.radio("Data source", ["Sample project", "Upload your data"])
    st.divider()
    st.markdown("**Evaluation rules**")
    st.caption("One label per item. Every ground-truth item is expected from each annotator.")
    ignore_case = st.checkbox("Ignore label capitalization", value=False)
    st.caption("Leading and trailing spaces are always trimmed. Item IDs and annotator names remain case-sensitive.")
    st.divider()
    st.markdown("**Templates**")
    for filename, label in [("ground_truth.csv", "Ground truth CSV"), ("annotations.csv", "Annotations CSV")]:
        st.download_button(label, (ROOT / "samples" / filename).read_bytes(), filename, "text/csv", width="stretch")
    st.caption("Files are processed in memory. This MVP does not save evaluation history.")

st.caption("WORKSPACE  /  ANNOTATION QUALITY")
st.title("Every annotation, measured.")
st.write("Compare annotators against ground truth. Find errors, track coverage, and export the evidence.")

try:
    if mode == "Sample project":
        st.info("Sample project · Customer sentiment · 12 ground-truth items · 3 annotators")
        raw_truth = read_table((ROOT / "samples/ground_truth.csv").read_bytes(), "ground_truth.csv")
        raw_annotations = read_table((ROOT / "samples/annotations.csv").read_bytes(), "annotations.csv")
        truth_map = {"item_id": "item_id", "label": "label"}
        annotation_map = {"item_id": "item_id", "annotator": "annotator", "label": "label"}
    else:
        st.subheader("1. Add your files")
        left, right = st.columns(2)
        with left:
            gt_file = st.file_uploader("Ground truth", type=["csv", "xlsx", "json"], help="A unique item ID and one expected label per row. XLSX: first worksheet.")
        with right:
            ann_file = st.file_uploader("Annotator submissions", type=["csv", "xlsx", "json"], help="Combine annotators into one file with item ID, annotator, and label columns.")
        if not gt_file or not ann_file:
            st.info("Upload both files to continue, or select Sample project in the sidebar.")
            st.code("Ground truth: item_id,label\nAnnotations:  item_id,annotator,label", language="text")
            st.stop()
        raw_truth = read_table(gt_file.getvalue(), gt_file.name)
        raw_annotations = read_table(ann_file.getvalue(), ann_file.name)
        st.subheader("2. Match your columns")
        left, right = st.columns(2)
        fields = [("item_id", "Item ID column", ["item_id", "id", "task_id"]),
                  ("label", "Label column", ["label", "class", "annotation", "ground_truth"])]
        with left:
            st.markdown("**Ground truth**")
            truth_map = mapping(raw_truth, fields, "gt")
        with right:
            st.markdown("**Annotations**")
            annotation_map = mapping(raw_annotations, fields + [
                ("annotator", "Annotator column", ["annotator", "annotator_id", "worker_id", "name"])], "ann")
        with st.expander("Preview uploaded data"):
            st.dataframe(raw_truth.head(10), hide_index=True, width="stretch")
            st.dataframe(raw_annotations.head(10), hide_index=True, width="stretch")

    truth = prepare(raw_truth, truth_map, "Ground truth", ignore_case)
    annotations = prepare(raw_annotations, annotation_map, "Annotations", ignore_case)
    summary, details, by_label = evaluate(truth, annotations)
except ValidationError as exc:
    st.error(str(exc))
    st.stop()

st.divider()
cols = st.columns(4)
cols[0].metric("Annotators evaluated", len(summary))
cols[1].metric("Mean accuracy", f"{summary.accuracy.mean():.1%}")
cols[2].metric("Mean macro F1", f"{summary.f1.mean():.1%}")
cols[3].metric("Mean coverage", f"{summary.coverage.mean():.1%}")
st.caption("Means give each annotator equal weight. Accuracy includes missing items; coverage measures completion.")
if summary.unknown_ids.sum():
    st.warning(f"{summary.unknown_ids.sum()} submissions reference IDs outside ground truth. They are excluded from scores and listed in item review.")
if summary.missing.sum():
    st.warning(f"{summary.missing.sum()} expected annotations are missing across annotators. They count against accuracy, recall, and F1.")

overview, review, metrics_tab, exports = st.tabs(["Overview", "Item review", "Label performance", "Export reports"])
with overview:
    st.subheader("Annotator performance")
    st.caption("Ordered by macro F1, then accuracy. Review coverage alongside quality.")
    display = summary.copy()
    ratios = ["accuracy", "precision", "recall", "f1", "coverage"]
    display[ratios] *= 100
    st.dataframe(display, hide_index=True, width="stretch", column_config=percent_columns(ratios))
    chart = summary.melt(id_vars="annotator", value_vars=["accuracy", "f1", "coverage"], var_name="Metric", value_name="Score")
    chart["Metric"] = chart.Metric.map({"accuracy": "Accuracy", "f1": "Macro F1", "coverage": "Coverage"})
    fig = px.bar(chart, x="annotator", y="Score", color="Metric", barmode="group",
                 color_discrete_sequence=["#167D7F", "#5064BA", "#B7C9DC"],
                 labels={"annotator": "Annotator"}, template="plotly_white")
    fig.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 1.05], legend_title_text="", margin=dict(t=20, b=20))
    st.plotly_chart(fig, width="stretch")

with review:
    st.subheader("Inspect the evidence")
    left, middle, right = st.columns([1, 1, 2])
    who = left.multiselect("Annotators", summary.annotator.tolist(), default=summary.annotator.tolist())
    statuses = middle.multiselect("Status", ["correct", "incorrect", "missing", "unknown ID"], default=["incorrect", "missing", "unknown ID"])
    search = right.text_input("Find an item ID", placeholder="Search item IDs…")
    filtered = details[details.annotator.isin(who) & details.status.isin(statuses)]
    if search:
        filtered = filtered[filtered.item_id.str.contains(search, case=False, regex=False)]
    # Include optional source text without affecting evaluation or treating it as HTML.
    text_col = next((col for col in raw_truth.columns if str(col).lower() in ["text", "content", "description"]
                     and col not in truth_map.values()), None)
    if text_col is not None:
        context = pd.DataFrame({"item_id": raw_truth[truth_map["item_id"]].astype(str).str.strip(),
                                "context": raw_truth[text_col].astype(str)})
        filtered = filtered.merge(context, on="item_id", how="left")
    st.caption(f"{len(filtered):,} matching rows")
    st.dataframe(filtered.fillna("—"), hide_index=True, width="stretch")
    st.download_button("Download filtered review", csv_bytes(filtered), "filtered_review.csv", "text/csv")

with metrics_tab:
    st.subheader("Where labels get confused")
    chosen = st.selectbox("Annotator", summary.annotator.tolist())
    label_display = by_label[by_label.annotator.eq(chosen)].drop(columns="annotator").copy()
    label_display[["precision", "recall", "f1"]] *= 100
    st.dataframe(label_display, hide_index=True, width="stretch",
                 column_config=percent_columns(["precision", "recall", "f1"]))
    rows = details[details.annotator.eq(chosen) & details.status.ne("unknown ID")].copy()
    # Prefix real predictions so a label literally called 'Missing' stays distinct.
    rows["prediction"] = rows.actual.map(lambda value: "Missing annotation" if pd.isna(value) else f"Label: {value}")
    matrix = pd.crosstab(rows.expected, rows.prediction)
    heatmap = px.imshow(matrix, text_auto=True, color_continuous_scale="Teal", aspect="auto",
                        labels={"x": "Submitted label", "y": "Ground truth", "color": "Items"})
    heatmap.update_layout(margin=dict(t=20, b=20))
    st.plotly_chart(heatmap, width="stretch")

with exports:
    st.subheader("Take your results with you")
    st.write("Download scores, per-label metrics, and every comparison row in one ZIP file.")
    archive = BytesIO()
    with ZipFile(archive, "w") as bundle:
        for name, frame in [("annotator_scores.csv", summary), ("item_comparisons.csv", details), ("label_metrics.csv", by_label)]:
            bundle.writestr(name, csv_bytes(frame))
        bundle.writestr("evaluation_rules.txt", (
            "Single-label evaluation. IDs and names are trimmed and case-sensitive.\n"
            f"Label case ignored: {ignore_case}. Label whitespace is trimmed.\n"
            "Every ground-truth item is expected from every annotator.\n"
            "Accuracy = correct / ground-truth items. Coverage = known submitted items / ground-truth items.\n"
            "Missing annotations count as errors and false negatives, with no positive prediction.\n"
            "Unknown IDs are excluded from metrics and retained in item comparisons.\n"
            "Precision, recall, and F1 are macro averages over truth labels and predicted labels on known items.\n"
            "Undefined label metrics are zero. Support is the ground-truth count.\n"
            "Dashboard means give every annotator equal weight. CSV ratios use the range 0 to 1.\n"
            "Potential spreadsheet formulas in text cells are prefixed with an apostrophe on export.\n"
        ))
    st.download_button("Download evaluation report (.zip)", archive.getvalue(), "annotation_evaluation.zip", "application/zip", type="primary")
    st.download_button("Download annotator scores (.csv)", csv_bytes(summary), "annotator_scores.csv", "text/csv")

with st.expander("How scores are calculated"):
    st.markdown("""
    - **Accuracy:** correct labels divided by all ground-truth items.
    - **Coverage:** submitted labels on known IDs divided by all ground-truth items.
    - **Precision:** how often a submitted label is correct.
    - **Recall:** how often an expected label is found, including missed work.
    - **Macro F1:** the mean of per-label F1 scores, giving each label equal weight.

    Precision, recall, and F1 use all ground-truth labels plus any extra labels predicted on known items.
    Undefined label metrics are zero. Unknown IDs are excluded from scores. Duplicate IDs in ground truth,
    duplicate annotator/item pairs, and blank required values must be fixed before evaluation.

    This version evaluates **one class label per item**. Bounding boxes, text spans, multi-label tasks,
    free-text answers, and transcription similarity require different scoring methods.
    Annotators with no rows in the submission file cannot be detected.
    """)

