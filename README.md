# Annotation Lab

A local Streamlit MVP for evaluating annotators against ground truth. Supports single-label classification: one label for each item and annotator.

## Start

Requires Python 3.11 or newer. From this folder in PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open http://127.0.0.1:8501. Start with **Sample project**, then select **Upload your data**. Stop the server with Ctrl+C. An environment already prepared in this workspace can be started with only the last command.

Alternatively, run `.\run.ps1` from PowerShell after installing the dependencies. The helper starts the app from its own folder.

## Files

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

## Evaluation

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

This MVP does not evaluate bounding boxes, segmentation, text spans, multi-label selections, or free-text similarity.

## Verification

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The tests cover scoring, missing/unknown items, incorrect predictions, independent annotators, validation, case handling, file formats, and export escaping. Streamlit AppTest also checks dashboard rendering, sample metrics, review filtering, annotator selection, and the upload screen.
