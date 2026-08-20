"""Streamlit front end for the attendance sheet extractor.

Flow: upload -> detect structure -> confirm the date headers -> extract ->
review and download. Nothing is extracted until the headers are confirmed,
because the dates are handwritten and only you can read them reliably.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import shutil
from io import BytesIO

import cv2
import numpy as np
import streamlit as st

import attendance_core as ac

st.set_page_config(page_title="Attendance sheet extractor",
                   page_icon="📋", layout="wide")


# --------------------------------------------------------------------------- #
# Cached heavy work
# --------------------------------------------------------------------------- #
# cache_resource, not cache_data: PageGrid holds large arrays and fitted
# polynomials that we do not want serialised on every rerun.
@st.cache_resource(show_spinner=False)
def analyse_cached(pdf_bytes: bytes, _progress=None):
    return ac.analyse(pdf_bytes, progress=_progress)


@st.cache_resource(show_spinner=False)
def extract_cached(pdf_bytes: bytes, with_names: bool, _sheet, _progress=None):
    return ac.extract(_sheet, with_names=with_names, progress=_progress)


@st.cache_resource(show_spinner=False)
def annotated_pdf_cached(pdf_bytes: bytes, _sheet, _records):
    return ac.annotate_pdf(pdf_bytes, _sheet, _records)


def reset_downstream():
    """A new upload or a header edit invalidates everything after it."""
    for k in ("records", "headers_confirmed", "date_labels", "sheet"):
        st.session_state.pop(k, None)


def tesseract_ready() -> bool:
    return shutil.which("tesseract") is not None


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title("📋 Attendance sheet extractor")
st.caption(
    "Reads a scanned attendance sheet, decides present/absent for every cell, and "
    "returns a CSV plus a copy of the PDF with a red box on each present cell. "
    "A signature counts as present; an empty cell, a vertical stroke or a "
    "struck-out cell counts as absent."
)

if not tesseract_ready():
    st.error(
        "**Tesseract is not installed.** pytesseract is only a wrapper around the "
        "Tesseract binary, which has to be installed separately.\n\n"
        "- macOS: `brew install tesseract`\n"
        "- Ubuntu/Debian: `sudo apt-get install -y tesseract-ocr`\n"
        "- Windows: https://github.com/UB-Mannheim/tesseract/wiki\n"
        "- Streamlit Community Cloud: add a `packages.txt` containing `tesseract-ocr`"
    )
    st.stop()

with st.sidebar:
    st.header("Settings")
    with_names = st.checkbox(
        "Read student names", value=True,
        help="Name OCR is the slow part. Turn it off for a quick pass that still "
             "reads roll numbers and all attendance marks.")
    st.divider()
    st.subheader("Sheet layout")
    st.caption(
        "Expecting **Sno | Roll No | Name | some mark columns**. The column count, "
        "the column pitch and which columns carry a date are all measured from the "
        "sheet, so layouts may differ between files.")
    st.divider()
    st.caption(f"Render resolution: {ac.DPI} DPI · the present/absent threshold is "
               f"chosen per sheet · review below {ac.REVIEW_CONF} confidence")


# --------------------------------------------------------------------------- #
# Step 1 - upload
# --------------------------------------------------------------------------- #
st.subheader("1. Upload the scanned sheet")
upload = st.file_uploader("Attendance PDF", type=["pdf"],
                          help="A scanned or photographed attendance sheet.")

if upload is None:
    st.info("Upload a PDF to begin.")
    st.stop()

pdf_bytes = upload.getvalue()
if st.session_state.get("file_sig") != (upload.name, len(pdf_bytes)):
    st.session_state["file_sig"] = (upload.name, len(pdf_bytes))
    reset_downstream()

st.success(f"Loaded **{upload.name}** · {len(pdf_bytes) / 1e6:.1f} MB")


# --------------------------------------------------------------------------- #
# Step 2 - structure
# --------------------------------------------------------------------------- #
st.subheader("2. Detected structure")

if "sheet" not in st.session_state:
    bar = st.progress(0.0, "Reading pages")
    try:
        st.session_state["sheet"] = analyse_cached(
            pdf_bytes, _progress=lambda f, m: bar.progress(min(f, 1.0), m))
    except Exception as e:                       # noqa: BLE001
        bar.empty()
        st.error(f"Could not read the table structure: {e}")
        st.stop()
    bar.empty()

sheet: ac.Sheet = st.session_state["sheet"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Pages", len(sheet.grids))
c2.metric("Student rows", sheet.n_rows)
c3.metric("Dated columns", len(sheet.live))
c4.metric("Cells to classify", sheet.n_rows * len(sheet.live))

with st.expander("Per-page detail"):
    st.dataframe(
        {"Page": [i + 1 for i in range(len(sheet.grids))],
         "Rows": [g.n_rows for g in sheet.grids],
         "Column pitch (px)": [round(g.pitch) for g in sheet.grids],
         "Mark columns": [g.n_mark for g in sheet.grids],
         "Rules detected": [f"{len(g.col_idx)} of {g.n_bounds}" for g in sheet.grids]},
        hide_index=True, width="stretch")
    st.caption(
        "Rules that were not detected - faint print, a shadow, the page edge - are "
        "reconstructed by fitting the regular progression the printed rules sit on, "
        "so fewer detected than total is normal and not a problem.")
    if sheet.page_errors:
        st.warning(
            "Skipped " + ", ".join(f"page {p} ({why})" for p, why in sheet.page_errors)
            + ". Any students on those pages are missing from the results.")

if sheet.outside_writing:
    pages = ", ".join(str(p) for p, _, _ in sheet.outside_writing)
    st.warning(
        f"**Handwritten entries outside the table on page {pages}.** "
        "Some names appear to have been written in by hand below or beside the "
        "printed grid. Those are **not** read - handwriting there cannot be "
        "transcribed reliably, so they are left for you to enter manually. "
        "Everything inside the printed table on those pages is processed normally.")

if not sheet.live:
    st.error("No dated columns found. The header row appears to be blank.")
    st.stop()


# --------------------------------------------------------------------------- #
# Step 3 - confirm the date headers
# --------------------------------------------------------------------------- #
st.subheader("3. Confirm the date columns")
st.caption(
    "The dates are handwritten, and handwriting OCR is not trustworthy, so please "
    "read them off the scan and type them in. These become the CSV column headers.")

default_labels = st.session_state.get(
    "date_labels", [f"Date {i + 1}" for i in range(len(sheet.live))])

cols = st.columns(len(sheet.live))
labels = []
for k, (cj, col) in enumerate(zip(sheet.live, cols)):
    with col:
        st.image(sheet.grids[0].header_crop(cj),
                 caption=f"Column {k + 1} header (page 1)", width="stretch")
        labels.append(st.text_input(f"Label for column {k + 1}",
                                    value=default_labels[k], key=f"lbl_{k}"))

peek = st.checkbox("Show this header on every page",
                   help="Useful when the date on page 1 is unclear.")
if peek:
    for k, cj in enumerate(sheet.live):
        st.write(f"**Column {k + 1}** — {labels[k] or '(unnamed)'}")
        row = st.columns(len(sheet.grids))
        for p, (g, col) in enumerate(zip(sheet.grids, row)):
            col.image(g.header_crop(cj), caption=f"p{p + 1}", width="stretch")

clean = [l.strip() for l in labels]
problems = []
if any(not l for l in clean):
    problems.append("every column needs a label")
if len(set(clean)) != len(clean):
    problems.append("labels must be unique")

if problems:
    st.warning("Before extracting: " + "; ".join(problems) + ".")
    st.stop()

if st.session_state.get("date_labels") != clean:
    st.session_state["date_labels"] = clean
    st.session_state.pop("records", None)         # relabel -> rebuild the tables

if st.button("Confirm headers and extract attendance", type="primary",
             width="stretch"):
    st.session_state["headers_confirmed"] = True

if not st.session_state.get("headers_confirmed"):
    st.info("Check the labels above, then press **Confirm headers and extract "
            "attendance**.")
    st.stop()


# --------------------------------------------------------------------------- #
# Step 4 - extract
# --------------------------------------------------------------------------- #
if "records" not in st.session_state:
    bar = st.progress(0.0, "Starting")
    st.session_state["records"] = extract_cached(
        pdf_bytes, with_names, sheet,
        _progress=lambda f, m: bar.progress(min(f, 1.0), m))
    bar.empty()

records = st.session_state["records"]
date_labels = st.session_state["date_labels"]
columns, rows, review_columns, review_rows = ac.build_tables(records, date_labels)
cal = ac.calibration(records, sheet.threshold)

st.subheader("4. Results")

total_cells = len(rows) * len(date_labels)
present = sum(r["Present"] for r in rows)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Students", len(rows))
m2.metric("Present marks", present, f"{present / total_cells:.0%} of cells")
m3.metric("Flagged for review", len(review_rows))
m4.metric("Missing names", sum(1 for r in rows if not r["Name"]))

tab_table, tab_student, tab_verify, tab_dl = st.tabs(
    ["📊 Table", "🔎 Student report", "🔍 Verification", "⬇️ Downloads"])


# --- table ----------------------------------------------------------------- #
with tab_table:
    show_conf = st.checkbox("Show per-cell confidence columns", value=False)
    display_cols = columns if show_conf else [c for c in columns
                                              if not c.startswith("conf ")]
    st.dataframe(ac.to_dataframe(columns, rows)[display_cols],
                 hide_index=True, width="stretch", height=520)

    st.markdown("**Attendance by date**")
    st.bar_chart({d: [sum(1 for r in rows if r[d] == "P")] for d in date_labels})


# --- student report -------------------------------------------------------- #
with tab_student:
    st.markdown("Enter a roll number to see which days that student attended.")
    query = st.text_input("Roll number or name", placeholder="e.g. BT25D029",
                          key="student_query")

    if query:
        rep = ac.student_report(rows, date_labels, query)
        if rep is None:
            st.warning(f"No student matching **{query}**.")
        else:
            head = st.columns([2, 2, 1, 1])
            head[0].metric("Roll number", rep["roll"] or "—")
            head[1].metric("Name", rep["name"] or "(not read)")
            head[2].metric("Days present", f"{rep['present']} / {rep['total']}")
            head[3].metric("Attendance", f"{rep['percent']:.0f}%")

            st.progress(rep["percent"] / 100.0)

            a, b = st.columns(2)
            with a:
                st.markdown("**Attended**")
                if rep["attended"]:
                    for d in rep["attended"]:
                        st.markdown(f"- ✅ {d}")
                else:
                    st.caption("None.")
            with b:
                st.markdown("**Absent**")
                if rep["missed"]:
                    for d in rep["missed"]:
                        st.markdown(f"- ❌ {d}")
                else:
                    st.caption("None — full attendance.")

            if rep["low_conf"]:
                st.warning(
                    "Low confidence on: " + ", ".join(rep["low_conf"])
                    + ". Worth checking these against the scan in the Verification tab.")

            if rep["alternatives"]:
                st.caption("Other partial matches: "
                           + ", ".join(f"{r['Roll No']} ({r['Name']})"
                                       for r in rep["alternatives"]))

    st.divider()
    st.markdown("**Whole-cohort summary**")
    perfect = sum(1 for r in rows if r["Present"] == len(date_labels))
    never = sum(1 for r in rows if r["Present"] == 0)
    s1, s2, s3 = st.columns(3)
    s1.metric("Attended every day", perfect)
    s2.metric("Attended none", never)
    s3.metric("Mean attendance",
              f"{np.mean([r['Present'] for r in rows]) / len(date_labels):.0%}")

    st.dataframe(
        ac.to_dataframe(columns, sorted(rows, key=lambda r: -r["Present"]))
        [["Sno", "Roll No", "Name", "Present", "Absent"]],
        hide_index=True, width="stretch", height=320)


# --- verification ---------------------------------------------------------- #
with tab_verify:
    st.markdown("#### Annotated page preview")
    page_no = st.number_input("Page", 1, len(sheet.grids), 1, key="prev_page")
    ann_bytes, n_boxes = annotated_pdf_cached(pdf_bytes, sheet, records)
    doc = ac.open_pdf(ann_bytes)
    pm = doc[int(page_no) - 1].get_pixmap(dpi=140)
    prev = np.frombuffer(pm.samples, np.uint8).reshape(pm.height, pm.width, pm.n)
    st.image(prev, caption=f"Page {page_no} — red boxes mark cells read as present",
             width="stretch")
    doc.close()

    st.divider()
    st.markdown("#### Calibration")
    st.caption(
        "The threshold should sit in an empty region rather than in the middle of "
        "the data. Cells piling up against the line mean it is making delicate "
        "calls; a clear gap means the two classes are genuinely separated.")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Threshold used", f"{cal['threshold']:.4f}",
              "chosen from this sheet")
    k2.metric("Highest absent", f"{cal['highest_absent']:.4f}")
    k3.metric("Lowest present", f"{cal['lowest_present']:.4f}")
    k4.metric("Gap", f"{cal['gap_ratio']:.1f}×",
              "healthy" if cal["gap_ratio"] > 1.5 else "narrow — check results")
    if cal["gap_ratio"] <= 1.5:
        st.warning(
            "The two classes are not cleanly separated on this sheet, so the "
            "threshold is making close calls. Treat the output as provisional and "
            "work through the flagged cells below.")

    hist, _ = np.histogram(cal["marks"], bins=np.linspace(0, 0.3, 60))
    st.bar_chart({"cells": hist}, height=200)
    st.caption(f"Ink fraction per cell, 0 to 0.3. Threshold at {cal['threshold']:.4f}.")

    st.divider()
    st.markdown(f"#### Cells flagged for review ({len(review_rows)})")
    if not review_rows:
        st.success("Nothing flagged — every cell was decided with confidence.")
    else:
        st.caption("Each crop is shown as it appears on the scan, with the call the "
                   "classifier made. Correct any you disagree with in the CSV.")
        per_row = 4
        for start in range(0, len(review_rows), per_row):
            chunk = review_rows[start:start + per_row]
            for item, col in zip(chunk, st.columns(per_row)):
                rec = records[item["Sno"] - 1]
                cell = rec["cells"][date_labels.index(item["Date"])]
                x0, y0, x1, y1 = cell["box"]
                crop = sheet.grids[rec["page"]].img[y0:y1, x0:x1]
                col.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
                          width="stretch")
                col.caption(
                    f"**#{item['Sno']} {item['Roll No']}** · {item['Date']}  \n"
                    f"read as **{item['Called']}** ({item['Reason']}), "
                    f"confidence {item['Confidence']:.2f}")

        st.dataframe(ac.to_dataframe(review_columns, review_rows),
                     hide_index=True, width="stretch")


# --- downloads ------------------------------------------------------------- #
with tab_dl:
    st.markdown("#### Download the results")
    stem = upload.name.rsplit(".", 1)[0]

    csv_main = ac.to_csv_bytes(columns, rows)
    csv_review = ac.to_csv_bytes(review_columns, review_rows)
    ann_bytes, n_boxes = annotated_pdf_cached(pdf_bytes, sheet, records)

    d1, d2, d3 = st.columns(3)
    with d1:
        st.download_button("⬇️ attendance.csv", csv_main,
                           file_name=f"{stem}_attendance.csv", mime="text/csv",
                           width="stretch", type="primary")
        st.caption(f"{len(rows)} rows · roll no, name, P/N per date, totals, "
                   "per-cell confidence")
    with d2:
        st.download_button("⬇️ attendance_review.csv", csv_review,
                           file_name=f"{stem}_attendance_review.csv", mime="text/csv",
                           width="stretch",
                           disabled=not review_rows)
        st.caption(f"{len(review_rows)} cells worth a human glance"
                   if review_rows else "Nothing flagged")
    with d3:
        st.download_button("⬇️ attendance_annotated.pdf", ann_bytes,
                           file_name=f"{stem}_annotated.pdf", mime="application/pdf",
                           width="stretch")
        st.caption(f"{n_boxes} red boxes on the original scan")

    st.divider()
    if st.button("Re-run extraction", help="Clears the cache and processes again"):
        st.cache_resource.clear()
        reset_downstream()
        st.rerun()
