# Attendance sheet extractor — Streamlit app

Upload a scanned attendance sheet, confirm the date columns, get a CSV and an
annotated PDF back.

## Files

| file | purpose |
|---|---|
| `app.py` | Streamlit UI |
| `attendance_core.py` | the extraction pipeline, UI-free and importable |
| `requirements.txt` | Python packages |
| `../packages.txt` | system packages (Tesseract), **must live in the repo root** |
| `test_app.py` | headless end-to-end test |

## Run locally

Tesseract is a system binary, not a Python package — `pip install pytesseract`
alone is not enough.

```bash
brew install tesseract                  # macOS
# sudo apt-get install -y tesseract-ocr # Ubuntu/Debian

pip install -r requirements.txt
streamlit run app.py
```

The app checks for the Tesseract binary on startup and tells you what to install
if it is missing.

## Deploy to Streamlit Community Cloud

Push the **whole repository** to GitHub and point Streamlit Cloud at
`streamlit_app/app.py`.

Community Cloud resolves the two dependency files from different places:

- `requirements.txt` — searched for in the entrypoint's directory first, then
  the repo root, so `streamlit_app/requirements.txt` is found.
- `packages.txt` — detected **only in the repo root**. It therefore sits at the
  top level, next to `streamlit_app/`. It installs Tesseract on the container;
  without it the app stops on startup with the "Tesseract is not installed"
  error.

Uploads are capped at 200 MB by default. To raise it, add `.streamlit/config.toml`:

```toml
[server]
maxUploadSize = 400
```

## How it works

The scans are photographs: pages are rotated *and bowed*, and the absent marks
are single pen strokes drawn continuously down twenty or more rows, so a naive
"is there ink in this cell?" test marks whole columns present. Three ideas do
most of the work:

1. **Ruled lines are fitted as curves**, not assumed straight, so cell boundaries
   follow the page warp. A straight-line model leaves boundaries slicing through
   the text near the page edges.
2. **Mark cells hold no printed content**, so once the fitted ruling is erased,
   every remaining dark pixel there is handwriting. This is pen-colour
   independent — a blue-channel test silently drops black and grey signatures.
3. **Cells are classified per connected component.** A cell often holds two
   *different* absent strokes, its own and the neighbouring column's bleeding
   past the rule; their union bounding box looks convincingly like a signature,
   while each component alone is plainly a thin vertical line.

A component that is narrow, much taller than wide, and runs most of the cell
height is discounted as a ruled stroke. The ink left over decides the label.

## Reading the output

`attendance.csv` — one row per student: roll number, name, `P`/`N` per date,
totals, and a confidence per cell.

`attendance_review.csv` — only the cells worth a human glance.

`<name>_annotated.pdf` — the original scan with a red box on every cell read as
present, drawn as vector graphics so the scan underneath is untouched.

**Check the calibration figure in the Verification tab.** It reports the highest
"absent" ink fraction and the lowest "present" one. A wide gap means the two
classes are genuinely separated and the threshold is not making delicate calls.
If the gap approaches 1×, treat the output as provisional and check the flagged
cells.

## Adapting it

- **Different date columns.** Nothing to change — how many are in use is
  detected from ink in the header cells, and you type the labels in the app.
- **Different number of students or pages.** Nothing to change; rows and pages
  are detected.
- **Different sheet layout.** `N_MARK_COLS` in `attendance_core.py` sets how many
  mark columns follow the Name column.
- **Different roll-number format.** `norm_roll()` encodes the `BT<2 digits><D|S><3
  digits>` pattern and the glyph repairs that go with it.
- **Classifier too strict or too loose.** `T_MARKAREA` is the threshold. Set it
  in the middle of whatever empty band the calibration histogram shows.

## Testing

```bash
python test_app.py
```

Stubs the file uploader, drives the app through header confirmation and
extraction via Streamlit's `AppTest`, and checks that all three exports are
produced and that the student lookup works.

## Known limitations

- Handwritten dates are not OCR'd — you type them. Handwriting OCR is not
  reliable enough to trust for column headers.
- A few names fail to read where a signature overlays the printed text. Roll
  numbers are recovered in these cases via a second pass, so rows stay
  identifiable.
- `Sno` comes from row order, not OCR; the OCR'd value is kept in the
  `Sno OCR check` column purely as a cross-check.
