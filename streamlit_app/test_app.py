"""Headless end-to-end test of the Streamlit app."""
import streamlit as st
from streamlit.testing.v1 import AppTest

PDF = "/tmp/two.pdf"
DATA = open(PDF, "rb").read()

class FakeUpload:
    name = "two.pdf"
    def getvalue(self): return DATA

# st.file_uploader has no AppTest widget driver, so stub it out.
st.file_uploader = lambda *a, **k: FakeUpload()

at = AppTest.from_file("app.py", default_timeout=300)
at.run()
assert not at.exception, at.exception
print("run 1 (pre-confirm):")
print("  metrics:", [(m.label, m.value) for m in at.metric])
print("  text inputs:", [(t.label, t.value) for t in at.text_input])
print("  buttons:", [b.label for b in at.button])

# fill in the date labels, then confirm
for i, t in enumerate(at.text_input):
    if t.label.startswith("Label for column"):
        t.set_value(f"D{i+1}")
at.run()
assert not at.exception, at.exception

btn = [b for b in at.button if "Confirm headers" in b.label]
assert btn, "confirm button missing"
btn[0].click().run()
assert not at.exception, at.exception

print("run 2 (post-extract):")
print("  metrics:", [(m.label, m.value) for m in at.metric])
print("  tabs:", len(at.tabs))
print("  download buttons:", [d.label for d in at.download_button])
# AppTest's DownloadButton proxy does not expose the payload, so verify the
# exported bytes by calling the export functions the app calls.
import attendance_core as ac
sheet = ac.analyse(DATA)
recs = ac.extract(sheet, with_names=True)
labels = [f"D{i+1}" for i in range(len(sheet.live))]
cols, rows, rcols, rrows = ac.build_tables(recs, labels)
csv_main = ac.to_csv_bytes(cols, rows)
csv_rev = ac.to_csv_bytes(rcols, rrows)
ann, nbox = ac.annotate_pdf(DATA, sheet, recs)
assert csv_main.count(b"\n") == len(rows) + 1
assert csv_rev.count(b"\n") == len(rrows) + 1
assert ann.startswith(b"%PDF") and len(ann) > 10000
assert len(at.download_button) == 3
print(f"    attendance.csv        {len(csv_main)} bytes, {len(rows)} rows")
print(f"    attendance_review.csv {len(csv_rev)} bytes, {len(rrows)} rows")
print(f"    annotated.pdf         {len(ann)} bytes, {nbox} red boxes")
rep = ac.student_report(rows, labels, rows[9]["Roll No"])
assert rep and rep["total"] == len(labels)
print(f"    student lookup {rep['roll']}: {rep['present']}/{rep['total']} = {rep['percent']:.0f}%")
print("OK - no exceptions")
