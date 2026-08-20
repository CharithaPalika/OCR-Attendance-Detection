"""Attendance sheet extraction - core pipeline.

UI-free, so the same code backs the Streamlit app, the notebook and any script.

The scans are photographs: pages are rotated *and bowed*, so every ruled line is
fitted as a curve rather than assumed straight. Three ideas do most of the work:

1. Ruled lines are fitted as quadratics, so cell boundaries follow the page warp.
2. Mark cells hold no printed content, so once the fitted ruling is erased every
   remaining dark pixel there is handwriting - no reliance on pen colour, which
   matters because the sheet mixes blue, black and grey pens.
3. Cells are classified per connected component. A component that is narrow, much
   taller than wide, and runs most of the cell height is a ruled "absent" stroke
   and is discounted; the ink left over decides the label.
"""

from __future__ import annotations

import io
import math
import re
from dataclasses import dataclass, field

import cv2
import fitz  # PyMuPDF
import numpy as np
import pytesseract

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
DPI = 300

# Sheet layout: Sno | Roll No | First Name | N mark columns
# The layout is measured per sheet, not assumed: how many mark columns follow the
# Name column, and how wide they are, both vary between sheets.
MAX_MARK_COLS = 20                        # sanity ceiling only

BLUE_T, GRAY_T = 25, 190                  # blue-pen isolation
LIVE_T = 0.02                             # header ink marking a column as in use

# Writing entered outside the ruled table. Both must be exceeded: blob count alone
# fires on scanner specks, ink fraction alone on a smudge.
OUTSIDE_BLOBS = 15
OUTSIDE_INK = 0.0015

# Classification thresholds. See `calibration()` - the gap between the classes is
# roughly sevenfold, so these sit in empty space rather than doing delicate work.
MIN_COMP = 0.0015                         # ignore components smaller than this
T_WFRAC = 0.30                            # component width / cell width
T_ASPECT = 2.0                            # component height / width
T_HFRAC = 0.35                            # component height / cell height
T_MARKAREA = 0.035                        # surviving ink needed to call "present"
CONF_RATIO = 3.0                          # ratio at which confidence saturates
REVIEW_CONF = 0.75                        # below this, flag for manual review


# --------------------------------------------------------------------------- #
# Rendering and masks
# --------------------------------------------------------------------------- #
def open_pdf(data: bytes) -> fitz.Document:
    return fitz.open(stream=data, filetype="pdf")


def render(doc: fitz.Document, pno: int, dpi: int = DPI):
    """Render a page. Returns the BGR image and the pixels-per-PDF-point scale."""
    pm = doc[pno].get_pixmap(dpi=dpi)
    a = np.frombuffer(pm.samples, np.uint8).reshape(pm.height, pm.width, pm.n)
    return cv2.cvtColor(a, cv2.COLOR_RGB2BGR), pm.width / doc[pno].rect.width


def binarize(img: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY_INV, 31, 15)


def make_masks(img: np.ndarray):
    """ink = blue pen; printed = black print and ruling, with the blue removed."""
    b, g, r = cv2.split(img.astype(np.int16))
    blue = np.clip(b - r, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ink = (blue > BLUE_T) & (gray < GRAY_T)
    ink = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_OPEN,
                           np.ones((3, 3), np.uint8)) > 0

    dark = binarize(img) > 0
    printed = dark & ~(cv2.dilate(ink.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0)
    return (ink * 255).astype(np.uint8), (printed * 255).astype(np.uint8)


def line_free_text(printed: np.ndarray, w: int, h: int) -> np.ndarray:
    """Printed mask with the ruling removed -> black text on white, for OCR."""
    hor = cv2.morphologyEx(printed, cv2.MORPH_OPEN,
                           cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, w // 40), 1)))
    ver = cv2.morphologyEx(printed, cv2.MORPH_OPEN,
                           cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, h // 40))))
    lines = cv2.dilate(cv2.bitwise_or(hor, ver), np.ones((5, 5), np.uint8))
    text = cv2.bitwise_and(printed, cv2.bitwise_not(lines))
    return cv2.bitwise_not(cv2.morphologyEx(text, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)))


# --------------------------------------------------------------------------- #
# Ruled lines, fitted as curves
# --------------------------------------------------------------------------- #
class Curve:
    """A ruled line fitted as y=f(x) (horizontal) or x=f(y) (vertical).

    Quadratic within the observed span, linear extrapolation outside it. A
    straight-line model leaves cell boundaries slicing through the text near the
    page edges, because the pages bow.
    """

    def __init__(self, t, v, deg: int = 2):
        o = np.argsort(t)
        t, v = np.asarray(t, float)[o], np.asarray(v, float)[o]
        self.t0, self.t1 = float(t[0]), float(t[-1])
        self.p = np.poly1d(np.polyfit(t, v, min(deg, max(1, len(t) - 2))))
        self.dp = self.p.deriv()
        self.mid = float(self.p((self.t0 + self.t1) / 2))

    def __call__(self, t):
        t = np.asarray(t, float)
        c = np.clip(t, self.t0, self.t1)
        return self.p(c) + self.dp(c) * (t - c)


def _profile(lab: np.ndarray, i: int, axis: int):
    """Centroid of component `i` as a function of the along-line coordinate."""
    ys, xs = np.where(lab == i)
    t, v = (xs, ys) if axis == 0 else (ys, xs)
    o = np.argsort(t, kind="stable")
    t, v = t[o], v[o]
    ut, idx = np.unique(t, return_index=True)          # np.unique needs sorted keys
    bnd = np.append(idx, len(t))
    return ut.astype(float), np.array([v[bnd[k]:bnd[k + 1]].mean()
                                       for k in range(len(ut))])


def _dedupe(curves, gap):
    curves.sort(key=lambda c: c.mid)
    out = []
    for c in curves:
        if out and c.mid - out[-1].mid < gap:
            continue
        out.append(c)
    return out


def find_hlines(bw, minfrac=0.45, kdiv=40):
    """Row rules, from the full dark mask: the hand-drawn absent strokes are
    vertical, so they cannot be mistaken for horizontal rules."""
    h, w = bw.shape
    m = cv2.morphologyEx(bw, cv2.MORPH_OPEN,
                         cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, w // kdiv), 1)))
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    out = []
    for i in range(1, n):
        if st[i][2] < w * minfrac:
            continue
        t, v = _profile(lab, i, 0)
        if len(t) >= 50:
            out.append(Curve(t, v))
    return _dedupe(out, 12)


def find_vlines(mask, ytop, ybot, minfrac=0.60):
    """Column rule candidates. Callers pass the raw binary: on some sheets the
    absent marks are ruled straight down the printed rules, so the ink-subtracted
    mask has the rules punched out of it. Hand-drawn lines that come through here
    are rejected later by the lattice fit, which they do not sit on."""
    printed = mask
    band = printed[int(ytop):int(ybot), :]
    bh = band.shape[0]
    m = cv2.morphologyEx(band, cv2.MORPH_OPEN,
                         cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, bh // 20))))
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 31)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    out = []
    for i in range(1, n):
        if st[i][3] < bh * minfrac:
            continue
        t, v = _profile(lab, i, 1)
        if len(t) >= 50:
            out.append(Curve(t + int(ytop), v))        # back to absolute y
    return _dedupe(out, 20)


# --------------------------------------------------------------------------- #
# Column lattice
# --------------------------------------------------------------------------- #
def _fit_suffix(xs, s, W, min_pf, max_pf, tol_frac):
    sub = xs[s:]
    if len(sub) < 4:
        return None
    lo, hi = min_pf * W, max_pf * W
    cand = set()
    for i in range(len(sub)):
        for j in range(i + 1, len(sub)):
            d = sub[j] - sub[i]
            for k in range(1, 6):
                p = d / k
                if lo <= p <= hi:
                    cand.add(round(p, 2))
    best = None
    for p in sorted(cand):
        tol = tol_frac * p
        for anchor in sub:
            idx = np.round((sub - anchor) / p)
            resid = np.abs(sub - (anchor + idx * p))
            inl = resid < tol
            if inl.sum() < 4:
                continue
            # One rule can be detected twice, a pixel apart. Keep the better fit
            # for each slot and demote the rest, rather than discarding the whole
            # hypothesis - a single duplicate would otherwise veto every pitch.
            n_dup = 0
            for slot in np.unique(idx[inl]):
                same = np.where(inl & (idx == slot))[0]
                if len(same) > 1:
                    inl[same] = False
                    inl[same[np.argmin(resid[same])]] = True
                    n_dup += len(same) - 1
            if inl.sum() < 4:
                continue
            ii = idx[inl]
            span = int(ii.max() - ii.min()) + 1
            empty = span - len(ii)
            # Duplicates are not failures of the hypothesis, so they leave both
            # the acceptance fraction and the score alone.
            usable = len(sub) - n_dup
            n_out = int((~inl).sum()) - n_dup
            if usable <= 0 or inl.sum() / usable < 0.80 or empty > 0.4 * span:
                continue
            score = inl.sum() - 1.0 * empty - 1.5 * n_out
            if best is None or score > best[0]:
                best = (score, p, inl, idx, span, empty)
    if best is None:
        return None
    score, p, inl, idx, span, empty = best
    ii, xx = idx[inl], sub[inl]
    poly = np.poly1d(np.polyfit(ii, xx, 2 if len(ii) >= 5 else 1))
    rms = float(np.sqrt(((xx - poly(ii)) ** 2).mean()))
    return dict(start=s, pitch=p, poly=poly, idx=idx, inlier=inl, span=span,
                empty=empty, rms=rms, n_inlier=int(inl.sum()))


def fit_lattice(xs, W, min_pitch_frac=0.025, max_pitch_frac=0.16, tol_frac=0.12,
                max_label_cols=5):
    """Find the arithmetic progression the printed column rules sit on.

    A line ruled down a column by hand is straight and full-height, so it is
    indistinguishable from a printed rule one at a time - but it does not land on
    the lattice, so fitting the progression rejects it. The fit also recovers
    rules the detector missed and yields the pitch and column count, none of
    which then have to be hardcoded.

    The label columns (Sno / Roll No / Name) are far wider and are not on the
    lattice, so the fit is allowed to describe only a suffix of the candidates.
    """
    xs = np.asarray(sorted(xs), float)
    out = []
    for s in range(0, min(max_label_cols + 1, max(1, len(xs) - 3))):
        r = _fit_suffix(xs, s, W, min_pitch_frac, max_pitch_frac, tol_frac)
        if r:
            out.append(r)
    if not out:
        return None
    # A lattice describing the real column run leaves no empty slot; one stretched
    # across the wide label columns does. Rank on that first.
    out.sort(key=lambda r: (r["empty"], -(r["n_inlier"]), r["rms"]))
    best = out[0]
    # Re-expand to the full candidate list. The fit only describes a suffix, and
    # returning short arrays invites a silent zip() misalignment at the call site.
    s0 = best["start"]
    full_idx = np.full(len(xs), np.nan)
    full_inl = np.zeros(len(xs), bool)
    full_idx[s0:] = best["idx"]
    full_inl[s0:] = best["inlier"]
    best["idx"], best["inlier"] = full_idx, full_inl
    best["xs"] = xs
    return best


def _support_at(bw, x, y0, y1, tol=4, step=3):
    """Fraction of scanlines with dark pixels at x. A printed rule scores high
    even where pen ink covers it; open paper does not."""
    ys = np.arange(int(y0), int(y1), step)
    xi = int(round(x))
    lo, hi = max(0, xi - tol), min(bw.shape[1], xi + tol + 1)
    if lo >= hi or not len(ys):
        return 0.0
    return float((bw[ys, lo:hi] > 0).any(axis=1).mean())


def _best_support(bw, xpred, y0, y1, half):
    """Search a window around a predicted position - extrapolating the lattice
    beyond the fitted range drifts by a few percent of the pitch."""
    xs = np.arange(int(xpred - half), int(xpred + half) + 1, 2)
    xs = xs[(xs >= 0) & (xs < bw.shape[1])]
    if not len(xs):
        return 0.0, float(xpred)
    sc = [_support_at(bw, x, y0, y1) for x in xs]
    k = int(np.argmax(sc))
    return sc[k], float(xs[k])


RULE_SUPPORT_T = 0.45     # rule present vs open paper; measured 0.6-1.0 vs 0.15-0.4


def _trim_stray_rules(hl, max_gap_ratio=2.2):
    """Drop leading/trailing horizontal rules that are far out of step with the
    row pitch - typically a doubled title rule from the photocopy."""
    if len(hl) < 5:
        return hl
    mids = np.array([c.mid for c in hl])
    gaps = np.diff(mids)
    pitch = float(np.median(gaps))
    if pitch <= 0:
        return hl
    lo = 0
    while lo < len(gaps) - 2 and gaps[lo] > max_gap_ratio * pitch:
        lo += 1
    hi = len(hl) - 1
    while hi > lo + 3 and gaps[hi - 1] > max_gap_ratio * pitch:
        hi -= 1
    return hl[lo:hi + 1]


# --------------------------------------------------------------------------- #
# Page geometry
# --------------------------------------------------------------------------- #
class PageGrid:
    """Reconstructed table geometry for one page, in render-pixel coordinates.

    Nothing about the layout is assumed: the column pitch, the number of mark
    columns and the table's extent are all measured from the page.
    """

    def __init__(self, img: np.ndarray, scale: float):
        self.img = img
        self.scale = scale
        self.ink, self.printed = make_masks(img)
        self.bw = binarize(img)
        self.shape = self.printed.shape
        H, W = self.shape

        hl = find_hlines(self.bw)
        # 3 rules = header + one data row, which some final pages legitimately have
        if len(hl) < 3:
            raise ValueError("row rules not found - is this a ruled attendance sheet?")

        # Rules are detected on the raw binary, not the ink-subtracted mask: on
        # some sheets the absent marks are ruled straight down the printed lines,
        # so subtracting the pen ink deletes the printed rule with it.
        vl = find_vlines(self.bw, hl[0].mid, hl[-1].mid, minfrac=0.5)
        if len(vl) < 4:
            raise ValueError("column rules not found")

        # Several scans carry a doubled title rule a few hundred pixels above the
        # header. Rather than trusting the vertical rules' extent - which is
        # unreliable when the print is faint at the top of a page - drop leading
        # and trailing rules that sit far out of step with the row pitch.
        hl = _trim_stray_rules(hl)
        vl = find_vlines(self.bw, hl[0].mid, hl[-1].mid, minfrac=0.5) or vl
        self.hl = hl

        ymid = (hl[0].mid + hl[-1].mid) / 2
        xs = np.array([float(c(ymid)) for c in vl])
        order = np.argsort(xs)
        xs, vl = xs[order], [vl[i] for i in order]
        self.vl = vl

        lat = fit_lattice(xs, W)
        if lat is None or lat["n_inlier"] < 5:
            # Faint print can starve the detector; ask again for shorter runs.
            vl2 = find_vlines(self.bw, hl[0].mid, hl[-1].mid, minfrac=0.32)
            if len(vl2) > len(vl):
                xs2 = np.array([float(c(ymid)) for c in vl2])
                o2 = np.argsort(xs2)
                xs2, vl2 = xs2[o2], [vl2[i] for i in o2]
                lat2 = fit_lattice(xs2, W)
                if lat2 is not None and (lat is None or
                                         lat2["n_inlier"] > lat["n_inlier"]):
                    xs, vl, lat = xs2, vl2, lat2
                    self.vl = vl
        if lat is None:
            raise ValueError("could not resolve the column grid")
        self.pitch = float(lat["pitch"])
        self._lat = lat

        # map lattice index -> detected Curve, so the page bow is preserved
        curve_by_idx = {}
        for x, i, ok, c in zip(xs, lat["idx"], lat["inlier"], vl):
            if ok:
                curve_by_idx[int(i)] = c
        lo_i, hi_i = min(curve_by_idx), max(curve_by_idx)

        # extend outwards while the page still shows a rule there
        y0d, y1d = hl[1].mid, hl[-1].mid
        pos = {i: float(lat["poly"](i)) for i in range(lo_i, hi_i + 1)}
        k = lo_i
        while k - 1 >= lo_i - 12:
            s, xb = _best_support(self.bw, float(lat["poly"](k - 1)), y0d, y1d,
                                  0.35 * self.pitch)
            if s < RULE_SUPPORT_T:
                break
            k -= 1
            pos[k] = xb
        j = hi_i
        while True:
            s, xb = _best_support(self.bw, float(lat["poly"](j + 1)), y0d, y1d,
                                  0.35 * self.pitch)
            if s < RULE_SUPPORT_T or xb >= W - 4:
                break
            j += 1
            pos[j] = xb
        mark_lo, mark_hi = k, j
        self.n_mark = mark_hi - mark_lo               # boundaries bound n_mark columns

        # label columns: the candidates left of the first mark boundary
        left = [(float(x), c) for x, c in zip(xs, vl)
                if x < pos[mark_lo] - 0.55 * self.pitch]
        self._x0, self._curve = {}, {}
        if len(left) >= 3:
            picked = [left[0]] + left[-2:]            # table edge, Sno/Roll, Roll/Name
        elif left:
            picked = ([left[0]] * (3 - len(left))) + left
        else:                                          # nothing found: fall back
            step = pos[mark_lo] / 3.0
            picked = [(step * t, None) for t in range(3)]
        for jj, (x, c) in enumerate(picked[:3]):
            self._x0[jj] = x
            self._curve[jj] = c
        for n, i in enumerate(range(mark_lo, mark_hi + 1)):
            self._x0[3 + n] = pos[i]
            self._curve[3 + n] = curve_by_idx.get(i)

        self.n_bounds = 3 + self.n_mark + 1
        self.col_idx = {j: c for j, c in self._curve.items() if c is not None}
        self.n_rows = len(self.hl) - 2
        self._text = self._text_alt = self._hand = None

    # -- columns ----------------------------------------------------------- #
    def col_x(self, j: int, y: float) -> float:
        """x of boundary j at height y.

        Boundaries whose rule was detected carry their own fitted curve. The rest
        borrow the nearest detected curve's shape and are offset to their own
        position, so they still follow the page bow.
        """
        j = max(0, min(self.n_bounds - 1, int(j)))
        c = self._curve.get(j)
        if c is not None:
            return float(c(y))
        near = min((k for k in self._curve if self._curve[k] is not None),
                   key=lambda k: abs(k - j), default=None)
        if near is None:
            return float(self._x0[j])
        return float(self._curve[near](y)) + self._x0[j] - self._x0[near]

    def force_n_mark(self, n: int) -> None:
        """Make every page of a sheet agree on the column count."""
        if n == self.n_mark:
            return
        step = self._x0[3 + self.n_mark] - self._x0[3 + self.n_mark - 1] \
            if self.n_mark >= 1 else self.pitch
        for extra in range(self.n_mark + 1, n + 1):
            self._x0[3 + extra] = self._x0[3 + extra - 1] + step
            self._curve[3 + extra] = None
        for drop in range(n + 1, self.n_mark + 1):
            self._x0.pop(3 + drop, None)
            self._curve.pop(3 + drop, None)
        self.n_mark = n
        self.n_bounds = 3 + n + 1
        self.col_idx = {j: c for j, c in self._curve.items() if c is not None}

    # -- cells ------------------------------------------------------------- #
    def cell_box(self, ri: int, cj: int):
        """Box of the cell in data row `ri` between boundaries cj, cj+1. Each edge
        is evaluated at the cell's own position, so the page bow is followed."""
        ymid0 = self.hl[ri + 1].mid
        xm = (self.col_x(cj, ymid0) + self.col_x(cj + 1, ymid0)) / 2
        yt, yb = float(self.hl[ri + 1](xm)), float(self.hl[ri + 2](xm))
        ym = (yt + yb) / 2
        return self.col_x(cj, ym), yt, self.col_x(cj + 1, ym), yb

    def header_box(self, cj: int):
        """Header cell for mark column cj, extended upward - the handwritten dates
        frequently ride above the rule."""
        xm = (self.col_x(3 + cj, self.hl[0].mid) + self.col_x(4 + cj, self.hl[0].mid)) / 2
        yt, yb = float(self.hl[0](xm)), float(self.hl[1](xm))
        ym = (yt + yb) / 2
        return (self.col_x(3 + cj, ym), max(0.0, yt - 0.75 * (yb - yt)),
                self.col_x(4 + cj, ym), yb)

    def header_ink(self, cj: int) -> float:
        x0, y0, x1, y1 = self.header_box(cj)
        sub = self.ink[int(y0):int(y1), max(0, int(x0) + 6):int(x1) - 6]
        return float(sub.mean() / 255.0) if sub.size else 0.0

    def header_crop(self, cj: int) -> np.ndarray:
        """RGB crop of a header cell, for showing the user the handwritten date."""
        x0, y0, x1, y1 = self.header_box(cj)
        crop = self.img[int(y0):int(y1), max(0, int(x0)):int(x1)]
        return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    # -- lazily built masks ------------------------------------------------- #
    @property
    def text(self) -> np.ndarray:
        if self._text is None:
            h, w = self.shape
            self._text = line_free_text(self.printed, w, h)
        return self._text

    @property
    def text_alt(self) -> np.ndarray:
        """Same, with the pen ink left in - a fallback when a signature overlays
        the printed roll number."""
        if self._text_alt is None:
            h, w = self.shape
            self._text_alt = line_free_text(self.bw, w, h)
        return self._text_alt

    def _overflow_mask(self, max_gap: int = 26) -> np.ndarray:
        """Printed name text that runs past the Name column into the mark cells.

        Mark cells are assumed to hold no printed content. That holds on sheets
        with a roomy Name column; on a narrow one a long name runs over the rule
        and would read as a signature. The spilled glyphs are separate components,
        so connectivity does not catch them.

        What does catch them is horizontal continuity: a spill is the same line of
        text continuing, so its glyphs stay within one character space of each
        other. Following that run rightwards from the name and stopping at the
        first real gap erases the spill and leaves a signature - which has white
        space around it - untouched.
        """
        h, w = self.shape
        out = np.zeros((h, w), np.uint8)
        # the ruling would otherwise bridge every gap
        pr = cv2.bitwise_and(self.printed, cv2.bitwise_not(self._rule_mask()))
        for ri in range(self.n_rows):
            _, yt, _, yb = self.cell_box(ri, 2)             # Name cell
            y0, y1 = max(0, int(yt) + 3), min(h, int(yb) - 3)
            if y1 - y0 < 6:
                continue
            x_name = int(self.col_x(2, (y0 + y1) / 2))      # Name column, left edge
            x_end = int(self.col_x(self.n_bounds - 1, (y0 + y1) / 2))
            x_name, x_end = max(0, x_name), min(w, x_end)
            if x_end - x_name < 10:
                continue

            prof = (pr[y0:y1, x_name:x_end] > 0).any(axis=0)
            ink = np.where(prof)[0]
            if not len(ink):
                continue
            cur = int(ink[0])                                # follow the text run
            for x in ink[1:]:
                if x - cur > max_gap:
                    break
                cur = int(x)
            bx = int(self.col_x(3, (y0 + y1) / 2)) - x_name  # first mark boundary
            if cur <= bx:
                continue                                     # name stops in its column

            # Erase only within the band the printed name itself occupies. Printed
            # capitals sit on a baseline at constant height, so a spill lies inside
            # that band, whereas a signature written alongside the name loops above
            # and below it and keeps enough ink to still register.
            name_band = pr[y0:y1, x_name:x_name + max(1, bx)]
            rowsy = np.where((name_band > 0).any(axis=1))[0]
            if not len(rowsy):
                continue
            pad = max(2, int(0.12 * (rowsy.max() - rowsy.min() + 1)))
            ya = y0 + max(0, int(rowsy.min()) - pad)
            yb2 = y0 + min(y1 - y0, int(rowsy.max()) + pad + 1)
            out[ya:yb2, x_name + bx:x_name + cur + 1] = \
                pr[ya:yb2, x_name + bx:x_name + cur + 1]
        return out

    @property
    def overflow(self) -> np.ndarray:
        if getattr(self, "_ovf", None) is None:
            self._ovf = self._overflow_mask()
        return self._ovf

    @property
    def hand(self) -> np.ndarray:
        """All handwriting in the mark columns, independent of pen colour."""
        if self._hand is None:
            hand = cv2.bitwise_and(self.bw, cv2.bitwise_not(self.overflow))
            hand = cv2.bitwise_and(hand, cv2.bitwise_not(self._rule_mask()))
            hand = cv2.morphologyEx(hand, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            h, w = self.shape
            keep = np.zeros((h, w), np.uint8)
            for y in range(h):
                keep[y, max(0, int(self.col_x(3, y))):int(self.col_x(self.n_bounds - 1, y))] = 255
            self._hand = cv2.bitwise_and(hand, keep)
        return self._hand

    # -- content written outside the ruled table ---------------------------- #
    def outside_ink(self, margin: float = 0.02) -> tuple:
        """Handwriting entered outside the ruled table.

        Some sheets continue past the last printed row with names written freehand
        below it. Those cannot be read reliably and are left for a person, so the
        page is reported rather than guessed at. Returns (n_blobs, ink_fraction).
        """
        h, w = self.shape
        pad = int(margin * h)
        top = int(min(self.hl[0](x) for x in (0, w - 1))) - pad
        bot = int(max(self.hl[-1](x) for x in (0, w - 1))) + pad
        left = int(min(self.col_x(0, y) for y in (top, bot))) - pad
        right = int(max(self.col_x(self.n_bounds - 1, y) for y in (top, bot))) + pad

        outside = self.ink.copy()                      # pen ink only, not print
        outside[max(0, top):max(0, bot), max(0, left):max(0, right)] = 0
        outside[:, :int(0.02 * w)] = 0                 # page edges and scan shadow
        outside[:, int(0.98 * w):] = 0
        outside[:int(0.02 * h), :] = 0
        outside[int(0.98 * h):, :] = 0
        outside = cv2.morphologyEx(outside, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        n, _, st, _ = cv2.connectedComponentsWithStats(outside, 8)
        blobs = sum(1 for i in range(1, n) if st[i][4] > 120)
        return blobs, float((outside > 0).mean())

    def _rule_mask(self, thick: int = 11) -> np.ndarray:
        h, w = self.shape
        m = np.zeros((h, w), np.uint8)
        xs = np.arange(0, w, 4)
        for c in self.hl:
            pts = np.stack([xs, np.clip(c(xs), 0, h - 1)], 1).astype(np.int32)
            cv2.polylines(m, [pts], False, 255, thick)
        ys = np.arange(0, h, 4)
        for j in range(self.n_bounds):
            xv = np.array([self.col_x(j, y) for y in ys])
            pts = np.stack([np.clip(xv, 0, w - 1), ys], 1).astype(np.int32)
            cv2.polylines(m, [pts], False, 255, thick)
        return m

# --------------------------------------------------------------------------- #
# OCR of the row labels
# --------------------------------------------------------------------------- #
ROLL_RE = re.compile(r"BT\d{2}[DS]\d{3}")
_BFIX = str.maketrans({"8": "B", "3": "B", "6": "B", "R": "B", "P": "B", "E": "B",
                       "1": "T", "7": "T", "I": "T", "Y": "T"})

DIGITS = "--psm 7 -c tessedit_char_whitelist=0123456789"
ALNUM = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def norm_roll(t):
    """Normalise an OCR'd roll number, repairing the usual glyph confusions.

    Tesseract routinely reads the B of BT as 8/3/R and the T as 1/7. The format
    is rigid enough (BT<2 digits><D|S><3 digits>) to repair these safely.
    """
    t = re.sub(r"[^A-Z0-9]", "", (t or "").upper())
    if not t:
        return None
    t = t[:2].translate(_BFIX) + t[2:]                 # repair only the prefix
    if not t.startswith("BT") and re.match(r"^\d{2}[A-Z0-9]\d{3}$", t):
        t = "BT" + t                                    # prefix lost entirely
    m = re.match(r"^BT(\d{2})([DS05836])(\d{3})$", t.replace("O", "0").replace("Q", "0"))
    if not m:
        return None
    mid = {"5": "S", "8": "S", "0": "D", "3": "D", "6": "D"}.get(m.group(2), m.group(2))
    return f"BT{m.group(1)}{mid}{m.group(3)}"


def _crop(T, box, inset_x=8, inset_y=5):
    xl, yt, xr, yb = box
    y0, y1 = int(yt) + inset_y, int(yb) - inset_y
    x0, x1 = max(0, int(xl) + inset_x), int(xr) - inset_x
    return None if (y1 - y0 < 8 or x1 - x0 < 8) else T[y0:y1, x0:x1]


def _pad(crop, scale):
    c = cv2.copyMakeBorder(crop, 25, 25, 25, 25, cv2.BORDER_CONSTANT, value=255)
    return cv2.resize(c, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def _ocr(crop, cfg, scale=2.5):
    if crop is None or crop.size == 0:
        return ""
    return pytesseract.image_to_string(_pad(crop, scale), config=cfg).strip()


def _ocr_words(crop, scale=2.0):
    """OCR a text block preserving word breaks.

    Deliberately no character whitelist: a whitelist destroys tesseract's word
    segmentation and returns SURYASNAIR at zero confidence rather than
    SURYA S NAIR at 90.
    """
    if crop is None or crop.size == 0:
        return ""
    d = pytesseract.image_to_data(_pad(crop, scale), config="--psm 6",
                                  output_type=pytesseract.Output.DICT)
    toks = [(d["block_num"][i], d["line_num"][i], d["left"][i], t.strip())
            for i, t in enumerate(d["text"]) if t.strip() and float(d["conf"][i]) > 30]
    toks.sort(key=lambda x: x[:3])
    s = " ".join(t for *_, t in toks).upper()
    return " ".join(re.sub(r"[^A-Z0-9 .()-]", "", s).split())


def ocr_row_labels(g: PageGrid, ri: int, with_names: bool = True):
    sno_raw = _ocr(_crop(g.text, g.cell_box(ri, 0), 4), DIGITS, 3.0)
    roll_raw = _ocr(_crop(g.text, g.cell_box(ri, 1)), ALNUM)

    roll = norm_roll(roll_raw)
    if roll is None:
        m = ROLL_RE.search(re.sub(r"[^A-Z0-9]", "", roll_raw.upper()))
        roll = m.group() if m else None
    if roll is None:                                    # retry, ink left in place
        roll = norm_roll(_ocr(_crop(g.text_alt, g.cell_box(ri, 1)), ALNUM, 3.0))

    name = _ocr_words(_crop(g.text, g.cell_box(ri, 2))) if with_names else ""
    sno = re.sub(r"\D", "", sno_raw)
    return dict(sno_ocr=int(sno) if sno.isdigit() else None,
                roll=roll, roll_raw=roll_raw, name=name)


# --------------------------------------------------------------------------- #
# Cell classification
# --------------------------------------------------------------------------- #
def _stroke_like(cw, ch, W, H) -> bool:
    """A ruled 'absent' stroke: narrow, much taller than wide, running most of
    the cell height."""
    return (cw / W < T_WFRAC) and (ch / max(1, cw) > T_ASPECT) and (ch / H > T_HFRAC)


def cell_features(g: PageGrid, ri: int, cj: int, inset: float = 0.10) -> dict:
    """Features for one mark cell, computed per connected component.

    Cells often hold two *different* absent strokes - their own and the
    neighbouring column's bleeding past the rule. Their union bounding box is
    wide and looks convincingly like a signature; each component alone is
    plainly a thin vertical line.
    """
    xl, yt, xr, yb = g.cell_box(ri, 3 + cj)
    w, h = xr - xl, yb - yt
    ix, iy = int(w * inset), int(h * inset)             # inset clears the ruling
    H0, W0 = g.shape
    x0, y0 = max(0, int(xl) + ix), max(0, int(yt) + iy)
    x1, y1 = min(W0, int(xr) - ix), min(H0, int(yb) - iy)

    base = dict(area=0.0, mark=0.0, wfrac=0.0, aspect=0.0, hfrac=0.0,
                ncomp=0, nstroke=0, overflow=0.0, box=(x0, y0, x1, y1))
    if x1 - x0 < 6 or y1 - y0 < 6:
        return base

    sub = (g.hand[y0:y1, x0:x1] > 0).astype(np.uint8)
    npx = sub.size
    n, lab, st, _ = cv2.connectedComponentsWithStats(sub, 8)
    H, W = sub.shape

    mark_px = nstroke = ncomp = 0
    best = (0.0, 0.0, 0.0, 0.0)
    for i in range(1, n):
        _, _, cw, ch, a = st[i]
        if a / npx < MIN_COMP:
            continue
        ncomp += 1
        if _stroke_like(cw, ch, W, H):
            nstroke += 1
            continue
        mark_px += a
        if a > best[0]:
            best = (a, cw / W, ch / max(1, cw), ch / H)

    ovf = g.overflow[y0:y1, x0:x1]
    base.update(area=float(sub.sum()) / npx, mark=float(mark_px) / npx,
                ncomp=int(ncomp), nstroke=int(nstroke),
                overflow=float((ovf > 0).mean()) if ovf.size else 0.0,
                wfrac=float(best[1]), aspect=float(best[2]), hfrac=float(best[3]))
    return base


def auto_threshold(marks, default: float = T_MARKAREA,
                   min_ratio: float = 1.25, lo: float = 0.008, hi: float = 0.20):
    """Choose the present/absent threshold from the sheet's own ink distribution.

    An absolute cut-off cannot serve every sheet: pen weight and cell size vary,
    so a signature covers a different fraction of its cell from one sheet to the
    next. The two classes are strongly bimodal, so the threshold belongs in the
    widest multiplicative gap between them. If no clear gap exists the default is
    kept and the caller is told the separation is poor.
    """
    m = np.sort(np.asarray([v for v in marks if v > 0.002], float))
    if len(m) < 12:
        return float(default), 1.0, 0.0, 0.0
    best = (1.0, 0.0, 0.0)
    for a, b in zip(m[:-1], m[1:]):
        if a < lo or b > hi or a <= 0:
            continue
        if b / a > best[0]:
            best = (b / a, float(a), float(b))
    ratio, a, b = best
    if ratio < min_ratio:
        return float(default), float(ratio), a, b
    return float(np.sqrt(a * b)), float(ratio), a, b


def classify(f: dict, thresh: float = None):
    """-> (label, confidence, reason).

    Confidence is distance from the threshold on a log scale: a cell an order of
    magnitude clear of it scores ~0.99, one sitting on it scores ~0.5.
    """
    t = T_MARKAREA if thresh is None else float(thresh)
    m = float(f["mark"])
    # Printed name text was stripped from this cell. Usually right, but a
    # signature written hard against the name can go with it, so never call such
    # a cell absent with confidence - surface it for review instead.
    if m < t and f.get("overflow", 0.0) > 0.01:
        return "N", 0.55, "name text removed - check"
    if m >= t:
        conf = 0.5 + 0.49 * min(1.0, math.log(m / t) / math.log(CONF_RATIO))
        return "P", round(conf, 3), "signature"
    if m <= 0:
        return "N", 0.99, "ruled through" if f["nstroke"] else "blank"
    conf = 0.5 + 0.49 * min(1.0, math.log(t / m) / math.log(CONF_RATIO))
    return "N", round(conf, 3), "ruled through" if f["nstroke"] else "mark too small"


# --------------------------------------------------------------------------- #
# Sheet-level driving
# --------------------------------------------------------------------------- #
@dataclass
class Sheet:
    """Geometry for every page, plus which mark columns carry a date."""
    grids: list = field(default_factory=list)
    header_ink: np.ndarray = None
    live: list = field(default_factory=list)
    n_mark: int = 0
    threshold: float = T_MARKAREA
    gap_ratio: float = 0.0
    page_errors: list = field(default_factory=list)
    outside_writing: list = field(default_factory=list)   # (page_no, blobs, ink)

    @property
    def n_rows(self) -> int:
        return sum(g.n_rows for g in self.grids)


def analyse(pdf_bytes: bytes, progress=None) -> Sheet:
    """Pass 1: geometry only. Fast, and enough to show the user the date headers."""
    doc = open_pdf(pdf_bytes)
    grids, errors = [], []
    for p in range(len(doc)):
        img, scale = render(doc, p)
        try:
            grids.append(PageGrid(img, scale))
        except ValueError as e:
            errors.append((p + 1, str(e)))           # a bad page must not kill the run
        if progress:
            progress((p + 1) / len(doc), f"Reading page {p + 1} of {len(doc)}")
    if not grids:
        raise ValueError(
            "No page yielded a readable table. This pipeline needs a printed "
            "ruled grid; a fully handwritten sign-in sheet cannot be read this way.")

    # Pages of one sheet must agree on the column count; a page where a faint edge
    # rule was missed is corrected to the majority rather than left inconsistent.
    counts = [g.n_mark for g in grids]
    n_mark = int(np.bincount(counts).argmax())
    for g in grids:
        g.force_n_mark(n_mark)

    # Entries written outside the ruled table are reported, never guessed at:
    # they are for a person to enter by hand.
    outside = []
    for p, g in enumerate(grids):
        blobs, frac = g.outside_ink()
        if blobs >= OUTSIDE_BLOBS and frac >= OUTSIDE_INK:
            outside.append((p + 1, int(blobs), float(frac)))

    hdr = np.array([[g.header_ink(c) for c in range(n_mark)] for g in grids])

    # Decided once for the whole sheet: per-page calls disagree (a faint date
    # here, a stray mark there), and the gap between real and unused is ~10x.
    live = [c for c in range(n_mark) if hdr.max(axis=0)[c] > LIVE_T]
    return Sheet(grids=grids, header_ink=hdr, live=live, n_mark=n_mark,
                 page_errors=errors, outside_writing=outside)


def extract(sheet: Sheet, with_names: bool = True, progress=None) -> list:
    """Pass 2: OCR the labels and classify every live mark cell.

    Cell features are gathered first and the present/absent threshold is chosen
    from the sheet's own ink distribution before anything is labelled - pen weight
    and cell size differ enough between sheets that one fixed cut-off does not
    serve them all.
    """
    rows, total, done = [], sheet.n_rows, 0
    for p, g in enumerate(sheet.grids):
        for ri in range(g.n_rows):
            lab = ocr_row_labels(g, ri, with_names=with_names)
            feats = [cell_features(g, ri, cj) for cj in sheet.live]
            rows.append((p, ri, lab, feats))
            done += 1
            if progress and done % 5 == 0:
                progress(0.9 * done / total, f"Reading row {done} of {total}")

    thresh, ratio, lo, hi = auto_threshold(
        [f["mark"] for _, _, _, fs in rows for f in fs])
    sheet.threshold, sheet.gap_ratio = float(thresh), float(ratio)

    records = []
    for p, ri, lab, feats in rows:
        cells = []
        for cj, f in zip(sheet.live, feats):
            mark, conf, why = classify(f, thresh)
            cells.append(dict(col=int(cj), label=mark, conf=float(conf), reason=why,
                              mark=round(float(f["mark"]), 5),
                              overflow=round(float(f.get("overflow", 0.0)), 4),
                              box=tuple(int(v) for v in f["box"])))
        records.append(dict(page=int(p), ri=int(ri), cells=cells, **lab))
    if progress:
        progress(1.0, f"Extracted {total} rows")
    return records


def calibration(records: list, thresh: float = None) -> dict:
    """Where the two classes sit relative to the threshold.

    If `gap_ratio` is close to 1 the threshold is doing delicate work and the
    output deserves checking; a wide gap means the classes are cleanly separated.
    """
    t = T_MARKAREA if thresh is None else float(thresh)
    m = np.array([c["mark"] for r in records for c in r["cells"]], float)
    below, above = m[m < t], m[m >= t]
    hi_absent = float(below.max()) if below.size else 0.0
    lo_present = float(above.min()) if above.size else 0.0
    return dict(marks=m, threshold=t, highest_absent=hi_absent, lowest_present=lo_present,
                gap_ratio=(lo_present / hi_absent) if hi_absent > 0 else float("inf"),
                n_present=int(above.size), n_absent=int(below.size))


# --------------------------------------------------------------------------- #
# Tables and exports
# --------------------------------------------------------------------------- #
def _py(v):
    """NumPy scalar -> native Python scalar; everything else untouched."""
    return v.item() if isinstance(v, np.generic) else v


def build_tables(records: list, date_labels: list):
    """-> (columns, rows, review_columns, review_rows), all native Python types.

    Values are hard-cast here rather than left as NumPy scalars: some
    NumPy/pandas pairings raise inside `maybe_convert_objects` on mixed
    object-dtype columns.
    """
    columns = (["Sno", "Roll No", "Name"] + list(date_labels) + ["Present", "Absent"]
               + [f"conf {d}" for d in date_labels] + ["Needs review", "Sno OCR check"])
    review_columns = ["Sno", "Roll No", "Name", "Date", "Called", "Confidence",
                      "Reason", "Ink fraction", "PDF page"]

    rows, review = [], []
    for i, r in enumerate(records):
        marks = [str(c["label"]) for c in r["cells"]]
        confs = [float(_py(c["conf"])) for c in r["cells"]]

        row = {"Sno": int(i + 1),
               "Roll No": str(r["roll"] or ""),
               "Name": str(r["name"] or "")}
        row.update({str(d): m for d, m in zip(date_labels, marks)})
        row["Present"] = int(marks.count("P"))
        row["Absent"] = int(marks.count("N"))
        row.update({f"conf {d}": c for d, c in zip(date_labels, confs)})
        row["Needs review"] = "; ".join(str(d) for d, c in zip(date_labels, confs)
                                        if c < REVIEW_CONF)
        row["Sno OCR check"] = "ok" if r["sno_ocr"] == i + 1 else f"read {r['sno_ocr']}"
        rows.append({k: row.get(k, "") for k in columns})

        for d, c in zip(date_labels, r["cells"]):
            if float(_py(c["conf"])) < REVIEW_CONF:
                review.append({"Sno": int(i + 1), "Roll No": str(r["roll"] or ""),
                               "Name": str(r["name"] or ""), "Date": str(d),
                               "Called": str(c["label"]),
                               "Confidence": float(_py(c["conf"])),
                               "Reason": str(c["reason"]),
                               "Ink fraction": float(_py(c["mark"])),
                               "PDF page": int(r["page"]) + 1})
    return columns, rows, review_columns, review


def to_csv_bytes(columns: list, rows: list) -> bytes:
    import csv
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(columns)
    w.writerows([[row[c] for c in columns] for row in rows])
    return buf.getvalue().encode("utf-8")


def to_dataframe(columns: list, rows: list):
    """Built one typed column at a time, and without passing `columns=` to a
    dict-based constructor - that route goes through `maybe_convert_objects`,
    which fails on NumPy/pandas builds with mismatched ABIs."""
    import pandas as pd

    int_cols = {"Sno", "Present", "Absent", "PDF page"}
    float_cols = {c for c in columns if c.startswith("conf ")} | {"Confidence", "Ink fraction"}
    out = {}
    for c in columns:
        vals = [row[c] for row in rows]
        if c in int_cols:
            out[c] = pd.Series(vals, dtype="int64")
        elif c in float_cols:
            out[c] = pd.Series(vals, dtype="float64")
        else:
            out[c] = pd.Series([str(v) for v in vals], dtype="object")
    return pd.DataFrame(out)[list(columns)]


def annotate_pdf(pdf_bytes: bytes, sheet: Sheet, records: list) -> tuple:
    """Red box around every cell judged present.

    Drawn as vector graphics into the original PDF, so the scan underneath is
    untouched and the file stays re-processable.
    """
    ann = open_pdf(pdf_bytes)
    n = 0
    for r in records:
        page = ann[r["page"]]
        s = sheet.grids[r["page"]].scale
        for c in r["cells"]:
            if c["label"] != "P":
                continue
            x0, y0, x1, y1 = (v / s for v in c["box"])          # px -> PDF points
            page.draw_rect(fitz.Rect(x0 - 1, y0 - 1, x1 + 1, y1 + 1),
                           color=(0.85, 0.1, 0.1), width=1.6)
            n += 1
    data = ann.tobytes(garbage=3, deflate=True)
    ann.close()
    return data, n


def student_report(rows: list, date_labels: list, query: str):
    """Attendance for one student, looked up by roll number or name."""
    q = re.sub(r"[^A-Z0-9]", "", (query or "").upper())
    if not q:
        return None
    exact = [r for r in rows if re.sub(r"[^A-Z0-9]", "", r["Roll No"].upper()) == q]
    hits = exact or [r for r in rows
                     if q in re.sub(r"[^A-Z0-9]", "", r["Roll No"].upper())
                     or q in re.sub(r"[^A-Z0-9]", "", r["Name"].upper())]
    if not hits:
        return None

    r = hits[0]
    attended = [d for d in date_labels if r[d] == "P"]
    missed = [d for d in date_labels if r[d] == "N"]
    total = len(date_labels)
    return dict(row=r, roll=r["Roll No"], name=r["Name"], sno=r["Sno"],
                attended=attended, missed=missed, total=total,
                present=len(attended),
                percent=(100.0 * len(attended) / total) if total else 0.0,
                low_conf=[d for d in date_labels if r.get(f"conf {d}", 1.0) < REVIEW_CONF],
                alternatives=hits[1:6])
