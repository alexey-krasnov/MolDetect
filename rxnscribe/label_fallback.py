"""
Layout + OCR fallback for molecule labels that the MolDetect coref decoder misses.

The decoder reliably finds [Mol] boxes on patent images but often emits no [Idt]
box for captions such as "EC0575" (single structure) or Roman numerals "XXIX"
(reaction schemes), and sometimes pairs a molecule with a wrong "label" (e.g. "HCl").

Approach:
  1. drop existing [Idt] boxes whose OCR text is clearly not a label (e.g. "HCl");
  2. OCR the whole page, keep label-like tokens (digits, 2a, EC0575, Roman numerals);
  3. match tokens to unlabelled molecules one-to-one by geometry (label below the
     molecule, horizontally aligned), cheapest pairs first.

Usage:
    preds = model.predict_image(img, coref=True, ocr=True)   # img = ORIGINAL image
    preds = add_missing_labels(preds, img, ocr=model.ocr_model)

`ocr` is an easyocr.Reader, or a callable(np.ndarray RGB) -> [(x1, y1, x2, y2, text), ...].
"""
import re
import numpy as np

MOL, IDT = 1, 3

# 030, 2a, 5-b, EC0575, CPD-12, "Compound 5"
LABEL_RE = re.compile(r'^(?:(?:compound|cpd\.?|example|ex\.?)\s*)?[A-Za-z]{0,4}-?\d{1,5}[A-Za-z]?(?:\(\w+\))?$', re.I)
# valid Roman numerals, case-sensitive (so "Cl" is not "CI")
ROMAN_RE = re.compile(r'^(?=[IVXLCDM])M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})$')
# chemical annotation, not a label: CH3, NH2, HCl, H3C, OH ...
ATOM_RE = re.compile(r'^(?:(?:Cl|Br|[CHNOSPFI])\d*)+$')
# substituent placeholders inside structures: R, R1, R2, Ra ... (not compound labels)
RGROUP_RE = re.compile(r'^R\d*[a-z]?$')


def normalize(text):
    t = text.strip().strip(',:;')
    # OCR often reads the last I of a Roman numeral as l / | / 1
    if re.fullmatch(r'[IVXl|1]+', t) and re.search('[VX]', t):
        t = re.sub(r'[l|1]', 'I', t)
    elif re.fullmatch(r'[Il|]{2,}', t):          # "Il", "lI", "||" -> "II" (digits like "11" stay)
        t = 'I' * len(t)
    return t


def is_label(text):
    t = normalize(text)
    if not t or len(t) > 12:
        return False
    if ROMAN_RE.match(t):
        return len(t) >= 2 or t in ('I', 'V', 'X')
    if ATOM_RE.match(t) and (re.search(r'[H\d]', t)):
        return False
    if RGROUP_RE.match(t):
        return False
    return bool(LABEL_RE.match(t))


def is_junk(text):
    """Clearly NOT a label: empty, or a chemical annotation such as HCl, CH3, NH2 (OCR may read HCl as HCI).
    Deliberately narrow: unusual but valid labels (A, 3aa, A-1-1, E3.1, (IV)) must never be dropped."""
    t = normalize(text)
    return not t or bool(ATOM_RE.match(t) and 'H' in t)


def _page_words(ocr, rgb, min_conf=0.2, margin=40):
    """-> list of (x1, y1, x2, y2, text) in original-image pixels.
    A white margin is added first: captions touching the scan edge are otherwise missed."""
    padded = np.pad(rgb, ((margin, margin), (margin, margin), (0, 0)), constant_values=255)
    if hasattr(ocr, 'readtext'):                       # easyocr.Reader
        words = []
        for poly, text, conf in ocr.readtext(padded, detail=1, paragraph=False):
            if conf < min_conf:
                continue
            xs, ys = [p[0] for p in poly], [p[1] for p in poly]
            words.append((min(xs), min(ys), max(xs), max(ys), text))
    else:
        words = list(ocr(padded))
    return [(x1 - margin, y1 - margin, x2 - margin, y2 - margin, t) for x1, y1, x2, y2, t in words]


def add_missing_labels(predictions, image, ocr, max_below=0.6, x_margin=0.15):
    """
    predictions: output of MolDetect.predict_image(..., coref=True)
    image:       the same PIL image that was passed to the model
    Returns a new predictions dict with wrong pairs removed and missing [Idt] boxes added.
    """
    W, H = image.size
    rgb = np.asarray(image.convert('RGB'))
    bboxes = [dict(b) for b in predictions['bboxes']]
    corefs = [list(c) for c in predictions['corefs']]

    # 1. drop [Idt] boxes whose text is clearly junk (e.g. "HCl") and their pairs; keep anything else
    drop = {i for i, b in enumerate(bboxes)
            if b['category_id'] == IDT and b.get('text') and is_junk(' '.join(b['text']))}
    if drop:
        remap, kept = {}, []
        for i, b in enumerate(bboxes):
            if i not in drop:
                remap[i] = len(kept)
                kept.append(b)
        corefs = [[remap[m], remap[t]] for m, t in corefs if m in remap and t in remap]
        bboxes = kept

    px = lambda b: (b['bbox'][0] * W, b['bbox'][1] * H, b['bbox'][2] * W, b['bbox'][3] * H)
    mols = {i: px(b) for i, b in enumerate(bboxes) if b['category_id'] == MOL}
    idts = [px(b) for b in bboxes if b['category_id'] == IDT]
    labelled = {m for m, _ in corefs}
    todo = [i for i in mols if i not in labelled]
    if not todo:
        return {'bboxes': bboxes, 'corefs': corefs}

    # 2. label-like OCR tokens that are not inside a molecule or an existing [Idt]
    cands = []
    for x1, y1, x2, y2, text in _page_words(ocr, rgb):
        t = normalize(text)
        if not is_label(t):
            continue
        cx, cy, lh = (x1 + x2) / 2, (y1 + y2) / 2, max(y2 - y1, 1)
        if any(a <= cx <= c and b <= cy <= d for a, b, c, d in idts):
            continue
        # inside a molecule (allow the bottom edge: the model box sometimes swallows the caption)
        if any(a <= cx <= c and b <= cy <= d - lh for a, b, c, d in mols.values()):
            continue
        cands.append((t, (x1, y1, x2, y2)))

    # 3. one-to-one matching, cheapest first: caption below the molecule, horizontally aligned
    pairs = []
    for i in todo:
        mx1, my1, mx2, my2 = mols[i]
        mw, mh = max(mx2 - mx1, 1), max(my2 - my1, 1)
        for j, (t, (x1, y1, x2, y2)) in enumerate(cands):
            cx, cy, lh = (x1 + x2) / 2, (y1 + y2) / 2, max(y2 - y1, 1)
            if not (mx1 - x_margin * mw <= cx <= mx2 + x_margin * mw):
                continue
            if not (my2 - 1.2 * lh <= cy <= my2 + max_below * mh):
                continue
            cost = abs(cy - my2) / mh + 0.5 * abs(cx - (mx1 + mx2) / 2) / mw
            pairs.append((cost, i, j))
    used_m, used_c = set(), set()
    for cost, i, j in sorted(pairs):
        if i in used_m or j in used_c:
            continue
        used_m.add(i)
        used_c.add(j)
        t, (x1, y1, x2, y2) = cands[j]
        bboxes.append({'category': '[Idt]', 'category_id': IDT, 'text': [t],
                       'bbox': (float(x1) / W, float(y1) / H, float(x2) / W, float(y2) / H),
                       'score': -1, 'source': 'fallback'})
        corefs.append([i, len(bboxes) - 1])
    return {'bboxes': bboxes, 'corefs': corefs}
