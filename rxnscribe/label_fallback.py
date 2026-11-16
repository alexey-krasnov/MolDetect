"""
Layout + OCR fallback for molecule labels that the MolDetect coref decoder misses.

The decoder reliably finds the [Mol] box on single-structure patent images
(e.g. EP2209374B1_D0057.tif) but never emits an [Idt] box for large serif
captions such as "EC0575". The caption sits in whitespace directly below the
molecule box, so we find it from pixels and read it with OCR.

Usage:
    preds = model.predict_image(img, coref=True)            # img = ORIGINAL image
    preds = add_missing_labels(preds, img, ocr=model.ocr_model)
"""
import re
import numpy as np
from PIL import Image

# e.g. 030, 2a, EC0575, CPD-12, Compound 5, I-12
LABEL_RE = re.compile(r'^(?:(?:compound|cpd\.?|example|ex\.?)\s*)?[A-Za-z]{0,4}[-\s]?\d{1,5}[A-Za-z]?(?:\(\w+\))?$', re.I)

IDT = 3
MOL = 1


def _ink_mask(gray, thr=170):
    return gray < thr


def _bands(profile, gap):
    """Split a boolean 1-D profile into [start, end) runs, merging runs closer than `gap`."""
    idx = np.flatnonzero(profile)
    if idx.size == 0:
        return []
    runs, s, p = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - p > gap:
            runs.append((s, p + 1))
            s = i
        p = i
    runs.append((s, p + 1))
    return runs


def _default_ocr(ocr, crop_rgb):
    """Return recognized text for a crop. `ocr` is an easyocr.Reader, a callable(np.ndarray)->str, or None."""
    if ocr is None:
        return ''
    if callable(ocr):
        return ocr(crop_rgb)
    # easyocr.Reader; restrict the alphabet — captions are alphanumeric
    out = ocr.readtext(crop_rgb, detail=0, paragraph=False,
                       allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-() .')
    return ' '.join(out)


def find_caption_below(gray, mol_box, other_mol_boxes=(), max_gap_frac=0.35, max_height_frac=0.45):
    """
    Locate the first ink line below `mol_box` (pixel xyxy). Returns pixel xyxy or None.
    The search is bounded by the image bottom and by the top of any molecule below.
    """
    H, W = gray.shape
    x1, y1, x2, y2 = [int(round(v)) for v in mol_box]
    mw, mh = x2 - x1, y2 - y1
    sx1, sx2 = max(0, int(x1 - 0.25 * mw)), min(W, int(x2 + 0.25 * mw))
    sy1 = max(0, y2 - max(1, int(0.02 * mh)))
    sy2 = min(H, y2 + int(max_height_frac * mh))
    for ox1, oy1, ox2, oy2 in other_mol_boxes:       # stop above the next structure
        if oy1 > y2 and ox1 < sx2 and ox2 > sx1:
            sy2 = min(sy2, int(oy1))
    if sy2 - sy1 < 4:
        return None

    ink = _ink_mask(gray[sy1:sy2, sx1:sx2])
    rows = _bands(ink.any(axis=1), gap=max(2, int(0.01 * mh)))
    if not rows:
        return None
    r0, r1 = rows[0]
    # first band must start reasonably close to the molecule
    if r0 > max_gap_frac * mh:
        return None
    line = ink[r0:r1]
    # columns: merge letters of one caption (gap ~ a few glyph heights)
    glyph_h = r1 - r0
    cols = _bands(line.any(axis=0), gap=max(4, int(1.5 * glyph_h)))
    if not cols:
        return None
    # choose the column group closest to the molecule's horizontal centre
    cx = (x1 + x2) / 2 - sx1
    c0, c1 = min(cols, key=lambda c: abs((c[0] + c[1]) / 2 - cx))
    return sx1 + c0, sy1 + r0, sx1 + c1, sy1 + r1


def _read(image_rgb, box, ocr, pad=8, scale=3):
    x1, y1, x2, y2 = box
    H, W = image_rgb.shape[:2]
    crop = image_rgb[max(0, y1 - pad):min(H, y2 + pad), max(0, x1 - pad):min(W, x2 + pad)]
    # white margin + upscale helps OCR on tight crops
    im = Image.fromarray(crop).convert('RGB')
    im = im.resize((im.width * scale, im.height * scale), Image.LANCZOS)
    canvas = Image.new('RGB', (im.width + 40, im.height + 40), (255, 255, 255))
    canvas.paste(im, (20, 20))
    return _default_ocr(ocr, np.asarray(canvas)).strip()


def add_missing_labels(predictions, image, ocr=None, label_re=LABEL_RE):
    """
    predictions: output of MolDetect.predict_image(..., coref=True)
    image:       the same PIL image that was passed to the model
    ocr:         easyocr.Reader (model.ocr_model) or callable(np.ndarray)->str
    Adds an [Idt] bbox with 'text' and a coref pair for every molecule that has no label.
    Returns a new predictions dict.
    """
    rgb = np.asarray(image.convert('RGB'))
    gray = np.asarray(image.convert('L'))
    H, W = gray.shape
    bboxes = [dict(b) for b in predictions['bboxes']]
    corefs = [list(c) for c in predictions['corefs']]

    mols = [i for i, b in enumerate(bboxes) if b['category_id'] == MOL]
    labelled = {c[0] for c in corefs}
    px = lambda b: (b['bbox'][0] * W, b['bbox'][1] * H, b['bbox'][2] * W, b['bbox'][3] * H)
    mol_px = {i: px(bboxes[i]) for i in mols}

    for i in mols:
        if i in labelled:
            continue
        others = [v for k, v in mol_px.items() if k != i]
        box = find_caption_below(gray, mol_px[i], others)
        if box is None:
            continue
        text = _read(rgb, box, ocr)
        if not text or not label_re.match(text):
            continue
        x1, y1, x2, y2 = box
        bboxes.append({'category': '[Idt]', 'category_id': IDT, 'text': [text],
                       'bbox': (float(x1) / W, float(y1) / H, float(x2) / W, float(y2) / H),
                       'score': -1, 'source': 'fallback'})
        corefs.append([i, len(bboxes) - 1])
    return {'bboxes': bboxes, 'corefs': corefs}
