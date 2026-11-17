"""
Layout + OCR fallback for molecule labels that the MolDetect coref decoder misses.

The decoder reliably finds [Mol] boxes on patent images but often emits no [Idt]
box for captions such as "EC0575" (single structure) or Roman numerals "XXIX"
(reaction schemes), and sometimes pairs a molecule with a wrong "label" (e.g. "HCl").

Approach:
  1. drop existing [Idt] boxes whose OCR text is clearly not a label (e.g. "HCl");
  2. OCR the whole page, keep label-like tokens (digits, 2a, EC0575, Roman numerals);
  3. match tokens to unlabelled molecules one-to-one by geometry (label below the
     molecule, horizontally aligned), cheapest pairs first;
  4. for molecules that are still unlabelled, re-read a strip under the molecule on a
     3x upscaled crop (page-wide OCR misses small captions such as XXX, IX);
  5. as a last resort attach a strong token (Roman numeral, "(Ia)") that sits just beside
     the molecule (e.g. a corner of the page);
  6. repair labels the model read truncated ("A-1-" -> "A-1-2") by re-reading the box;
  0. (first) clean the model's own labels: re-read empty/garbled ones, fix OCR slips
     (CHz -> CH2), drop junk and duplicate boxes, keep letter labels (A..H) on letter pages;
  7. keep labels in the page's dominant style, e.g. "(c22)" in a grid, even when their
     molecule was not detected (unpaired [Idt], no coref).

Label grammar: 12, 2a, EC0575, A-1-2, E-3.1, XXIX, (I), (Ia), (c22), (X-25), (S-1),
P-THDT-DBTH-FFTDZ, E-1.17"', d-4a-I. Rejected: HCl/CH3, R-groups (R2, R35O), isotopes (18F),
formula fragments (C8H17, 8H), and bare numbers on pages whose labels have another style.

Usage:
    preds = model.predict_image(img, coref=True, ocr=True)   # img = ORIGINAL image
    preds = add_missing_labels(preds, img, ocr=model.ocr_model)

`ocr` is an easyocr.Reader, or a callable(np.ndarray RGB) -> [(x1, y1, x2, y2, text), ...].
"""
import re
from collections import Counter

import numpy as np

MOL, IDT = 1, 3

# 030, 2a, 5-b, EC0575, CPD-12, "Compound 5"
LABEL_RE = re.compile(r'^(?:(?:compound|cpd\.?|example|ex\.?)\s*)?'
                      r'(?:[A-Za-z]{1,4}-?\d{1,5}(?:[-.]\d{1,3})*[A-Za-z]?|\d{1,5}[A-Za-z]?)(?:\(\w+\))?$', re.I)
# (I), (Ia), (IV), (12), (2a): parenthesised labels
PAREN_RE = re.compile(r'^\((?:[IVXLC]{1,6}|\d{1,3})[a-z]?\)$')
# (c22), (C5), (X-25), (S-1), (AB-3a): letter prefix + number in parentheses (dense grids of structures)
PAREN_ALNUM_RE = re.compile(r'^\([A-Za-z]{1,3}-?\d{1,3}[a-z]?\)$')
# P-THDT-DBTH-FFTDZ, DMO-DPP-DB: hyphenated acronym names (polymers, building blocks)
ACRONYM_RE = re.compile(r'^[A-Z]{1,6}(?:-[A-Z]{1,6}){1,5}$')
# E-1.17, E-1.17', E-1.17"' : label + primes
PRIME_RE = re.compile(r'^([^\'"]+)([\'"]{1,3})$')
# d-4a-I, ent-5-I, A-1-2: prefix-number-suffix
STEREO_RE = re.compile(r'^[A-Za-z]{1,4}-\d{1,3}[a-z]{0,2}(?:-(?:\d{1,3}|[IVX]{1,4}))+$')
# valid Roman numerals, case-sensitive (so "Cl" is not "CI")
ROMAN_RE = re.compile(r'^(?=[IVXLCDM])M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})$')
# chemical annotation, not a label: CH3, NH2, HCl, H3C, OH ...
ATOM_RE = re.compile(r'^(?:(?:Cl|Br|[CHNOSPFI])\d*)+$')
# substituent placeholders inside structures: R, R1, R2, Ra, R35O ... (not compound labels)
RGROUP_RE = re.compile(r'^R\d*[A-Za-z]?$')
# radiolabels written next to atoms: 18F, 11C, 125I, 99mTc ...
ISOTOPE_RE = re.compile(r'^(?:18F|11C|13C|14C|15N|17O|18O|2H|3H|32P|35S|123I|124I|125I|131I|64Cu|68Ga|89Zr|99mTc|177Lu)$')
# formula fragments read from structures: C8H17 (OCR: CsH17, CgH17), 8H, 17H
FORMULA_RE = re.compile(r'^(?:C\w{0,2}H\d+|\d{1,2}H)$')
# OCR confusions inside a digit position
_DIGIT_FIX = str.maketrans({'l': '1', 'I': '1', '|': '1', 'o': '0', 'O': '0', 'S': '5'})


def _fix_paren(t):
    """'(cl7)' -> '(c17)', '(clo)' -> '(c10)', '(CS)' -> '(C5)', '(c8_' -> '(c8)'.
    Returns (text, weak); weak = the number contains no real digit (e.g. '(CS)'), so only trust it in context."""
    m = re.fullmatch(r'\(([A-Za-z]{1,3}-?)([0-9lIoOS|]{1,4})([a-z]?)[)_\]]?', t)
    if not m or PAREN_RE.match(t):
        return t, False
    prefix, num, suf = m.groups()
    # the prefix may have swallowed a confusable digit: '(cl7)' = c + l7
    while len(prefix.rstrip('-')) > 1 and prefix[-1] in 'lIoOS|':
        num, prefix = prefix[-1] + num, prefix[:-1]
    fixed = num.translate(_DIGIT_FIX)
    if not fixed.isdigit():
        return t, False
    return f'({prefix}{fixed}{suf})', not re.search(r'\d', num)


def normalize(text):
    t = text.strip().strip(',:;').replace('’', "'").replace('”', '"').replace('″', '"')
    # OCR often reads the last I of a Roman numeral as l / | / 1
    if re.fullmatch(r'[IVXl|1]+', t) and re.search('[VX]', t):
        t = re.sub(r'[l|1]', 'I', t)
    elif re.fullmatch(r'[Il|]{2,}', t):          # "Il", "lI", "||" -> "II" (digits like "11" stay)
        t = 'I' * len(t)
    if t.startswith('('):
        fixed, weak = _fix_paren(t)
        if not weak:
            t = fixed
    t = re.sub(r'-\|$', '-I', t)                  # 'd-4a-|' -> 'd-4a-I'
    return t


def _is_core_label(t):
    if ROMAN_RE.match(t):
        return len(t) >= 2 or t in ('I', 'V', 'X')
    if ATOM_RE.match(t) and (re.search(r'[H\d]', t)):
        return False
    if RGROUP_RE.match(t) or ISOTOPE_RE.match(t) or FORMULA_RE.match(t):
        return False
    if ACRONYM_RE.match(t):
        return len(t) >= 6 and not ATOM_RE.match(t.replace('-', ''))
    if PAREN_RE.match(t):                         # (I), (Ia), (12): the Roman part must be a real numeral, not '(Cl)'
        inner = t[1:-1].rstrip('abcdefghijklmnopqrstuvwxyz')
        return inner.isdigit() or (bool(ROMAN_RE.match(inner)) and (len(inner) >= 2 or inner in 'IVX'))
    return bool(LABEL_RE.match(t) or PAREN_ALNUM_RE.match(t) or STEREO_RE.match(t))


def is_label(text):
    t = normalize(text)
    if not t or len(t) > 24:
        return False
    m = PRIME_RE.match(t)                         # E-1.17"' : judge the part before the primes
    if m:
        return _is_core_label(m.group(1))
    return _is_core_label(t)


def signature(text):
    """Label style on a page: digits collapsed and primes dropped, '(c22)' -> '(C9)', "E-1.17'" -> 'E-9.9', 'XXIV' -> 'ROMAN'."""
    t = normalize(text).rstrip('\'"')
    if ROMAN_RE.match(t):
        return 'ROMAN'
    if t.startswith('(') and t.endswith(')') and ROMAN_RE.match(t[1:-1].rstrip('abcdefghijklmnopqrstuvwxyz') or '-'):
        return '(ROMAN)'
    return re.sub(r'\d+', '9', t).upper()


def is_strong(text):
    """Unambiguous label: Roman numeral, (I)/(Ia), or letters+digits. Plain numbers ("80") are not strong."""
    t = normalize(text)
    return is_label(t) and not re.fullmatch(r'\d+', t)


def is_junk(text):
    """Clearly NOT a label: empty, or a chemical annotation such as HCl, CH3, NH2 (OCR may read HCl as HCI).
    Deliberately narrow: unusual but valid labels (A, 3aa, A-1-1, E3.1, (IV)) must never be dropped."""
    t = normalize(text)
    return not t or bool(ATOM_RE.match(t) and 'H' in t)


def _read(ocr, arr, min_conf):
    """-> [(x1, y1, x2, y2, text)] in the pixel coordinates of `arr`."""
    if hasattr(ocr, 'readtext'):                       # easyocr.Reader
        words = []
        for poly, text, conf in ocr.readtext(arr, detail=1, paragraph=False):
            if conf < min_conf:
                continue
            xs, ys = [p[0] for p in poly], [p[1] for p in poly]
            words.append((min(xs), min(ys), max(xs), max(ys), text))
        return words
    return list(ocr(arr))


def _page_words(ocr, rgb, min_conf=0.2, margin=40):
    """-> list of (x1, y1, x2, y2, text) in original-image pixels.
    A white margin is added first: captions touching the scan edge are otherwise missed."""
    padded = np.pad(rgb, ((margin, margin), (margin, margin), (0, 0)), constant_values=255)
    return [(x1 - margin, y1 - margin, x2 - margin, y2 - margin, t)
            for x1, y1, x2, y2, t in _read(ocr, padded, min_conf)]


def _crop_words(ocr, rgb, box, scale=3, margin=60, min_conf=0.5):
    """OCR of an upscaled crop of `box` (x1, y1, x2, y2 in px); words in original-image pixels."""
    from PIL import Image
    H, W = rgb.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in (max(box[0], 0), max(box[1], 0), min(box[2], W), min(box[3], H)))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return []
    crop = Image.fromarray(rgb[y1:y2, x1:x2]).resize(((x2 - x1) * scale, (y2 - y1) * scale), Image.LANCZOS)
    arr = np.pad(np.asarray(crop), ((margin, margin), (margin, margin), (0, 0)), constant_values=255)
    return [(x1 + (a - margin) / scale, y1 + (b - margin) / scale, x1 + (c - margin) / scale, y1 + (d - margin) / scale, t)
            for a, b, c, d, t in _read(ocr, arr, min_conf)]


def _repair_truncated(ocr, rgb, text, box):
    """Model read "A-1-" for a box that says "A-1-2": re-read the box upscaled, accept a longer reading that extends it."""
    H, W = rgb.shape[:2]
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    for pad, scale in ((0.0, 3), (0.0, 2), (0.15, 3)):
        for _, _, _, _, t in _crop_words(ocr, rgb, (x1 - pad * w, y1 - pad * h, x2 + pad * w, y2 + pad * h),
                                         scale=scale, min_conf=0.3):
            t = normalize(t)
            if len(t) > len(text) and t.startswith(text) and is_label(t):
                return t
    return text


SINGLE_LETTER_RE = re.compile(r'^[A-Z]$')


def fix_text(text, letter_page=False):
    """Common OCR slips in labels and R-group captions:
    '(R= ~0-CH;' -> '(R= -O-CH3)', 'CHz' -> 'CH2', 'NOz' -> 'NO2', 'AczO' -> 'Ac2O', 'Il-13' -> 'II-13',
    and '0' -> 'O' on pages whose labels are single letters."""
    t = re.sub(r'\s+', ' ', text.strip())
    t = t.replace('~', '-')
    t = re.sub(r'(?<=C[HF])[;:](?=[\s)_,]|$)', '3', t)          # CH; CF; -> CH3 CF3
    t = re.sub(r'(?<=CH)y(?=[\s)_,]|$)', '3', t)                # CHy -> CH3
    t = re.sub(r'(?<=CH)z(?=[\s)_,]|$)', '3', t)                # terminal CHz -> CH3 (a methyl end group)
    t = re.sub(r'(?<=[CHNO])z(?=[\s)_,\-]|$)', '2', t)          # CHz- (chain) NOz -> CH2 NO2
    t = re.sub(r'(?<=Ac)z(?=O)', '2', t)                        # AczO -> Ac2O
    t = re.sub(r'(?<=[(=\s-])0(?=-)', 'O', t)                   # -0-CH3 -> -O-CH3
    t = t.rstrip('_;:, ')
    if t.startswith('(') and t.count('(') > t.count(')'):
        t += ')'
    t = re.sub(r'^I[lL](?=-\d)', 'II', t)                       # Il-13 / IL-13 -> II-13
    if letter_page and t == '0':
        t = 'O'
    return t


def _quality(t, letter_page=False):
    """How much a text looks like a finished label: 0 (empty/garbled) .. 5 (well-formed label)."""
    if not t:
        return 0
    s = 0
    if is_label(t) or (letter_page and SINGLE_LETTER_RE.match(t)) \
            or re.fullmatch(r'[ivx]{1,4}', t) or re.fullmatch(r'\d{1,3}\.\d{1,3}', t):   # i, ii, 4.12 are valid labels
        s += 3
    if t.count('(') == t.count(')'):
        s += 1
    if not re.search(r'[~_;:|]', t):
        s += 1
    if re.fullmatch(r'\(R\s*=\s*\S.*\)', t):                    # (R= CH3), (R = -O-CH2-Phenyl)
        s += 2
    return s


def _line_text(words):
    """Join OCR words of a crop in reading order (rows by y, then x)."""
    if not words:
        return ''
    words = sorted(words, key=lambda w: (w[1] + w[3]) / 2)
    rows, cur = [], [words[0]]
    for w in words[1:]:
        last = cur[-1]
        if (w[1] + w[3]) / 2 - (last[1] + last[3]) / 2 < 0.5 * max(last[3] - last[1], 1):
            cur.append(w)
        else:
            rows.append(cur)
            cur = [w]
    rows.append(cur)
    return ' '.join(' '.join(w[4] for w in sorted(r, key=lambda w: w[0])) for r in rows)


def _reread(ocr, rgb, box, letter_page=False):
    """Re-read a label box on upscaled crops; returns the best-looking reading ('' if none)."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    reads = []
    for pad, scale in ((0.0, 3), (0.15, 3), (0.3, 2)):
        words = _crop_words(ocr, rgb, (x1 - pad * w, y1 - pad * h, x2 + pad * w, y2 + pad * h),
                            scale=scale, min_conf=0.2)
        reads += [fix_text(normalize(wd[4]), letter_page) for wd in words]
        reads.append(fix_text(normalize(_line_text(words)), letter_page))
    reads = [r for r in dict.fromkeys(reads) if r]
    if not reads:
        return ''
    return max(reads, key=lambda r: (_quality(r, letter_page), len(r)))


def _gap(a, b):
    """distance between two boxes (0 if they touch/overlap)"""
    dx = max(b[0] - a[2], 0, a[0] - b[2])
    dy = max(b[1] - a[3], 0, a[1] - b[3])
    return (dx * dx + dy * dy) ** 0.5


def _iou(a, b):
    w, h = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    i = w * h
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def _remove(bboxes, corefs, drop):
    """remove boxes `drop` (indices) and every coref pair that uses them; indices are remapped"""
    if not drop:
        return bboxes, corefs
    remap, kept = {}, []
    for i, b in enumerate(bboxes):
        if i not in drop:
            remap[i] = len(kept)
            kept.append(b)
    return kept, [[remap[m], remap[t]] for m, t in corefs if m in remap and t in remap]


def clean_model_labels(bboxes, corefs, rgb, ocr, reread=True):
    """
    Clean the model's own [Idt] boxes before the fallback runs:
      a. re-read boxes whose text is empty or garbled ('', '(IIE', '(G)', 'A-') and fix OCR slips (CHz -> CH2);
      b. drop junk ('HCl', 'CH3'; a lone 'H' is kept on pages whose labels are letters A, B, C ...)
         and degenerate empty boxes;
      c. remove duplicates: the model sometimes emits the same label twice (IoU > 0.5), e.g. 'E-211' and
         'E-2.11' - keep the more complete reading (OCR drops characters such as dots, it rarely invents
         them), then the one closer to its molecule.
    """
    H, W = rgb.shape[:2]
    px = lambda b: (b['bbox'][0] * W, b['bbox'][1] * H, b['bbox'][2] * W, b['bbox'][3] * H)
    text = lambda b: ' '.join(b.get('text') or [])
    idt = [i for i, b in enumerate(bboxes) if b['category_id'] == IDT]
    letters = sum(1 for i in idt if SINGLE_LETTER_RE.match(normalize(text(bboxes[i]))))
    letter_page = letters >= 2

    drop = set()
    for i in idt:
        b = bboxes[i]
        t = fix_text(normalize(text(b)), letter_page)
        if reread and ocr is not None and _quality(t, letter_page) < 4:
            new = _reread(ocr, rgb, px(b), letter_page)
            lost = sum(t.count(c) - new.count(c) for c in '.-') > 0     # a re-read must not drop a dot or hyphen
            if _quality(new, letter_page) > _quality(t, letter_page) and not (t and lost):
                t = new
                b['source'] = 'reread'
        if t != text(b):
            b['text'] = [t]
        x1, y1, x2, y2 = px(b)
        if not t and (x2 - x1 < 4 or y2 - y1 < 4):
            drop.add(i)                                       # empty sliver at the image edge
        elif t and is_junk(t) and not (letter_page and SINGLE_LETTER_RE.match(t)):
            drop.add(i)

    mol_of = {t: m for m, t in corefs}
    keep = [i for i in idt if i not in drop]
    for a_i, a in enumerate(keep):
        for c in keep[a_i + 1:]:
            if a in drop or c in drop or _iou(px(bboxes[a]), px(bboxes[c])) <= 0.5:
                continue
            def rank(i):                                          # better text first, then the closer molecule
                d = _gap(px(bboxes[i]), px(bboxes[mol_of[i]])) if i in mol_of else float('inf')
                return (_quality(text(bboxes[i]), letter_page), len(text(bboxes[i])), -d, -i)
            drop.add(min((a, c), key=rank))
    return _remove(bboxes, corefs, drop)


def _components(rgb, box, pad=2):
    """Connected dark components of a label box: [(x1, y1, x2, y2)] in crop coordinates, plus the crop height."""
    from scipy import ndimage
    H, W = rgb.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in (max(box[0] - pad, 0), max(box[1] - pad, 0),
                                              min(box[2] + pad, W), min(box[3] + pad, H)))
    if x2 - x1 < 6 or y2 - y1 < 6:
        return [], 0
    gray = rgb[y1:y2, x1:x2].mean(axis=2)
    ink = gray < max(min(gray.min() + 0.5 * (gray.max() - gray.min()), 200), 60)   # simple adaptive threshold
    lab, n = ndimage.label(ink, structure=np.ones((3, 3)))
    comps = []
    for sl in ndimage.find_objects(lab):
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if h * w >= 4:
            comps.append((sl[1].start, sl[0].start, sl[1].stop, sl[0].stop))
    return sorted(comps), y2 - y1


def restore_punctuation(text, rgb, box):
    """
    Recover a dot or hyphen the OCR dropped from a short compound label ('412' -> '4.12', 'E3.1' -> 'E-3.1',
    'E44.1"' -> 'E-4.1"'). The label box is split into glyph blobs: tall blobs are letters/digits, a small blob on
    the baseline is a dot, a flat blob at mid height is a hyphen. A separator is inserted only when the blobs
    show it AND the number of tall blobs agrees with the number of characters of the text (otherwise the glyphs
    touch or the reading is unreliable and the text is left untouched).
    """
    m = re.fullmatch(r"([A-Za-z]{0,4}[\dA-Za-z.\-]*?)([\'\"]*)", text)
    if not m or len(text) > 12 or not re.search(r'\d', text):
        return text
    core, primes = m.groups()
    comps, ch = _components(rgb, box)
    if len(comps) < 2:
        return text
    top = max(c[3] - c[1] for c in comps)                    # height of the tallest glyph
    seps, tall = [], []
    for c in comps:
        w, h, cy = c[2] - c[0], c[3] - c[1], (c[1] + c[3]) / 2
        base = max(cc[3] for cc in comps if cc[3] - cc[1] >= 0.6 * top)    # baseline of the glyphs
        if h >= 0.6 * top:
            tall.append(c)
        elif h <= 0.3 * top and w <= 0.45 * top and abs(c[3] - base) <= 0.15 * top:
            seps.append(((c[0] + c[2]) / 2, '.'))
        elif h <= 0.3 * top and w >= 0.25 * top and 0.3 * top <= (base - cy) <= 0.75 * top:
            seps.append(((c[0] + c[2]) / 2, '-'))
    alnum = [ch_ for ch_ in core if ch_.isalnum()]
    # separator position = number of tall glyphs to its left
    found = [(sum(1 for t in tall if (t[0] + t[2]) / 2 < x), kind) for x, kind in seps]
    have = [(sum(1 for ch_ in core[:i] if ch_.isalnum()), ch_) for i, ch_ in enumerate(core) if ch_ in '.-']
    replaced = False
    if len(alnum) == len(tall) + 1 and [k for _, k in found].count('-') == 1 and '-' not in core:
        # the OCR read the hyphen as a letter/digit: 'E44.1' = E, hyphen, 4, ., 1  -> drop that character
        pos = [p for p, k in found if k == '-'][0]
        if 0 < pos < len(alnum):
            alnum = alnum[:pos] + alnum[pos + 1:]
            have = [(p - 1 if p > pos else p, k) for p, k in have]
            replaced = True
    if len(alnum) != len(tall):
        return text                                           # touching glyphs / unreliable reading
    want = sorted(set(found) | set(have))
    if not replaced and want == sorted(have):
        return text                                           # nothing missing
    return _build(alnum, want) + primes


def _build(alnum, seps):
    """rebuild 'E-4.1' from alnum chars and (position, separator) pairs"""
    out = []
    for i, ch_ in enumerate(alnum):
        for p, k in seps:
            if p == i:
                out.append(k)
        out.append(ch_)
    out += [k for p, k in seps if p >= len(alnum)]
    return ''.join(out)


def apply_page_dot_style(bboxes):
    """
    A dot fused with the previous digit cannot be seen by blob analysis ('4.12' read as '412'). If at least two
    labels of the page have the form '<prefix>.<n>' (4.1, 4.10, 4.2 ...), a label '<prefix><digits>' without a dot
    on the same page is written '<prefix>.<digits>'. Returns the number of labels changed.
    """
    texts = {i: b['text'][0] for i, b in enumerate(bboxes)
             if b['category_id'] == IDT and b.get('text') and len(b['text']) == 1}
    prefixes = Counter(m.group(1) for t in texts.values() for m in [re.fullmatch(r"(\D{0,3}\d)\.\d{1,3}", t)] if m)
    n = 0
    for pre, cnt in prefixes.items():
        if cnt < 2:
            continue
        for i, t in texts.items():
            m = re.fullmatch(re.escape(pre) + r"(\d{1,2})", t)
            if m:
                bboxes[i]['text'] = [pre + '.' + m.group(1)]
                bboxes[i]['punct_fixed'] = True
                n += 1
    return n


def add_missing_labels(predictions, image, ocr, max_below=0.6, x_margin=0.15, repair=True, unpaired=True, punctuation=True):
    """
    predictions: output of MolDetect.predict_image(..., coref=True)
    image:       the same PIL image that was passed to the model
    unpaired:    also return labels in the page's dominant style that have no detected molecule
                 (as [Idt] boxes without a coref pair)
    punctuation: recover dots/hyphens the OCR dropped ('412' -> '4.12', 'E3.1' -> 'E-3.1')
    Returns a new predictions dict with wrong pairs removed and missing [Idt] boxes added.
    """
    out = _add_missing_labels(predictions, image, ocr, max_below, x_margin, repair, unpaired)
    if punctuation:
        rgb = np.asarray(image.convert('RGB'))
        H, W = rgb.shape[:2]
        for b in out['bboxes']:
            if b['category_id'] == IDT and b.get('text') and len(b['text']) == 1:
                x1, y1, x2, y2 = b['bbox']
                try:
                    new = restore_punctuation(b['text'][0], rgb, (x1 * W, y1 * H, x2 * W, y2 * H))
                except Exception:
                    continue
                if new != b['text'][0]:
                    b['text'] = [new]
                    b['punct_fixed'] = True
        apply_page_dot_style(out['bboxes'])
    return out


def _add_missing_labels(predictions, image, ocr, max_below=0.6, x_margin=0.15, repair=True, unpaired=True, punctuation=True):
    """
    predictions: output of MolDetect.predict_image(..., coref=True)
    image:       the same PIL image that was passed to the model
    unpaired:    also return labels in the page's dominant style that have no detected molecule
                 (as [Idt] boxes without a coref pair)
    Returns a new predictions dict with wrong pairs removed and missing [Idt] boxes added.
    """
    W, H = image.size
    rgb = np.asarray(image.convert('RGB'))
    bboxes = [dict(b) for b in predictions['bboxes']]
    corefs = [list(c) for c in predictions['corefs']]

    # 1. clean the model's own labels: re-read garbled/empty ones, drop junk ("HCl") and duplicates
    bboxes, corefs = clean_model_labels(bboxes, corefs, rgb, ocr, reread=repair)
    px = lambda b: (b['bbox'][0] * W, b['bbox'][1] * H, b['bbox'][2] * W, b['bbox'][3] * H)

    # 1b. truncated model readings ("A-1-") -> re-read the box
    if repair:
        for b in bboxes:
            t = ' '.join(b.get('text') or [])
            if b['category_id'] == IDT and len(t) >= 2 and t[-1] in '-.' and len(b['text']) == 1:
                new = _repair_truncated(ocr, rgb, t, px(b))
                if new != t:
                    b['text'] = [new]
                    b['source'] = 'repaired'

    mols = {i: px(b) for i, b in enumerate(bboxes) if b['category_id'] == MOL}
    idts = [px(b) for b in bboxes if b['category_id'] == IDT]
    labelled = {m for m, _ in corefs}
    todo = [i for i in mols if i not in labelled]
    if not todo:
        return {'bboxes': bboxes, 'corefs': corefs}

    def inside_existing(x1, y1, x2, y2):
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        return any(a <= cx <= c and b <= cy <= d for a, b, c, d in idts)

    def inside_molecule(x1, y1, x2, y2):
        # the bottom edge is allowed: the model box sometimes swallows the caption
        cx, cy, lh = (x1 + x2) / 2, (y1 + y2) / 2, max(y2 - y1, 1)
        return any(a <= cx <= c and b <= cy <= d - lh for a, b, c, d in mols.values())

    # 2. label-like OCR tokens that are not inside a molecule or an existing [Idt]
    words = _page_words(ocr, rgb)

    def in_text(x1, y1, x2, y2):
        """part of a sentence or reagent line: '23' under 'general procedure', '(i)' before '[18F]Bu4N'"""
        lh = max(y2 - y1, 1)
        for a, b, c, d, t in words:
            if (a, b, c, d) == (x1, y1, x2, y2):
                continue
            same_line = min(y2, d) - max(y1, b) > 0.5 * min(lh, max(d - b, 1))
            if same_line and max(a - x2, x1 - c) < 0.8 * lh and len(re.findall(r'[A-Za-z]', t)) >= 3:
                return True
            stacked = min(x2, c) - max(x1, a) > 0
            if stacked and max(b - y2, y1 - d) < 0.8 * lh and re.search(r'[a-z]{3,}', t):
                return True
        return False

    cands, weak = [], []
    for x1, y1, x2, y2, t in words:
        if inside_existing(x1, y1, x2, y2) or inside_molecule(x1, y1, x2, y2) or in_text(x1, y1, x2, y2):
            continue
        if is_label(t):
            cands.append((fix_text(normalize(t)), (x1, y1, x2, y2)))
        else:
            fixed, is_weak = _fix_paren(normalize(t)) if t.strip().startswith('(') else (t, False)
            if is_weak and is_label(fixed):          # '(CS)' -> '(C5)': kept only if the page uses that style
                weak.append((fixed, (x1, y1, x2, y2)))

    # page label style: if most labels on the page look like '(c9)', a bare '5' or '(C5)' read from 'CS' is judged by it
    sigs = Counter(signature(' '.join(b['text'])) for b in bboxes
                   if b['category_id'] == IDT and b.get('text') and is_label(' '.join(b['text'])))
    sigs.update(signature(t) for t, _ in cands)
    dominant, n_dom = sigs.most_common(1)[0] if sigs else (None, 0)
    if n_dom < 3 or n_dom < 0.5 * sum(sigs.values()):
        dominant = None
    if dominant is not None:
        cands += [c for c in weak if signature(c[0]) == dominant]
        if dominant != '9':                           # bare numbers are structure text on such pages
            cands = [c for c in cands if signature(c[0]) != '9']
    used_c = set()

    def attach(mol, cand, source='fallback'):
        t, (x1, y1, x2, y2) = cand
        bboxes.append({'category': '[Idt]', 'category_id': IDT, 'text': [t],
                       'bbox': (float(x1) / W, float(y1) / H, float(x2) / W, float(y2) / H),
                       'score': -1, 'source': source})
        if mol is not None:
            corefs.append([mol, len(bboxes) - 1])
        idts.append((x1, y1, x2, y2))

    def assign(molecules, cost_fn):
        """one-to-one, cheapest pair first; returns molecules that are still unlabelled"""
        pairs = []
        for i in molecules:
            for j, (t, box) in enumerate(cands):
                if j in used_c:
                    continue
                c = cost_fn(mols[i], box, t)
                if c is not None:
                    pairs.append((c, i, j))
        done = set()
        for c, i, j in sorted(pairs):
            if i in done or j in used_c:
                continue
            done.add(i)
            used_c.add(j)
            attach(i, cands[j])
        return [i for i in molecules if i not in done]

    def below(mol, box, t):
        """caption below the molecule, horizontally aligned"""
        mx1, my1, mx2, my2 = mol
        x1, y1, x2, y2 = box
        mw, mh = max(mx2 - mx1, 1), max(my2 - my1, 1)
        cx, cy, lh = (x1 + x2) / 2, (y1 + y2) / 2, max(y2 - y1, 1)
        if not (mx1 - x_margin * mw <= cx <= mx2 + x_margin * mw):
            return None
        if not (my2 - 1.2 * lh <= cy <= my2 + max_below * mh):
            return None
        return abs(cy - my2) / mh + 0.5 * abs(cx - (mx1 + mx2) / 2) / mw

    todo = assign(todo, below)

    # 3. small captions are missed by page-wide OCR: re-read a 3x upscaled strip under each remaining molecule
    for i in todo:
        mx1, my1, mx2, my2 = mols[i]
        mw, mh = max(mx2 - mx1, 1), max(my2 - my1, 1)
        strip = (mx1 - x_margin * mw, my2 - 0.12 * mh, mx2 + x_margin * mw, my2 + 0.3 * mh)
        for x1, y1, x2, y2, t in _crop_words(ocr, rgb, strip):
            if not is_label(t) or inside_existing(x1, y1, x2, y2):
                continue
            if dominant not in (None, '9') and signature(t) == '9':
                continue
            if any(abs(x1 - a) < 0.5 * (x2 - x1) and abs(y1 - b) < 0.5 * (y2 - y1) for _, (a, b, _, _) in cands):
                continue                                  # already known from the page pass
            cands.append((normalize(t), (x1, y1, x2, y2)))
    todo = assign(todo, below)

    # 4. last resort: a strong token (XXIX, (Ia)) right beside the molecule, e.g. in a corner of the page
    def beside(mol, box, t):
        if not is_strong(t):
            return None
        mx1, my1, mx2, my2 = mol
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        dx = max(mx1 - cx, 0, cx - mx2)
        dy = max(my1 - cy, 0, cy - my2)
        size = max(mx2 - mx1, my2 - my1, 1)
        if (dx == 0 and dy == 0) or max(dx, dy) > 0.35 * size:
            return None                                   # inside the molecule or too far away
        return (dx * dx + dy * dy) ** 0.5 / size

    assign(todo, beside)

    # 5. labels in the page's own style whose molecule the model did not detect (dense grids):
    #    keep them as unpaired [Idt] boxes - the label is still found, only the molecule link is missing
    if unpaired and dominant is not None:
        for j, c in enumerate(cands):
            if j not in used_c and is_strong(c[0]) and signature(c[0]) == dominant \
                    and not inside_existing(*c[1]):
                used_c.add(j)
                attach(None, c, source='fallback-unpaired')
    return {'bboxes': bboxes, 'corefs': corefs}
