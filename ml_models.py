"""
Home for the three ML models used alongside the rule-based engine in
analyzer.py:

  1. Sentence-BERT (SBERT)       -> resume <-> job description matching
  2. spaCy (NER + PhraseMatcher) -> structured extraction of skills,
     education, work experience, and certifications
  3. XGBoost                    -> learned ATS-parseability score (0-100)
Every model degrades gracefully to None/empty if its package or model file
isn't installed, and analyzer.py falls back to rule-based behaviour.
"""

import os
import re
import logging

import numpy as np

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
ATS_MODEL_PATH = os.path.join(MODEL_DIR, "ats_model.joblib")

# 1. Sentence-BERT — resume <-> job description semantic similarity
# (This also backs analyzer.jd_match_score - kept here so all three "model"
# integrations live in one place and are easy to find/swap.)

try:
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import cos_sim
except ImportError:
    SentenceTransformer = None
    cos_sim = None
_sbert_model = None
_sbert_load_attempted = False

def get_sbert_model():
    """Lazily loads all-MiniLM-L6-v2 once per process. Only attempts once —
    if the download/load fails (no internet, blocked registry, disk issue,
    etc.) it's cached as unavailable so every subsequent request falls back
    to TF-IDF immediately instead of retrying and raising on every call."""
    global _sbert_model, _sbert_load_attempted
    if _sbert_load_attempted:
        return _sbert_model
    _sbert_load_attempted = True
    if SentenceTransformer is None:
        return None
    try:
        logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
        _sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        _sbert_model = None
    return _sbert_model
def sbert_similarity(text_a, text_b):
    """Cosine similarity in [0, 1] between two texts using SBERT embeddings.
    Returns None if sentence-transformers isn't installed (caller should
    fall back to TF-IDF cosine similarity)."""
    model = get_sbert_model()
    if model is None:
        return None
    emb_a = model.encode(text_a)
    emb_b = model.encode(text_b)
    return float(cos_sim(emb_a, emb_b)[0][0])

# 2. spaCy — structured extraction: education, experience, certifications

try:
    import spacy
    from spacy.matcher import PhraseMatcher
except ImportError:
    spacy = None
    PhraseMatcher = None
_spacy_nlp = None
_spacy_load_attempted = False

DEGREES = [
    "bachelor of science", "bachelor of arts", "bachelor of engineering",
    "bachelor of technology", "master of science", "master of arts",
    "master of business administration", "master of engineering",
    "b.sc", "b.s.", "bs", "b.e.", "be", "b.tech", "btech", "b.a.", "ba",
    "m.sc", "m.s.", "ms", "m.e.", "me", "m.tech", "mtech", "m.a.", "ma",
    "mba", "phd", "ph.d.", "doctorate", "associate degree", "diploma",
]

CERT_KEYWORDS = [
    "aws certified", "azure certified", "google cloud certified",
    "pmp", "project management professional", "scrum master", "csm",
    "cissp", "comptia", "ccna", "ccnp", "ckad", "cka", "terraform associate",
    "certified kubernetes", "six sigma", "itil", "oracle certified",
    "microsoft certified", "certified ethical hacker", "ceh",
    "salesforce certified", "tableau certified", "databricks certified",
    "certified data scientist", "certified scrum product owner", "cspo",
]

_JOB_TITLE_HINTS = re.compile(
    r"\b(engineer|developer|manager|analyst|scientist|designer|intern|"
    r"consultant|architect|lead|director|specialist|administrator|"
    r"associate|coordinator|officer)\b",
    re.I,
)

_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")

def get_spacy_nlp():
    """Lazily loads the small English pipeline. Only attempts once per
    process (subsequent calls just return whatever the first attempt got,
    including None) so a missing model doesn't retry-and-fail on every
    request."""
    global _spacy_nlp, _spacy_load_attempted
    if _spacy_load_attempted:
        return _spacy_nlp
    _spacy_load_attempted = True
    if spacy is None:
        return None
    try:
        _spacy_nlp = spacy.load("en_core_web_sm", disable=["lemmatizer"])
    except OSError:
        _spacy_nlp = None
    return _spacy_nlp
def _build_matcher(nlp, phrases):
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(p) for p in phrases]
    matcher.add("PHRASES", patterns)
    return matcher
def extract_entities_spacy(text, skill_vocab=None, experience_text=None, education_text=None):
    """Runs spaCy NER + PhraseMatcher over the resume to pull out:
      - education: [{"degree": ..., "institution": ..., "year": ...}]
      - experience: [{"title": ..., "organization": ..., "dates": ...}]
      - certifications: [str, ...]
      - organizations: raw ORG entities spaCy found (used as a sanity list)
    Returns None if spaCy / the English model isn't available so callers
    can skip this enrichment entirely.
    """
    nlp = get_spacy_nlp()
    if nlp is None:
        return None
    doc = nlp(text[:20000])  # cap for speed/memory on very long documents
    orgs = list(dict.fromkeys(ent.text.strip() for ent in doc.ents if ent.label_ == "ORG"))
    dates = list(dict.fromkeys(ent.text.strip() for ent in doc.ents if ent.label_ == "DATE"))

    # --- Certifications: phrase-match against a curated list ---
    cert_matcher = _build_matcher(nlp, CERT_KEYWORDS)
    cert_hits = cert_matcher(doc)
    certifications = sorted({doc[s:e].text.strip() for _, s, e in cert_hits}, key=str.lower)

    # --- Education: line-level pass looking for a degree phrase, then
    # borrowing the nearest ORG entity + year on that line/window as the
    # institution/year ---
    degree_matcher = _build_matcher(nlp, DEGREES)
    education = []
    seen_degrees = set()
    edu_source = education_text if education_text and education_text.strip() else text
    for line in edu_source.splitlines():
        line = line.strip()
        if not line:
            continue
        line_doc = nlp.make_doc(line)
        hits = degree_matcher(line_doc)
        if not hits:
            continue
        for _, s, e in hits:
            degree_text = line_doc[s:e].text
            key = degree_text.lower()
            if key in seen_degrees:
                continue
            seen_degrees.add(key)
            year_match = _YEAR_RE.search(line)
            # institution = an ORG entity spaCy recognises on this same
            # line, excluding anything that overlaps the degree phrase
            # itself (the small model sometimes tags "Bachelor of Science
            # in X" as an ORG, which would otherwise get picked as both the
            # degree and its own "institution").
            institution = None
            line_ents = nlp(line).ents
            for ent in line_ents:
                if ent.label_ == "ORG" and ent.text.strip().lower() != degree_text.strip().lower() \
                        and degree_text.strip().lower() not in ent.text.strip().lower():
                    institution = ent.text.strip()
                    break
            education.append({
                "degree": degree_text.strip(),
                "institution": institution,
                "year": year_match.group(0) if year_match else None,
            })

    # --- Experience: lines that look like a job-title line (contains a
    # role-ish word and isn't a bullet) paired with the nearest ORG entity
    # and DATE span found on the same or adjacent line ---
    experience = []
    exp_source = experience_text if experience_text and experience_text.strip() else text
    lines = [l.strip() for l in exp_source.splitlines()]
    for i, line in enumerate(lines):
        words = line.split()
        if not line or not (2 <= len(words) <= 10):
            continue
        if re.match(r"^[•\-\*\u2022➤◆■●▪]", line):
            continue
        if line.endswith("."):
            continue  # sentence-like lines (summaries, bullets) aren't title lines
        if not _JOB_TITLE_HINTS.search(line):
            continue
        # Require Title Case-ish styling (most words capitalised) so we
        # don't sweep in lowercase prose sentences that merely contain a
        # role-ish word (e.g. "...working with our engineer team...").
        capitalised = sum(1 for w in words if w[:1].isupper())
        if capitalised < max(1, len(words) - 2):
            continue
        # Common resume pattern: "Job Title, Company Name" on one line —
        # trust that literal split over NER, which can mis-tag the title
        # itself as an ORG.
        org = None
        title_for_display = line
        if "," in line:
            head, _, tail = line.partition(",")
            tail = tail.strip()
            if tail and _JOB_TITLE_HINTS.search(head):
                org = tail
                title_for_display = head.strip()
        window = " ".join(lines[max(0, i - 1):i + 2])
        window_doc = nlp(window)
        if not org:
            org = next((ent.text.strip() for ent in window_doc.ents if ent.label_ == "ORG"), None)
        date_span = next((ent.text.strip() for ent in window_doc.ents if ent.label_ == "DATE"), None)
        if org or date_span:
            experience.append({
                "title": title_for_display,
                "organization": org,
                "dates": date_span,
            })

    # de-dup experience entries by (title, organization)
    dedup, seen = [], set()
    for e in experience:
        k = (e["title"].lower(), (e["organization"] or "").lower())
        if k not in seen:
            seen.add(k)
            dedup.append(e)

    # --- Skills via PhraseMatcher over the same skill vocabulary the rest
    # of the app uses, purely as an NER-based cross-check / extra recall on
    # top of the FlashText pass already done in analyzer.py ---
    ner_skills = set()
    if skill_vocab:
        skill_matcher = _build_matcher(nlp, list(skill_vocab))
        for _, s, e in skill_matcher(doc):
            ner_skills.add(doc[s:e].text)
    return {
        "education": education[:8],
        "experience": dedup[:10],
        "certifications": certifications,
        "organizations": orgs[:15],
        "ner_skills": sorted(ner_skills),
    }

# 3. XGBoost — learned ATS-parseability score
# No public labeled "did this resume pass an ATS" dataset exists, so this
# model trains on synthetic feature vectors labeled from the same
# structural heuristics the rule-based checker uses. This lets it learn
# non-linear interactions (e.g. tables are worse when headers are also
# non-standard) instead of flat point deductions. Swap in a real labeled
# dataset here later if one becomes available.

ATS_FEATURE_NAMES = [
    "word_count",
    "has_email",
    "has_phone",
    "has_linkedin",
    "num_core_sections",       # experience/education/skills/summary found, 0-4
    "has_tables",
    "has_images",
    "gap_line_ratio",          # fraction of lines with big internal gaps (multi-column signal)
    "has_unusual_bullets",
    "matched_standard_headers",  # of experience/education/skills, 0-3
    "skills_count",
    "bullet_count",
    "quantified_bullet_ratio",
]

try:
    import joblib
except ImportError:
    joblib = None
try:
    import xgboost as xgb
except ImportError:
    xgb = None
_ats_model = None
_ats_model_load_attempted = False

def build_ats_feature_vector(
    word_count, has_email, has_phone, has_linkedin, num_core_sections,
    has_tables, has_images, gap_line_ratio, has_unusual_bullets,
    matched_standard_headers, skills_count, bullet_count,
    quantified_bullet_ratio,
):
    return np.array([[
        word_count, int(has_email), int(has_phone), int(has_linkedin),
        num_core_sections, int(has_tables), int(has_images), gap_line_ratio,
        int(has_unusual_bullets), matched_standard_headers, skills_count,
        bullet_count, quantified_bullet_ratio,
    ]], dtype=float)
def _synthetic_ats_training_data(n=6000, seed=42):
    """Generates random-but-plausible resumes-as-feature-vectors plus a
    noisy proxy label built from domain heuristics, for training only."""
    rng = np.random.default_rng(seed)
    word_count = rng.integers(80, 1400, n)
    has_email = rng.integers(0, 2, n)
    has_phone = rng.integers(0, 2, n)
    has_linkedin = rng.integers(0, 2, n)
    num_core_sections = rng.integers(0, 5, n)
    has_tables = rng.integers(0, 2, n)
    has_images = rng.integers(0, 2, n)
    gap_line_ratio = rng.random(n) * 0.4
    has_unusual_bullets = rng.integers(0, 2, n)
    matched_standard_headers = rng.integers(0, 4, n)
    skills_count = rng.integers(0, 30, n)
    bullet_count = rng.integers(0, 40, n)
    quantified_bullet_ratio = rng.random(n)
    X = np.column_stack([
        word_count, has_email, has_phone, has_linkedin, num_core_sections,
        has_tables, has_images, gap_line_ratio, has_unusual_bullets,
        matched_standard_headers, skills_count, bullet_count,
        quantified_bullet_ratio,
    ]).astype(float)
    length_score = np.where(
        (word_count >= 350) & (word_count <= 900), 15,
        np.clip(15 - np.abs(word_count - 625) / 60, 0, 15),
    )
    y = (
        has_email * 12
        + has_phone * 6
        + has_linkedin * 4
        + num_core_sections * 6
        + length_score
        - has_tables * 18
        - has_images * 10
        - gap_line_ratio * 40
        - has_unusual_bullets * 6
        + matched_standard_headers * 8
        + np.clip(skills_count, 0, 15) * 1.2
        + np.clip(bullet_count, 0, 20) * 0.6
        + quantified_bullet_ratio * 10
        # interaction: tables AND non-standard headers compound badly
        - (has_tables * (matched_standard_headers < 2)) * 10
    )
    y = y + rng.normal(0, 4, n)  # label noise
    y = np.clip(y, 0, 100)
    return X, y
def train_ats_xgboost_model(save_path=ATS_MODEL_PATH):
    """Trains the XGBoost ATS-score regressor on synthetic data and saves
    it with joblib. Run this offline (see train_ats_model.py); analyzer.py
    only ever loads the saved model at request time."""
    if xgb is None or joblib is None:
        raise RuntimeError("xgboost and joblib must be installed to train the ATS model.")
    X, y = _synthetic_ats_training_data()
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
    )
    model.fit(X, y)
    joblib.dump({"model": model, "feature_names": ATS_FEATURE_NAMES}, save_path)
    return model
def get_ats_model():
    global _ats_model, _ats_model_load_attempted
    if _ats_model_load_attempted:
        return _ats_model
    _ats_model_load_attempted = True
    if joblib is None or not os.path.exists(ATS_MODEL_PATH):
        return None
    try:
        bundle = joblib.load(ATS_MODEL_PATH)
        _ats_model = bundle["model"]
    except Exception:
        _ats_model = None
    return _ats_model
def predict_ats_score(feature_vector):
    """Returns a 0-100 predicted ATS-parseability score, or None if the
    model isn't available (caller should rely on the rule-based risk_level
    alone in that case)."""
    model = get_ats_model()
    if model is None:
        return None
    try:
        pred = float(model.predict(feature_vector)[0])
        return round(max(0.0, min(100.0, pred)), 1)
    except Exception:
        return None
def ats_feature_importances():
    """Returns {feature_name: importance} for the trained model, or None."""
    model = get_ats_model()
    if model is None:
        return None
    try:
        importances = model.feature_importances_
        return {name: round(float(imp), 4) for name, imp in zip(ATS_FEATURE_NAMES, importances)}
    except Exception:
        return None