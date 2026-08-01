# SCANLINE — AI Resume Analyzer

A resume analyzer with an HTML/CSS/JS frontend and a Python (Flask) backend.
It scores a resume for ATS-friendliness, content quality, and skill coverage,
and — if you paste in a job description — a keyword/similarity match score
against that specific role.

**No API required.** The scoring engine runs entirely locally using
`scikit-learn` (TF-IDF + cosine similarity) and regex/rule-based NLP —
there's nothing to sign up for and no API key needed to use the app.

## Project structure

All files live together in one folder:

```
resume-analyzer/
├── app.py               Flask server + API routes
├── analyzer.py          Parsing, scoring, and suggestion logic
├── ml_models.py         SBERT / spaCy / XGBoost model integrations
├── train_ats_model.py   Offline script that trains ats_model.joblib
├── ats_model.joblib     Pre-trained XGBoost ATS-score model
├── requirements.txt
├── index.html
├── style.css
└── script.js
```

`app.py` serves `index.html`, `style.css`, and `script.js` directly, so a
single Flask process runs the whole app — no separate frontend server needed.

## Machine learning models used

On top of the rule-based scoring engine, SCANLINE uses three real ML models
(all in `ml_models.py`). Every one degrades gracefully to the rule-based
fallback if its package or model file isn't installed, so the app still
works end-to-end without any of them.

| Model | Role | Where it's used |
|---|---|---|
| **Sentence-BERT** (`all-MiniLM-L6-v2`, via `sentence-transformers`) | Semantic resume ↔ job-description matching — catches meaning overlap ("led a team of engineers" vs. "managed engineering staff") that keyword/TF-IDF overlap misses. Falls back to TF-IDF cosine similarity if unavailable. | `jd_match_score()` → the JD similarity score and required/preferred keyword match |
| **spaCy** (`en_core_web_sm`, NER + PhraseMatcher) | Structured extraction of **education** (degree / institution / year), **work experience** (title / organization / dates), and **certifications**, plus an NER-based cross-check pass on skills. | `score_resume()` → the "Extracted profile" card in the UI |
| **XGBoost** | A learned, non-linear ATS-parseability score (0-100) that sits alongside the existing rule-based Low/Medium/High risk checklist. Trained on synthetic feature vectors labeled from the same structural heuristics the rule-based checker uses (tables, images, multi-column layout, missing headers, etc.), so it can weigh interactions between them (e.g. tables are much worse when headers are *also* non-standard) instead of flat point deductions. | `analyze_ats_risk()` → "ML-predicted ATS pass score" shown in the ATS risk card |

Run `python3 train_ats_model.py` to retrain the XGBoost model (it writes
`ats_model.joblib` next to it); the server only ever loads that file, it
never trains anything at request time. See the comments above
`_synthetic_ats_training_data` in `ml_models.py` for why synthetic labels
are used (no public labeled ATS-outcome dataset exists) and what to change
if you later get real labeled data.

The `/api/health` endpoint reports which of the three models actually
loaded in your environment, e.g.:

```json
{"ml_models": {"sbert_jd_matching": true, "spacy_entity_extraction": true, "xgboost_ats_score": true}}
```

## Setup

From inside this folder:

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` installs spaCy's small English model directly from its
GitHub release wheel, so no separate `python -m spacy download` step is
needed. `ats_model.joblib` (the pre-trained XGBoost model) already ships in
this folder — no training step is required to run the app.


## Run

```bash
python3 app.py
```

Then open http://localhost:5000 in your browser. Don't open `index.html`
directly by double-clicking it — it needs to be served by Flask so it can
reach the `/api/analyze` endpoint. Always start it with `python3 app.py`.

## How scoring works

| Component | What it checks |
|---|---|
| **Structure score** | Contact info present, key sections found (Experience, Education, Skills, Summary), sensible resume length |
| **Content score** | Ratio of bullet points starting with strong action verbs, use of quantified achievements (numbers/%), overused buzzwords |
| **Skills score** | Coverage against a curated database of programming languages, frameworks, tools, and soft skills |
| **JD match** (optional) | TF-IDF cosine similarity between resume and job description, plus matched/missing keyword lists |

The overall score is a weighted blend of these components (weighted more
heavily toward JD match when a job description is supplied).

### Section-wise resume score

The resume is split into its Contact / Summary / Experience / Education /
Skills / Projects zones (by locating standard section headers), and each
zone gets its own 0-100 score — so you can see exactly which section is
dragging the resume down instead of just one overall number.

### Resume completeness score

A simple checklist-style score (0-100%) answering "is everything here?" —
email, phone, LinkedIn, each major section, enough listed skills, and at
least one quantified achievement. This is deliberately separate from the
quality-focused structure/content scores above.

### ATS risk analysis

Flags formatting choices that commonly break Applicant Tracking System
parsers: tables used for layout, embedded images/logos, likely multi-column
layouts, non-standard bullet/icon glyphs, and missing standard section
headers. Returns an overall Low/Medium/High risk level plus a tip for each
issue found. This is a heuristic proxy — it can only see what a parser could
extract, the same limitation a real ATS has.

### Personal info & bias-risk check

Scans for demographic details that are common on resume templates in some
regions — a photo, date of birth/age, marital status, nationality, gender,
religion — but that many employers hiring in the US, UK, Canada, and the EU
specifically don't want to see. Several of these are protected
characteristics, so some companies' ATS platforms and hiring policies are
configured to flag or auto-reject applications that include them rather
than take on discrimination risk. Purely advisory (Clean/Low/Medium/High) —
it never changes `structure_score`, `overall_score`, or any other existing
score.

### Highlighted resume

Renders your resume's own extracted text back on the page — not just
scores — with weak sentences/bullets marked in one color and keywords
matched against your skills (or the job description / target role, when
supplied) marked in another, so you can see exactly *where* an issue lives
instead of only reading it as a number. Keywords the job description or
target role calls for but that don't appear anywhere in your resume are
listed separately underneath (they obviously can't be highlighted inside
text that doesn't contain them).

### Duplicate content detection

Scans for redundancy that the section/content scores above wouldn't
otherwise catch:

- **Repeated skills** — the same skill listed twice in the Skills section
  (a copy-paste slip), or a skill mentioned so many times across the whole
  document that it looks like deliberate keyword stuffing.
- **Repeated sentences** — identical sentences appearing more than once,
  often a sign of a resume stitched together from a template or old draft.
- **Repeated bullet points** — bullets duplicated verbatim across different
  jobs/projects, plus near-identical bullets that differ only by a number
  (e.g. the same "migrated X to microservices, cutting latency by N%" claim
  reused for two different roles).

This is a reporting layer only — it doesn't change `structure_score`,
`content_score`, `skills_score`, or any other existing score.

### Bullet point rewrite

Finds the weakest bullet points across both your Experience and Projects
sections (ones that don't start with a strong action verb or don't include
a quantified result) and suggests a rewrite for each, using a local
rule-based rewriter — one unified feature covering both sections, with each
suggestion tagged by where it came from. Bullets sourced from Projects get
an extra check for a named tech stack, since that's a signal recruiters
specifically look for there.

## Supported file types

PDF, DOCX, and TXT — up to 5MB.

## Troubleshooting

- **"No resume file uploaded" / nothing happens when you click Run scan**:
  make sure you're visiting `http://localhost:5000` (served by Flask), not
  opening `index.html` as a local `file://` path.
- **`pip install` fails on an old Python version**: this project targets
  Python 3.9+. Check your version with `python3 --version`.
- **Port already in use**: run with a different port,
  `PORT=5050 python3 app.py`, then visit `http://localhost:5050`.

## Notes

- All parsing and scoring happens on your own server; nothing is sent
  anywhere else.
- The skills database and keyword lists are intentionally editable —
  open `analyzer.py` and extend `SKILL_DB`, `ACTION_VERBS`, or
  `WEAK_PHRASES` to tune it to your industry.
