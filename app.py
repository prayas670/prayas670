import os
import tempfile

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from analyzer import extract_text, score_resume, analyze_ats_risk, get_bullet_rewrites
import ml_models

FRONTEND_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
CORS(app)

ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[-1].lower() in ALLOWED_EXTENSIONS
@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")
@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        # Surfaces which of the three ML integrations are actually active in
        # this deployment (each degrades gracefully to rule-based logic if
        # its package/model file is missing) — handy for debugging setup.
        "ml_models": {
            "sbert_jd_matching": ml_models.get_sbert_model() is not None,
            "spacy_entity_extraction": ml_models.get_spacy_nlp() is not None,
            "xgboost_ats_score": ml_models.get_ats_model() is not None,
        },
    })
@app.route("/api/analyze", methods=["POST"])
def analyze():
    if "resume" not in request.files:
        return jsonify({"error": "No resume file uploaded. Use form field 'resume'."}), 400
    file = request.files["resume"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type. Please upload a PDF, DOCX, or TXT file."}), 400
    file.seek(0, os.SEEK_END)
    size = file.tell()
    file.seek(0)
    if size > MAX_FILE_SIZE:
        return jsonify({"error": "File too large. Max size is 5 MB."}), 400
    jd_text = request.form.get("job_description", "").strip()
    target_role = request.form.get("target_role", "").strip()
    suffix = "." + file.filename.rsplit(".", 1)[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name
    try:
        text = extract_text(tmp_path, file.filename)
        if not text or not text.strip():
            return jsonify({"error": "Could not extract any text from this file. Try a different format."}), 422
        ats_risk = analyze_ats_risk(tmp_path, file.filename, text)
        analysis = score_resume(
            text,
            jd_text if jd_text else None,
            ats_risk=ats_risk,
            target_role=target_role if target_role else None,
        )
        analysis["ats_risk"] = ats_risk

        # Bullet point rewrite suggestions (Experience + Projects), using
        # the local rule-based rewriter.
        analysis["bullet_rewrites"] = get_bullet_rewrites(text)
        return jsonify(analysis)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Analysis failed: {e}"}), 500
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="127.0.0.1", port=port, debug=debug)