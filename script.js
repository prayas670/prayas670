const API_BASE = ""; // same-origin; change to e.g. "http://localhost:5000" if serving frontend separately

const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const dzTitle = document.getElementById("dzTitle");
const dzSub = document.getElementById("dzSub");
const scanBeam = document.getElementById("scanBeam");
const fileChip = document.getElementById("fileChip");
const fileNameEl = document.getElementById("fileName");
const clearFileBtn = document.getElementById("clearFile");
const analyzeBtn = document.getElementById("analyzeBtn");
const jdInput = document.getElementById("jdInput");
const targetRoleInput = document.getElementById("targetRoleInput");
const errorMsg = document.getElementById("errorMsg");
const report = document.getElementById("report");
const downloadPdfBtn = document.getElementById("downloadPdfBtn");

let selectedFile = null;

// --- Dropzone ---

dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
});

["dragenter", "dragover"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  })
);
["dragleave", "drop"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
  })
);
dropzone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files[0];
  if (file) setFile(file);
});

fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) setFile(fileInput.files[0]);
});

clearFileBtn.addEventListener("click", (e) => {
  e.stopPropagation();
  selectedFile = null;
  fileInput.value = "";
  fileChip.hidden = true;
  dropzone.classList.remove("has-file");
  dzTitle.textContent = "Drop resume here";
  dzSub.textContent = "PDF, DOCX or TXT — max 5MB";
  analyzeBtn.disabled = true;
});

function setFile(file) {
  const ext = file.name.split(".").pop().toLowerCase();
  if (!["pdf", "docx", "txt"].includes(ext)) {
    showError("Unsupported file type. Please upload a PDF, DOCX, or TXT file.");
    return;
  }
  if (file.size > 5 * 1024 * 1024) {
    showError("File too large — max 5MB.");
    return;
  }
  hideError();
  selectedFile = file;
  fileNameEl.textContent = file.name;
  fileChip.hidden = false;
  dropzone.classList.add("has-file");
  dzTitle.textContent = "Ready to scan";
  dzSub.textContent = "Click 'Run scan' to analyze";
  analyzeBtn.disabled = false;
}

function showError(msg) {
  errorMsg.textContent = msg;
  errorMsg.hidden = false;
}
function hideError() {
  errorMsg.hidden = true;
}

// --- Analyze ---

analyzeBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  hideError();
  report.hidden = true;
  setScanning(true);
  const formData = new FormData();
  formData.append("resume", selectedFile);
  formData.append("job_description", jdInput.value.trim());
  formData.append("target_role", targetRoleInput.value.trim());
  try {
    const res = await fetch(`${API_BASE}/api/analyze`, {
      method: "POST",
      body: formData,
    });
    const data = await res.json();
    if (!res.ok) {
      showError(data.error || "Something went wrong analyzing your resume.");
      return;
    }
    renderReport(data);
  } catch (err) {
    showError("Could not reach the analysis server. Is the backend running?");
  } finally {
    setScanning(false);
  }
});

function setScanning(on) {
  analyzeBtn.disabled = on || !selectedFile;
  analyzeBtn.querySelector(".btn-label").textContent = on ? "Scanning..." : "Run scan";
  scanBeam.classList.toggle("scanning", on);
}

// --- Rendering ---

function renderReport(data) {
  report.hidden = false;
  animateGauge(data.overall_score);
  renderGrade(data.overall_score);
  const subRow = document.getElementById("subscoreRow");
  subRow.innerHTML = "";
  const subs = [
    ["Structure", data.structure_score],
    ["Content quality", data.content_score],
    ["Skills coverage", data.skills_score],
  ];
  if (data.jd_match) subs.push(["JD similarity", data.jd_match.similarity]);
  subs.forEach(([label, val], i) => {
    const div = document.createElement("div");
    div.className = "subscore";
    div.innerHTML = `<span class="subscore-label">${label}</span><span class="subscore-value">${val == null ? "—" : "0/100"}</span>`;
    subRow.appendChild(div);
    if (val != null) {
      const valueEl = div.querySelector(".subscore-value");
      setTimeout(() => countUpTo(valueEl, val, "/100"), i * 60);
    }
  });
  renderSectionScores(data.section_scores);
  renderHeatmap(data.section_scores);
  renderCompleteness(data.completeness);
  renderAtsRisk(data.ats_risk);
  renderPersonalInfoRisk(data.personal_info_risk);
  renderResumeHighlights(data.text_highlights);
  renderDuplicateContent(data.duplicate_content);
  renderEntityProfile(data.entities);
  renderBulletRewrites(data.bullet_rewrites);
  renderProjectQuality(data);
  renderDashboard(data);
  const skillChips = document.getElementById("skillChips");
  skillChips.innerHTML = "";
  if (data.skills_found.length === 0) {
    skillChips.innerHTML = `<span class="dz-sub">No known skills detected — consider adding a Skills section.</span>`;
  } else {
    data.skills_found.forEach((s) => {
      const span = document.createElement("span");
      span.className = "chip";
      span.textContent = s;
      skillChips.appendChild(span);
    });
  }
  renderJDMatch(data);
  renderTargetRoleMatch(data);
  const suggList = document.getElementById("suggestionsList");
  suggList.innerHTML = "";
  data.suggestions.forEach((s) => {
    const li = document.createElement("li");
    li.textContent = s;
    suggList.appendChild(li);
  });
  initScrollReveal();
  report.scrollIntoView({ behavior: "smooth", block: "start" });
}

// --- Scroll-reveal animation ---
let scrollRevealObserver = null;

function initScrollReveal() {
  const targets = Array.from(document.querySelectorAll("#report .card, #report .subscore"))
    .filter((el) => !el.hidden && el.offsetParent !== null);
  targets.forEach((el, i) => {
    el.classList.remove("in-view");
    el.classList.add("reveal");
    el.style.transitionDelay = `${Math.min(i * 40, 400)}ms`;
  });
  if (scrollRevealObserver) scrollRevealObserver.disconnect();
  scrollRevealObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          const el = entry.target;
          el.classList.add("in-view");
          scrollRevealObserver.unobserve(el);
          // Clear the stagger delay after it plays so later hover transitions aren't delayed too.
          setTimeout(() => { el.style.transitionDelay = ""; }, 1000);
        }
      });
    },
    { threshold: 0.1, rootMargin: "0px 0px -40px 0px" }
  );
  targets.forEach((el) => scrollRevealObserver.observe(el));
}

function renderKeywordChips(containerId, keywords, chipClass, emptyText) {
  const el = document.getElementById(containerId);
  el.innerHTML = "";
  if (!keywords || !keywords.length) {
    const span = document.createElement("span");
    span.className = "dz-sub";
    span.textContent = emptyText;
    el.appendChild(span);
    return;
  }
  keywords.forEach((k) => {
    const span = document.createElement("span");
    span.className = `chip ${chipClass}`;
    span.textContent = k;
    el.appendChild(span);
  });
}

function renderDensityChips(containerId, keywords, densityMap, isMatch) {
  const el = document.getElementById(containerId);
  el.innerHTML = "";
  const list = keywords || [];
  if (list.length === 0) {
    const span = document.createElement("span");
    span.className = "dz-sub";
    span.textContent = isMatch ? "None matched yet" : "None found";
    el.appendChild(span);
    return;
  }
  const entries = list.map((k) => [k, (densityMap && densityMap[k]) || 0]);
  entries.sort((a, b) => b[1] - a[1]); // Sort by density descending
  entries.forEach(([k, count]) => {
    const span = document.createElement("span");
    span.className = `chip ${isMatch ? "match" : "miss"} density`;
    span.innerHTML = isMatch && count
      ? `${escapeHtml(k)} <span class="density-badge">${count}</span>`
      : escapeHtml(k);
    el.appendChild(span);
  });
}

function scoreColor(score) {
  if (score < 50) return "#E2584F"; // coral
  if (score < 75) return "#D9A238"; // amber
  return "#4FBE82"; // green
}

// --- Section-wise score ---

// --- Overall letter grade ---

function scoreToGrade(score) {
  if (score >= 97) return "A+";
  if (score >= 93) return "A";
  if (score >= 90) return "A-";
  if (score >= 87) return "B+";
  if (score >= 83) return "B";
  if (score >= 80) return "B-";
  if (score >= 77) return "C+";
  if (score >= 73) return "C";
  if (score >= 70) return "C-";
  if (score >= 65) return "D+";
  if (score >= 60) return "D";
  return "F";
}

function renderGrade(score) {
  const el = document.getElementById("gaugeGrade");
  if (!el) return;
  const grade = scoreToGrade(score);
  el.textContent = grade;
  el.className = "gauge-grade grade-" + grade[0].toLowerCase();
  stampVerdict(grade);
}

// Picks the stamp verdict/color for the grade and replays its animation.
function stampVerdict(grade) {
  const stamp = document.getElementById("gradeStamp");
  const text = document.getElementById("stampText");
  if (!stamp || !text) return;
  const letter = grade[0];
  let verdict = "REVIEW";
  let color = "#D9A238"; // amber
  if (letter === "A") { verdict = "CLEARED"; color = "#4FBE82"; }
  else if (letter === "B") { verdict = "STRONG"; color = "#3E9C8C"; }
  else if (letter === "C") { verdict = "REVIEW"; color = "#D9A238"; }
  else { verdict = "FLAGGED"; color = "#E2584F"; }
  text.textContent = verdict;
  stamp.style.setProperty("--stamp-color", color);
  stamp.classList.remove("show");
  // eslint-disable-next-line no-unused-expressions
  void stamp.offsetWidth; // restart the CSS animation on repeat scans
  requestAnimationFrame(() => stamp.classList.add("show"));
}

function renderSectionScores(sectionScores) {
  const wrap = document.getElementById("sectionBars");
  wrap.innerHTML = "";
  if (!sectionScores) return;
  Object.values(sectionScores).forEach((s) => {
    const row = document.createElement("div");
    row.className = "section-bar-row" + (s.present ? "" : " not-present");
    const label = document.createElement("div");
    label.className = "section-bar-label";
    let subtext = "";
    if (!s.present) subtext = "Not found";
    else if (typeof s.count === "number") subtext = `${s.count} skills matched`;
    else if (typeof s.bullets_detected === "number") subtext = `${s.bullets_detected} bullets detected`;
    label.innerHTML = `${s.label}${subtext ? `<span class="sub">${subtext}</span>` : ""}`;
    const track = document.createElement("div");
    track.className = "section-bar-track";
    const fill = document.createElement("div");
    fill.className = "section-bar-fill";
    const val = s.score == null ? 0 : s.score;
    fill.style.width = "0%";
    fill.style.background = s.present ? scoreColor(val) : "var(--line)";
    track.appendChild(fill);
    const valueEl = document.createElement("div");
    valueEl.className = "section-bar-value";
    valueEl.textContent = s.score == null ? "—" : `${s.score}/100`;
    row.appendChild(label);
    row.appendChild(track);
    row.appendChild(valueEl);
    wrap.appendChild(row);
    requestAnimationFrame(() => { fill.style.width = `${val}%`; });
  });
}
// --- Resume heatmap ---

function renderHeatmap(sectionScores) {
  const container = document.getElementById("heatmapContainer");
  container.innerHTML = "";
  if (!sectionScores) return;
  Object.values(sectionScores).forEach(s => {
    if (!s.present) return;
    const block = document.createElement("div");
    block.className = "heatmap-block";
    block.style.backgroundColor = scoreColor(s.score || 0);
    let weight = 1;
    if (s.label.toLowerCase().includes("experience")) weight = 3;
    if (s.label.toLowerCase().includes("education")) weight = 1.5;
    if (s.label.toLowerCase().includes("skills")) weight = 2;
    if (s.label.toLowerCase().includes("projects")) weight = 2;
    block.style.flexGrow = weight;

    block.style.color = "#fff";
    const displayScore = s.score == null ? "—" : s.score;
    block.innerHTML = `<span class="heatmap-label">${escapeHtml(s.label)}</span><span class="heatmap-score">${displayScore}/100</span>`;
    container.appendChild(block);
  });
}
// --- Completeness score ---

function renderCompleteness(completeness) {
  if (!completeness) return;
  document.getElementById("completenessPct").textContent = `${completeness.score}/100`;
  const fill = document.getElementById("completenessBarFill");
  fill.style.width = "0%";
  requestAnimationFrame(() => { fill.style.width = `${completeness.score}%`; });
  const list = document.getElementById("completenessChecklist");
  list.innerHTML = "";
  completeness.checklist.forEach(([label, passed]) => {
    const li = document.createElement("li");
    li.className = passed ? "pass" : "fail";
    li.innerHTML = `<span class="mark">${passed ? "✓" : "✕"}</span><span>${label}</span>`;
    list.appendChild(li);
  });
}

// --- ATS risk analysis ---

function renderAtsRisk(atsRisk) {
  if (!atsRisk) return;
  const badge = document.getElementById("riskBadge");
  badge.textContent = `${atsRisk.risk_level} risk`;
  badge.className = `risk-badge ${atsRisk.risk_level.toLowerCase()}`;
  const mlLine = document.getElementById("atsMlScoreLine");
  const mlValue = document.getElementById("atsMlScoreValue");
  if (atsRisk.ml_ats_score !== null && atsRisk.ml_ats_score !== undefined) {
    mlLine.hidden = false;
    mlValue.textContent = atsRisk.ml_ats_score;
  } else if (mlLine) {
    mlLine.hidden = true;
  }
  const list = document.getElementById("atsIssues");
  list.innerHTML = "";
  if (!atsRisk.issues.length) {
    const p = document.createElement("p");
    p.className = "ats-clean";
    p.textContent = "No common ATS red flags detected — tables, images, multi-column layouts, and non-standard headers all look clear.";
    list.appendChild(p);
    return;
  }
  atsRisk.issues.forEach((issue) => {
    const li = document.createElement("li");
    li.className = issue.severity;
    li.innerHTML = `<span class="ats-issue-title">${issue.issue}</span><p class="ats-issue-tip">${issue.tip}</p>`;
    list.appendChild(li);
  });
}

// --- Personal info / bias-risk check ---

function renderPersonalInfoRisk(personalInfoRisk) {
  const card = document.getElementById("personalInfoCard");
  if (!card) return;
  if (!personalInfoRisk) {
    card.hidden = true;
    return;
  }
  card.hidden = false;
  const badge = document.getElementById("personalInfoBadge");
  badge.textContent = personalInfoRisk.risk_level === "Clean" ? "Clean" : `${personalInfoRisk.risk_level} risk`;
  badge.className = `risk-badge ${personalInfoRisk.risk_level.toLowerCase()}`;
  const list = document.getElementById("personalInfoIssues");
  list.innerHTML = "";
  if (!personalInfoRisk.flags || personalInfoRisk.flags.length === 0) {
    const p = document.createElement("p");
    p.className = "ats-clean";
    p.textContent = "No personal/demographic details detected — no photo, date of birth, marital status, nationality, gender, or religion found on the page.";
    list.appendChild(p);
    return;
  }
  personalInfoRisk.flags.forEach((flag) => {
    const li = document.createElement("li");
    li.className = flag.severity;
    li.innerHTML = `<span class="ats-issue-title">${escapeHtml(flag.item)}</span><p class="ats-issue-tip">${escapeHtml(flag.tip)}</p>`;
    list.appendChild(li);
  });
}

// --- Highlighted resume ---

function renderResumeHighlights(textHighlights) {
  const card = document.getElementById("resumeHighlightCard");
  if (!card) return;
  if (!textHighlights || !textHighlights.segments || textHighlights.segments.length === 0) {
    card.hidden = true;
    return;
  }
  card.hidden = false;
  const textEl = document.getElementById("resumeHighlightText");
  const html = textHighlights.segments
    .map((seg) => {
      const escaped = escapeHtml(seg.text);
      if (seg.type === "weak") return `<mark class="hl-weak">${escaped}</mark>`;
      if (seg.type === "keyword") return `<mark class="hl-keyword">${escaped}</mark>`;
      return escaped;
    })
    .join("");
  textEl.innerHTML = html || `<span class="resume-highlight-empty">No text extracted.</span>`;

  const missingWrap = document.getElementById("resumeMissingWrap");
  const missingChips = document.getElementById("resumeMissingChips");
  const legendMissing = document.getElementById("legendMissingWrap");
  const missing = textHighlights.missing_keywords || [];
  missingChips.innerHTML = "";
  if (missing.length === 0) {
    missingWrap.hidden = true;
    if (legendMissing) legendMissing.hidden = true;
  } else {
    missingWrap.hidden = false;
    if (legendMissing) legendMissing.hidden = false;
    missing.forEach((kw) => {
      const span = document.createElement("span");
      span.className = "chip miss";
      span.textContent = kw;
      missingChips.appendChild(span);
    });
  }
}

// --- Duplicate content detection ---

function renderDuplicateContent(duplicateContent) {
  const card = document.getElementById("duplicateContentCard");
  if (!card) return;
  if (!duplicateContent) {
    card.hidden = true;
    return;
  }
  card.hidden = false;

  const total = duplicateContent.total_issues || 0;
  const badge = document.getElementById("duplicateBadge");
  let label = "Clean", cls = "low";
  if (total > 3) { label = "Needs cleanup"; cls = "high"; }
  else if (total > 0) { label = "Minor"; cls = "medium"; }
  badge.textContent = label;
  badge.className = `risk-badge ${cls}`;

  const cleanLi = (msg) => `<li class="clean"><span class="mark">✓</span><span>${msg}</span></li>`;

  // Repeated Skills
  const skillsList = document.getElementById("duplicateSkillsList");
  skillsList.innerHTML = "";
  const dupSkills = duplicateContent.duplicate_skills || [];
  const overSkills = duplicateContent.overused_skills || [];
  if (dupSkills.length === 0 && overSkills.length === 0) {
    skillsList.innerHTML = cleanLi("No repeated skills detected.");
  } else {
    dupSkills.forEach((d) => {
      const li = document.createElement("li");
      li.className = "fail";
      li.innerHTML = `<span class="mark count">${d.count}×</span><span>"${escapeHtml(d.skill)}" is listed ${d.count} times in your Skills section — remove the duplicates.</span>`;
      skillsList.appendChild(li);
    });
    overSkills.forEach((d) => {
      const li = document.createElement("li");
      li.className = "fail";
      li.innerHTML = `<span class="mark count">${d.count}×</span><span>"${escapeHtml(d.skill)}" is mentioned ${d.count} times across your resume — possible keyword stuffing.</span>`;
      skillsList.appendChild(li);
    });
  }

  // Repeated Sentences
  const sentList = document.getElementById("duplicateSentencesList");
  sentList.innerHTML = "";
  const dupSentences = duplicateContent.duplicate_sentences || [];
  if (dupSentences.length === 0) {
    sentList.innerHTML = cleanLi("No repeated sentences detected.");
  } else {
    dupSentences.forEach((d) => {
      const li = document.createElement("li");
      li.className = "fail";
      li.innerHTML = `<span class="mark count">${d.count}×</span><span>"${escapeHtml(d.sentence)}"</span>`;
      sentList.appendChild(li);
    });
  }

  // Repeated Bullet Points
  const bulletList = document.getElementById("duplicateBulletsList");
  bulletList.innerHTML = "";
  const dupBullets = duplicateContent.duplicate_bullets || [];
  const nearBullets = duplicateContent.near_duplicate_bullets || [];
  if (dupBullets.length === 0 && nearBullets.length === 0) {
    bulletList.innerHTML = cleanLi("No repeated or near-identical bullet points detected.");
  } else {
    dupBullets.forEach((d) => {
      const li = document.createElement("li");
      li.className = "fail";
      li.innerHTML = `<span class="mark count">${d.count}×</span><span>"${escapeHtml(d.bullet)}"</span>`;
      bulletList.appendChild(li);
    });
    nearBullets.forEach((d) => {
      const li = document.createElement("li");
      li.className = "fail";
      li.innerHTML = `<span class="mark count">${d.similarity}%</span><span>"${escapeHtml(d.bullet_a)}" is very similar to "${escapeHtml(d.bullet_b)}" — make sure each role shows a distinct contribution.</span>`;
      bulletList.appendChild(li);
    });
  }
}

// --- Extracted profile (spaCy) ---

function renderEntityProfile(entities) {
  const card = document.getElementById("profileCard");
  if (!entities) {
    card.hidden = true;
    return;
  }
  const hasAny = (entities.education && entities.education.length)
    || (entities.experience && entities.experience.length)
    || (entities.certifications && entities.certifications.length);
  if (!hasAny) {
    card.hidden = true;
    return;
  }
  card.hidden = false;
  const eduList = document.getElementById("entityEducationList");
  eduList.innerHTML = "";
  if (!entities.education.length) {
    const li = document.createElement("li");
    li.textContent = "No degree detected.";
    eduList.appendChild(li);
  } else {
    entities.education.forEach((edu) => {
      const li = document.createElement("li");
      const parts = [edu.degree];
      if (edu.institution) parts.push(edu.institution);
      if (edu.year) parts.push(edu.year);
      li.innerHTML = `<span class="mark">🎓</span><span>${parts.join(" — ")}</span>`;
      eduList.appendChild(li);
    });
  }
  const expList = document.getElementById("entityExperienceList");
  expList.innerHTML = "";
  if (!entities.experience.length) {
    const li = document.createElement("li");
    li.textContent = "No structured role/company entries detected.";
    expList.appendChild(li);
  } else {
    entities.experience.forEach((exp) => {
      const li = document.createElement("li");
      const parts = [exp.title];
      if (exp.organization) parts.push(exp.organization);
      if (exp.dates) parts.push(exp.dates);
      li.innerHTML = `<span class="mark">💼</span><span>${parts.join(" — ")}</span>`;
      expList.appendChild(li);
    });
  }
  const certChips = document.getElementById("entityCertChips");
  certChips.innerHTML = "";
  if (!entities.certifications.length) {
    certChips.innerHTML = `<span class="dz-sub">No certifications detected.</span>`;
  } else {
    entities.certifications.forEach((cert) => {
      const span = document.createElement("span");
      span.className = "chip match";
      span.textContent = cert;
      certChips.appendChild(span);
    });
  }
}

// --- Bullet point rewrites ---

function renderBulletRewrites(bulletRewrites) {
  const card = document.getElementById("bulletCard");
  const list = document.getElementById("bulletList");
  const subtitle = document.getElementById("bulletSubtitle");
  list.innerHTML = "";
  if (!bulletRewrites || bulletRewrites.length === 0) {
    card.hidden = true;
    return;
  }
  card.hidden = false;
  subtitle.textContent = "Rewrites for the weakest bullet points across your Experience and Projects sections, using strong action verbs and quantified impact.";
  bulletRewrites.forEach((b) => {
    const item = document.createElement("div");
    item.className = "bullet-item";
    const sectionTag = b.section ? `<div class="bullet-item-head"><span class="bullet-section-tag">${escapeHtml(b.section)}</span></div>` : "";
    item.innerHTML = `
      ${sectionTag}
      <div class="bullet-original"><span class="bullet-tag before">Before</span><span>${escapeHtml(b.original)}</span></div>
      <div class="bullet-suggested"><span class="bullet-tag after">After</span><span>${escapeHtml(b.suggested)}</span></div>
      ${b.notes && b.notes.length ? `<ul class="bullet-notes">${b.notes.map((n) => `<li>${escapeHtml(n)}</li>`).join("")}</ul>` : ""}
    `;
    list.appendChild(item);
  });
}

// Animates a subscore value counting up from zero, e.g. "0/100".
function countUpTo(el, targetValue, suffix, duration = 700, decimals = 0) {
  const start = performance.now();
  function tick(now) {
    const progress = Math.min((now - start) / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    el.textContent = `${(eased * targetValue).toFixed(decimals)}${suffix}`;
    if (progress < 1) requestAnimationFrame(tick);
    else el.textContent = `${targetValue.toFixed(decimals)}${suffix}`;
  }
  requestAnimationFrame(tick);
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

function animateGauge(score) {
  const circumference = 377; // 2 * PI * 60, matches stroke-dasharray in CSS
  const fill = document.getElementById("gaugeFill");
  const offset = circumference - (score / 100) * circumference;
  let color = "#4FBE82"; // green
  if (score < 50) color = "#E2584F"; // coral
  else if (score < 75) color = "#D9A238"; // amber
  fill.style.stroke = color;

  fill.style.transition = "none";
  fill.style.strokeDashoffset = circumference;
  requestAnimationFrame(() => {
    fill.style.transition = "stroke-dashoffset 1s cubic-bezier(0.65,0,0.35,1), stroke 0.4s ease";
    fill.style.strokeDashoffset = offset;
  });
  let current = 0;
  const target = score;
  const duration = 800;
  const start = performance.now();
  const numberEl = document.getElementById("scoreNumber");
  function tick(now) {
    const progress = Math.min((now - start) / duration, 1);
    current = progress * target;
    numberEl.textContent = Math.round(current);
    if (progress < 1) requestAnimationFrame(tick);
    else numberEl.textContent = Math.round(target);
  }
  requestAnimationFrame(tick);
}

// --- Dashboard & PDF export ---

if (downloadPdfBtn) {
  downloadPdfBtn.addEventListener("click", async () => {
    const element = document.getElementById("report");
    downloadPdfBtn.disabled = true;
    const originalLabel = downloadPdfBtn.textContent;
    downloadPdfBtn.textContent = "Preparing PDF...";
    downloadPdfBtn.style.display = "none";

    // Scroll-reveal elements only reach opacity:1 once actually scrolled
    // into view — anything below the fold is still faded at click time,
    // so force everything visible before the snapshot is taken.
    const revealEls = element.querySelectorAll(".reveal");
    revealEls.forEach((el) => {
      el.classList.add("in-view");
      el.style.transitionDelay = "0s";
    });
    document.body.classList.add("pdf-exporting");
    element.classList.add("pdf-exporting");

    // Let the layout/style changes above actually paint, and give any
    // still-settling chart animations a moment to finish, before capturing.
    await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
    await new Promise((resolve) => setTimeout(resolve, 200));

    try {
      await html2pdf()
        .from(element)
        .set({
          margin: 0.4,
          filename: "Resume-Analysis-Report.pdf",
          image: { type: "jpeg", quality: 0.98 },
          html2canvas: {
            scale: 2,
            useCORS: true,
            backgroundColor: "#0B0D0B", // matches --bg-dark; without this html2canvas defaults to white and washes out every card
            windowWidth: element.scrollWidth,
          },
          jsPDF: { unit: "in", format: "letter", orientation: "portrait" },
          // "legacy" and "css" are two separate page-break algorithms; running
          // both at once is a known cause of extra phantom blank space in
          // html2pdf output, so "css" alone is used here. Only small, atomic
          // pieces are listed as avoid — not whole .card containers — so a
          // long card (e.g. Bullet point rewrites) can flow across a page
          // break instead of being pushed whole onto the next page and
          // leaving the rest of the previous page empty.
          pagebreak: {
            mode: ["css"],
            avoid: [
              ".card.chart-card",
              ".subscore",
              ".heatmap-block",
              ".bullet-item",
              ".checklist li",
              ".suggestions li",
            ],
          },
        })
        .save();
    } catch (err) {
      showError("Could not generate the PDF report. Please try again.");
    } finally {
      document.body.classList.remove("pdf-exporting");
      element.classList.remove("pdf-exporting");
      revealEls.forEach((el) => { el.style.transitionDelay = ""; });
      downloadPdfBtn.disabled = false;
      downloadPdfBtn.textContent = originalLabel;
      downloadPdfBtn.style.display = "inline-block";
    }
  });
}

let radarChartInstance = null;
let barChartInstance = null;

// Shared tooltip styling for all charts.
const CHART_TOOLTIP_BASE = {
  backgroundColor: 'rgba(18, 19, 15, 0.92)',
  titleColor: '#F2ECDD',
  bodyColor: '#C9BFA9',
  borderColor: 'rgba(204, 149, 68, 0.35)',
  borderWidth: 1,
  padding: 10,
  cornerRadius: 10,
  displayColors: true,
  boxPadding: 4,
  titleFont: { family: "'Outfit', sans-serif", weight: '600', size: 13 },
  bodyFont: { family: "'Inter', sans-serif", size: 12.5 },
};

function renderDashboard(data) {
  document.getElementById("dashboardGrid").style.display = "grid";
  const ctxRadar = document.getElementById("radarChart").getContext("2d");
  if (radarChartInstance) radarChartInstance.destroy();
  const labelsRadar = ["Structure", "Content", "Skills"];
  const dataRadar = [data.structure_score, data.content_score, data.skills_score];
  if (data.jd_match) {
    labelsRadar.push("JD Match");
    dataRadar.push(data.jd_match.similarity);
  }
  const radarFill = ctxRadar.createLinearGradient(0, 0, 0, 260);
  radarFill.addColorStop(0, 'rgba(204, 149, 68, 0.38)');
  radarFill.addColorStop(1, 'rgba(62, 156, 140, 0.08)');
  radarChartInstance = new Chart(ctxRadar, {
    type: 'radar',
    data: {
      labels: labelsRadar,
      datasets: [{
        label: 'Score',
        data: dataRadar,
        backgroundColor: radarFill,
        borderColor: '#CC9544',
        borderWidth: 2.5,
        pointBackgroundColor: '#0B0D0B',
        pointBorderColor: '#3E9C8C',
        pointBorderWidth: 2,
        pointRadius: 5,
        pointHoverRadius: 7,
        pointHoverBackgroundColor: '#3E9C8C',
        pointHoverBorderColor: '#F2ECDD',
        tension: 0.15,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 900, easing: 'easeOutQuart' },
      scales: {
        r: {
          min: 0, max: 100,
          angleLines: { color: 'rgba(242,236,221,0.07)' },
          grid: { color: 'rgba(242,236,221,0.07)', circular: true },
          pointLabels: { color: '#C9BFA9', font: { family: "'Inter', sans-serif", size: 12, weight: '500' } },
          ticks: { color: '#7A705C', backdropColor: 'transparent', stepSize: 25 }
        }
      },
      plugins: {
        legend: { display: false },
        tooltip: { ...CHART_TOOLTIP_BASE, callbacks: { label: (ctx) => ` ${ctx.label}: ${ctx.raw}/100` } }
      }
    }
  });
  const ctxBar = document.getElementById("barChart").getContext("2d");
  if (barChartInstance) barChartInstance.destroy();
  const labelsBar = [];
  const dataBar = [];
  Object.values(data.section_scores).forEach(s => {
    labelsBar.push(s.label);
    dataBar.push(s.score || 0);
  });
  const barFill = ctxBar.createLinearGradient(0, 0, 0, 260);
  barFill.addColorStop(0, '#3E9C8C');
  barFill.addColorStop(1, '#CC9544');
  const barHoverFill = ctxBar.createLinearGradient(0, 0, 0, 260);
  barHoverFill.addColorStop(0, '#E3B563');
  barHoverFill.addColorStop(1, '#B98A3E');
  barChartInstance = new Chart(ctxBar, {
    type: 'bar',
    data: {
      labels: labelsBar,
      datasets: [{
        label: 'Section Score',
        data: dataBar,
        backgroundColor: barFill,
        hoverBackgroundColor: barHoverFill,
        borderRadius: 8,
        borderSkipped: false,
        maxBarThickness: 42,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 900, easing: 'easeOutQuart' },
      scales: {
        y: {
          beginAtZero: true, max: 100,
          grid: { color: 'rgba(242,236,221,0.05)' },
          border: { display: false },
          ticks: { color: '#7A705C', font: { size: 11 } }
        },
        x: {
          grid: { display: false },
          border: { display: false },
          ticks: { color: '#C9BFA9', font: { family: "'Inter', sans-serif", size: 11.5, weight: '500' } }
        }
      },
      plugins: {
        legend: { display: false },
        tooltip: { ...CHART_TOOLTIP_BASE, callbacks: { label: (ctx) => ` ${Number(ctx.formattedValue).toFixed(0)}/100` } }
      }
    }
  });
}

let skillGapChartInstance = null;
function renderSkillGap(matchedList, missingList) {
  const ctx = document.getElementById("skillGapChart").getContext("2d");
  if (skillGapChartInstance) skillGapChartInstance.destroy();
  const matchedCount = matchedList ? matchedList.length : 0;
  const missingCount = missingList ? missingList.length : 0;
  const matchedFill = ctx.createLinearGradient(0, 0, 0, 200);
  matchedFill.addColorStop(0, '#6ED4A3');
  matchedFill.addColorStop(1, '#3E9C8C');
  const missingFill = ctx.createLinearGradient(0, 0, 0, 200);
  missingFill.addColorStop(0, '#EB8079');
  missingFill.addColorStop(1, '#E2584F');

  // Shows the match % in the doughnut hole.
  const centerTextPlugin = {
    id: 'centerText',
    afterDraw(chart) {
      const total = matchedCount + missingCount;
      if (!total) return;
      const pct = Math.round((matchedCount / total) * 100);
      const { ctx, chartArea: { left, right, top, bottom } } = chart;
      const cx = (left + right) / 2;
      const cy = (top + bottom) / 2;
      ctx.save();
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.font = "700 26px 'Outfit', sans-serif";
      ctx.fillStyle = '#F2ECDD';
      ctx.fillText(`${pct}%`, cx, cy - 8);
      ctx.font = "600 10.5px 'JetBrains Mono', monospace";
      ctx.fillStyle = '#A69C87';
      ctx.fillText('MATCHED', cx, cy + 14);
      ctx.restore();
    }
  };
  skillGapChartInstance = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['Matched', 'Missing'],
      datasets: [{
        data: [matchedCount, missingCount],
        backgroundColor: [matchedFill, missingFill],
        borderColor: '#12140F',
        borderWidth: 3,
        hoverOffset: 10,
        spacing: 2,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: '72%',
      animation: { duration: 900, easing: 'easeOutQuart', animateRotate: true, animateScale: true },
      plugins: {
        legend: {
          position: 'bottom',
          labels: { color: '#C9BFA9', font: { family: "'Inter', sans-serif", size: 12 }, usePointStyle: true, pointStyle: 'circle', padding: 16 }
        },
        tooltip: CHART_TOOLTIP_BASE
      }
    },
    plugins: [centerTextPlugin]
  });
}

function renderProjectQuality(data) {
  const c = document.getElementById("projectQualityCard");
  if (!data.project_quality || !data.project_quality.metrics || Object.keys(data.project_quality.metrics).length === 0) {
    c.hidden = true;
    return;
  }
  c.hidden = false;
  const r = document.getElementById("projectSubscoreRow");
  r.innerHTML = "";
  const pq = data.project_quality;
  const subs = [
    { label: "Overall", score: pq.score },
    { label: "Tech Stack", score: pq.metrics.tech_stack_score },
    { label: "Complexity", score: pq.metrics.complexity_score },
    { label: "Impact", score: pq.metrics.impact_score },
    { label: "Action Verbs", score: pq.metrics.action_verbs_score }
  ];
  subs.forEach((s, i) => {
    const div = document.createElement("div");
    div.className = "subscore";
    div.innerHTML = `<span class="subscore-label">${s.label}</span><span class="subscore-value">${s.score == null ? "—" : "0/100"}</span>`;
    r.appendChild(div);
    if (s.score != null) {
      const valueEl = div.querySelector(".subscore-value");
      setTimeout(() => countUpTo(valueEl, s.score, "/100"), i * 60);
    }
  });
  const sl = document.getElementById("projectSuggestionsList");
  sl.innerHTML = "";
  pq.suggestions.forEach(sug => {
    const li = document.createElement("li");
    li.textContent = sug;
    sl.appendChild(li);
  });
}

function renderTargetRoleMatch(data) {
  const c = document.getElementById("targetRoleCard");
  const trm = data.target_role_match;
  if (!trm) {
    c.hidden = true;
    return;
  }
  c.hidden = false;
  const subtitle = document.getElementById("targetRoleSubtitle");
  const matchedWrap = document.getElementById("targetRoleMatchedChips");
  const missingWrap = document.getElementById("targetRoleMissingChips");
  matchedWrap.innerHTML = "";
  missingWrap.innerHTML = "";
  if (!trm.recognized) {
    subtitle.textContent = `"${trm.role_input}" wasn't recognized — try a more common title (e.g. "Data Analyst", "Software Engineer", "Product Manager").`;
    matchedWrap.innerHTML = "";
    missingWrap.innerHTML = "";
    return;
  }
  subtitle.textContent = `Matched against typical "${trm.matched_role}" skills — ${trm.match_score}% coverage.`;
  if (trm.matched_skills.length === 0) {
    matchedWrap.innerHTML = `<span class="dz-sub">None yet.</span>`;
  } else {
    trm.matched_skills.forEach((s) => {
      const span = document.createElement("span");
      span.className = "chip match";
      span.textContent = s;
      matchedWrap.appendChild(span);
    });
  }
  if (trm.missing_skills.length === 0) {
    missingWrap.innerHTML = `<span class="dz-sub">Nothing missing — great coverage!</span>`;
  } else {
    trm.missing_skills.forEach((s) => {
      const span = document.createElement("span");
      span.className = "chip miss";
      span.textContent = s;
      missingWrap.appendChild(span);
    });
  }
}

function renderJDMatch(data) {
  const c = document.getElementById("jdCard");
  if (!data.jd_match) {
    c.hidden = true;
    return;
  }
  c.hidden = false;
  const jd = data.jd_match;
  document.getElementById("jdSimilarity").textContent = jd.similarity + "/100";
  renderDensityChips("reqMatchedChips", jd.required_matched || [], jd.keyword_density, true);
  renderDensityChips("reqMissingChips", jd.required_missing || [], jd.keyword_density, false);
  const prefWrap = document.getElementById("preferredSkillsWrap");
  if ((jd.preferred_matched && jd.preferred_matched.length > 0) || (jd.preferred_missing && jd.preferred_missing.length > 0)) {
      prefWrap.hidden = false;
      renderDensityChips("prefMatchedChips", jd.preferred_matched || [], jd.keyword_density, true);
      renderDensityChips("prefMissingChips", jd.preferred_missing || [], jd.keyword_density, false);
  } else {
      prefWrap.hidden = true;
  }
  const reqsSection = document.getElementById("jdReqsSection");
  if (jd.education_requirements || jd.experience_requirements) {
      reqsSection.hidden = false;
      document.getElementById("jdEdu").textContent = jd.education_requirements || "Not specified";
      document.getElementById("jdExp").textContent = jd.experience_requirements || "Not specified";
  } else {
      reqsSection.hidden = true;
  }
  const allMatched = (jd.required_matched || []).concat(jd.preferred_matched || []);
  const allMissing = (jd.required_missing || []).concat(jd.preferred_missing || []);
  renderSkillGap(allMatched, allMissing);
}