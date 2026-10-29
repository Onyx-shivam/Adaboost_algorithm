import os
import pickle

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template_string, request

app = Flask(__name__)

# --------------------------------------------------------------------------
# 1. LOAD MODEL
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "Adaboost_algorithm.pkl")

# If the exact name isn't found, use any .pkl file sitting next to app.py
if not os.path.exists(MODEL_PATH):
    pkl_files = [f for f in os.listdir(BASE_DIR) if f.lower().endswith(".pkl")]
    if not pkl_files:
        raise FileNotFoundError(
            f"No .pkl model file found in {BASE_DIR}. "
            "Copy your model file into this folder, next to app.py."
        )
    MODEL_PATH = os.path.join(BASE_DIR, pkl_files[0])

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except Exception:
    import joblib  # fallback in case the model was saved with joblib

    model = joblib.load(MODEL_PATH)

# --------------------------------------------------------------------------
# 2. CONFIGURATION  (edit this part to match YOUR dataset)
# --------------------------------------------------------------------------
# Form fields are created automatically from the model's column names.
# Use META to customise any column. The key is the column name in lowercase.
#   label   -> text shown above the field
#   default -> sample value pre-filled in the form
#   step    -> number step (1 for whole numbers, 0.1 for decimals)
#   min/max -> optional limits
#   options -> turns the field into a dropdown: [("Text shown", value), ...]
META = {
    "age":             {"default": 18, "step": 1, "min": 10, "max": 100},
    "gender":          {"default": 0},
    "state":           {"default": 0},
    "family_income":   {"default": 60000, "step": 100},
    "high_school_gpa": {"label": "High School GPA", "default": 3.6, "step": 0.01},
    "sat_score":       {"label": "SAT Score", "default": 1250, "step": 10},
    "act_score":       {"label": "ACT Score", "default": 27, "step": 1},
    "attendance_rate": {"default": 95, "step": 0.1},
    "ap_courses":      {"label": "AP Courses", "default": 3, "step": 1},
    # Example of a dropdown (use YOUR encoding from training):
    # "gender": {"options": [("Male", 0), ("Female", 1)], "default": 0},
}

# Friendly names for the predicted classes.
# Keys are the values in model.classes_ (0, 1, 2 ... or text labels).
CLASS_LABELS = {
    # 0: "Not Eligible",
    # 1: "Eligible",
}

PALETTE = ["#5B5BF0", "#FF6B6B", "#20C9A6", "#FFB020", "#E056FD", "#2D98DA"]
RESULT_COLORS = ["#20C9A6", "#FF6B6B", "#FFB020", "#5B5BF0", "#E056FD", "#2D98DA"]

ICONS = [
    ("age", "🎂"), ("gender", "👤"), ("state", "📍"), ("income", "💰"),
    ("gpa", "🎓"), ("sat", "📝"), ("act", "✏️"), ("attend", "📅"),
    ("ap", "📚"), ("course", "📚"), ("score", "📊"), ("rate", "📈"),
    ("height", "📏"), ("weight", "⚖️"), ("salary", "💰"), ("price", "🏷️"),
]


def icon_for(name):
    n = name.lower()
    for key, ico in ICONS:
        if key in n:
            return ico
    return "✨"


# --------------------------------------------------------------------------
# 3. BUILD FEATURE LIST AUTOMATICALLY
# --------------------------------------------------------------------------
if hasattr(model, "feature_names_in_"):
    RAW_NAMES = [str(n) for n in model.feature_names_in_]
else:
    n_feat = int(getattr(model, "n_features_in_", 1))
    RAW_NAMES = [f"Feature {i + 1}" for i in range(n_feat)]

FEATURES = []
for i, nm in enumerate(RAW_NAMES):
    m = META.get(nm.lower(), {})
    FEATURES.append(
        {
            "name": nm,
            "label": m.get("label", nm.replace("_", " ").title()),
            "default": m.get("default", ""),
            "step": m.get("step", "any"),
            "min": m.get("min", ""),
            "max": m.get("max", ""),
            "options": m.get("options"),
            "icon": icon_for(nm),
            "color": PALETTE[i % len(PALETTE)],
        }
    )

FEATURE_NAMES = [f["name"] for f in FEATURES]
USE_NAMES = hasattr(model, "feature_names_in_")


def label_for(cls):
    return CLASS_LABELS.get(cls, CLASS_LABELS.get(str(cls), str(cls)))


# --------------------------------------------------------------------------
# 4. UI  (HTML + CSS + JS all inside app.py)
# --------------------------------------------------------------------------
PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AdaBoost Predictor</title>
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
  :root{
    --bg:#F4F5FF; --card:#FFFFFF; --ink:#1B1F3B; --muted:#7C82A1;
    --indigo:#5B5BF0; --teal:#20C9A6; --coral:#FF6B6B; --amber:#FFB020;
    --line:#E9EBF6;
  }
  *{box-sizing:border-box;margin:0;padding:0}
  html{scroll-behavior:smooth}
  body{
    font-family:'Poppins',system-ui,sans-serif;background:var(--bg);color:var(--ink);
    min-height:100vh;padding:34px 16px 70px;position:relative;overflow-x:hidden;
  }

  /* floating colour blobs */
  .blob{position:fixed;border-radius:50%;filter:blur(90px);opacity:.55;z-index:0;animation:float 14s ease-in-out infinite}
  .b1{width:380px;height:380px;background:#C9C9FF;top:-100px;left:-100px}
  .b2{width:340px;height:340px;background:#B8F3E6;bottom:-100px;right:-80px;animation-delay:-5s}
  .b3{width:260px;height:260px;background:#FFD6D6;top:45%;right:-90px;animation-delay:-9s}
  @keyframes float{0%,100%{transform:translate(0,0)}50%{transform:translate(30px,-34px)}}

  .wrap{max-width:900px;margin:0 auto;position:relative;z-index:1}

  /* header */
  header{
    position:relative;overflow:hidden;color:#fff;border-radius:30px;padding:42px 40px;
    background:linear-gradient(125deg,#5B5BF0 0%,#7A5CF5 38%,#2D98DA 72%,#20C9A6 100%);
    box-shadow:0 30px 60px -28px rgba(91,91,240,.75);
  }
  header::before,header::after{content:"";position:absolute;border-radius:50%;background:rgba(255,255,255,.12)}
  header::before{width:260px;height:260px;right:-60px;top:-90px}
  header::after{width:160px;height:160px;right:140px;bottom:-90px;background:rgba(255,255,255,.09)}
  .badge{
    display:inline-flex;align-items:center;gap:8px;background:rgba(255,255,255,.2);
    backdrop-filter:blur(6px);padding:6px 14px;border-radius:999px;font-size:12px;font-weight:600;
    letter-spacing:.4px;margin-bottom:16px;
  }
  .pulse{width:8px;height:8px;border-radius:50%;background:#7CFFD9;box-shadow:0 0 0 0 rgba(124,255,217,.8);animation:pulse 1.8s infinite}
  @keyframes pulse{70%{box-shadow:0 0 0 9px rgba(124,255,217,0)}100%{box-shadow:0 0 0 0 rgba(124,255,217,0)}}
  header h1{font-size:38px;font-weight:800;letter-spacing:-.8px;line-height:1.15;position:relative}
  header p{margin-top:10px;opacity:.92;font-size:15px;max-width:520px;position:relative}
  .chips{display:flex;gap:10px;flex-wrap:wrap;margin-top:24px;position:relative}
  .chip-s{background:rgba(255,255,255,.18);border:1px solid rgba(255,255,255,.25);padding:8px 16px;border-radius:14px;font-size:12px;font-weight:500}
  .chip-s b{display:block;font-size:18px;font-weight:700}

  /* cards */
  .card{
    background:rgba(255,255,255,.9);backdrop-filter:blur(10px);border-radius:28px;
    padding:32px;margin-top:24px;border:1px solid #fff;
    box-shadow:0 20px 50px -26px rgba(27,31,59,.28);
  }
  .card-head{display:flex;align-items:center;justify-content:space-between;gap:16px;flex-wrap:wrap;margin-bottom:20px}
  .card h2{font-size:18px;font-weight:700;display:flex;align-items:center;gap:10px}
  .tag{width:12px;height:12px;border-radius:4px;background:var(--coral);transform:rotate(45deg)}
  .progress{display:flex;align-items:center;gap:12px;font-size:12px;color:var(--muted);font-weight:500}
  .p-track{width:140px;height:8px;background:var(--line);border-radius:999px;overflow:hidden}
  .p-fill{height:100%;width:0;background:linear-gradient(90deg,var(--indigo),var(--teal));border-radius:999px;transition:width .4s ease}

  /* fields */
  .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:18px}
  .field{
    background:#FBFBFF;border:1.5px solid var(--line);border-radius:18px;padding:14px 16px;
    display:flex;gap:14px;align-items:center;transition:.25s;
  }
  .field:hover{transform:translateY(-2px);box-shadow:0 12px 24px -16px var(--c)}
  .field:focus-within{border-color:var(--c);background:#fff;box-shadow:0 0 0 4px color-mix(in srgb,var(--c) 16%,transparent)}
  .ico{
    flex:none;width:44px;height:44px;border-radius:14px;display:grid;place-items:center;font-size:20px;
    background:color-mix(in srgb,var(--c) 15%,white);
  }
  .f-body{flex:1;min-width:0}
  label{display:block;font-size:11px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.7px}
  input,select{
    width:100%;border:none;background:transparent;font:inherit;font-size:17px;font-weight:600;
    color:var(--ink);padding:4px 0 0;outline:none;appearance:none;-moz-appearance:textfield;
  }
  input::placeholder{color:#B9BDD3;font-weight:400}
  input::-webkit-outer-spin-button,input::-webkit-inner-spin-button{-webkit-appearance:none}
  select{cursor:pointer}

  /* buttons */
  .actions{display:flex;gap:12px;margin-top:28px;flex-wrap:wrap}
  button{font:inherit;font-weight:600;font-size:15px;border:none;cursor:pointer;padding:15px 28px;border-radius:16px;transition:.25s}
  .primary{
    flex:1;min-width:200px;color:#fff;position:relative;overflow:hidden;
    background:linear-gradient(120deg,var(--indigo),#8A5CF6 55%,var(--coral));
    background-size:200% 100%;box-shadow:0 16px 30px -14px rgba(91,91,240,.9);
  }
  .primary:hover{transform:translateY(-3px);background-position:100% 0}
  .primary:disabled{opacity:.65;cursor:wait;transform:none}
  .ghost{background:#fff;color:var(--ink);border:1.5px solid var(--line)}
  .ghost:hover{border-color:var(--indigo);color:var(--indigo)}
  .spin{display:inline-block;width:14px;height:14px;border:2px solid rgba(255,255,255,.4);border-top-color:#fff;border-radius:50%;animation:rot .7s linear infinite;vertical-align:-2px;margin-right:8px}
  @keyframes rot{to{transform:rotate(360deg)}}

  /* result */
  #result{display:none;animation:pop .5s cubic-bezier(.2,.9,.3,1.2);position:relative;overflow:hidden}
  #result::before{content:"";position:absolute;left:0;top:0;bottom:0;width:8px;background:var(--rc,var(--teal))}
  @keyframes pop{from{opacity:0;transform:translateY(18px) scale(.98)}to{opacity:1;transform:none}}
  .res{display:flex;gap:34px;align-items:center;flex-wrap:wrap}
  .ring{position:relative;width:150px;height:150px;flex:none}
  .ring svg{width:100%;height:100%;transform:rotate(-90deg)}
  .ring circle{fill:none;stroke-width:11;stroke-linecap:round}
  .r-bg{stroke:var(--line)}
  #ringFill{stroke:var(--rc,var(--teal));stroke-dasharray:326.7;stroke-dashoffset:326.7;transition:stroke-dashoffset 1s ease}
  .ring-txt{position:absolute;inset:0;display:grid;place-content:center;text-align:center}
  .ring-txt b{font-size:30px;font-weight:800;line-height:1}
  .ring-txt span{font-size:11px;color:var(--muted);font-weight:500;margin-top:4px}
  .res-main{flex:1;min-width:220px}
  .res-small{font-size:12px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:1px}
  .res-label{font-size:36px;font-weight:800;letter-spacing:-.6px;color:var(--rc,var(--teal));margin:2px 0 6px}
  .bars{margin-top:14px;display:grid;gap:12px}
  .bar-row{display:grid;grid-template-columns:120px 1fr 56px;align-items:center;gap:12px;font-size:13px;font-weight:500}
  .track{height:10px;background:var(--line);border-radius:999px;overflow:hidden}
  .fill{height:100%;border-radius:999px;width:0;transition:width .9s ease}
  .error{color:var(--coral);font-weight:600;font-size:14px}
  footer{text-align:center;margin-top:30px;color:var(--muted);font-size:12px}
  @media(max-width:560px){
    header{padding:30px 24px}header h1{font-size:28px}
    .card{padding:22px}.bar-row{grid-template-columns:90px 1fr 48px}
  }
</style>
</head>
<body>
<div class="blob b1"></div><div class="blob b2"></div><div class="blob b3"></div>

<div class="wrap">
  <header>
    <span class="badge"><span class="pulse"></span> AdaBoost Classifier &middot; Live</span>
    <h1>Smart Prediction Studio</h1>
    <p>Enter the details below and let the boosted ensemble model make the call, with a full confidence breakdown.</p>
    <div class="chips">
      <div class="chip-s"><b>{{ features|length }}</b>Input fields</div>
      <div class="chip-s"><b>{{ n_estimators }}</b>Weak learners</div>
      <div class="chip-s"><b>{{ n_classes }}</b>Possible outcomes</div>
    </div>
  </header>

  <div class="card">
    <div class="card-head">
      <h2><span class="tag"></span> Input Details</h2>
      <div class="progress"><span id="pText">0 / {{ features|length }} filled</span><div class="p-track"><div class="p-fill" id="pFill"></div></div></div>
    </div>
    <div class="grid">
      {% for f in features %}
      <div class="field" style="--c:{{ f.color }}">
        <div class="ico">{{ f.icon }}</div>
        <div class="f-body">
          <label for="f{{ loop.index0 }}">{{ f.label }}</label>
          {% if f.options %}
          <select id="f{{ loop.index0 }}" onchange="updateProgress()">
            {% for text, val in f.options %}
            <option value="{{ val }}" {% if val == f.default %}selected{% endif %}>{{ text }}</option>
            {% endfor %}
          </select>
          {% else %}
          <input type="number" step="{{ f.step }}" id="f{{ loop.index0 }}"
                 {% if f.min != '' %}min="{{ f.min }}"{% endif %} {% if f.max != '' %}max="{{ f.max }}"{% endif %}
                 value="{{ f.default }}" placeholder="Enter value" oninput="updateProgress()">
          {% endif %}
        </div>
      </div>
      {% endfor %}
    </div>
    <div class="actions">
      <button class="primary" id="go" onclick="predict()">Predict Now &rarr;</button>
      <button class="ghost" onclick="clearForm()">Clear</button>
      <button class="ghost" onclick="resetForm()">Sample values</button>
    </div>
  </div>

  <div class="card" id="result">
    <div class="res">
      <div class="ring" id="ringBox">
        <svg viewBox="0 0 120 120">
          <circle class="r-bg" cx="60" cy="60" r="52"></circle>
          <circle id="ringFill" cx="60" cy="60" r="52"></circle>
        </svg>
        <div class="ring-txt"><b id="ringNum">0%</b><span>confidence</span></div>
      </div>
      <div class="res-main">
        <div class="res-small">Predicted outcome</div>
        <div class="res-label" id="chip"></div>
        <div class="bars" id="bars"></div>
      </div>
    </div>
  </div>

  <div class="card" id="errbox" style="display:none"><p class="error" id="errtext"></p></div>

  <footer>Built with Flask &amp; scikit-learn &middot; AdaBoost ensemble</footer>
</div>

<script>
const COUNT = {{ features|length }};
const DEFAULTS = {{ defaults|tojson }};

function el(i){ return document.getElementById('f'+i); }

function updateProgress(){
  let filled = 0;
  for(let i=0;i<COUNT;i++){ if(el(i).value !== '') filled++; }
  document.getElementById('pText').textContent = filled + ' / ' + COUNT + ' filled';
  document.getElementById('pFill').style.width = (filled/COUNT*100) + '%';
}

function resetForm(){
  for(let i=0;i<COUNT;i++){ el(i).value = DEFAULTS[i]; }
  hideAll(); updateProgress();
}
function clearForm(){
  for(let i=0;i<COUNT;i++){
    if(el(i).tagName === 'SELECT'){ el(i).selectedIndex = 0; } else { el(i).value = ''; }
  }
  hideAll(); updateProgress();
}
function hideAll(){
  document.getElementById('result').style.display='none';
  document.getElementById('errbox').style.display='none';
}

async function predict(){
  const btn = document.getElementById('go');
  const values = [];
  for(let i=0;i<COUNT;i++){
    const v = el(i).value;
    if(v === ''){ el(i).focus(); return showError('Please fill in all the fields.'); }
    values.push(parseFloat(v));
  }
  btn.disabled = true; btn.innerHTML = '<span class="spin"></span>Predicting...';
  document.getElementById('errbox').style.display='none';
  try{
    const res = await fetch('/predict',{
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({values})
    });
    const data = await res.json();
    if(!res.ok){ throw new Error(data.error || 'Something went wrong'); }
    showResult(data);
  }catch(e){
    showError(e.message);
  }finally{
    btn.disabled = false; btn.innerHTML = 'Predict Now &rarr;';
  }
}

function showResult(d){
  const box = document.getElementById('result');
  box.style.setProperty('--rc', d.color);
  document.getElementById('chip').textContent = d.label;

  const ringBox = document.getElementById('ringBox');
  const ring = document.getElementById('ringFill');
  if(d.confidence !== null){
    ringBox.style.display = 'block';
    document.getElementById('ringNum').textContent = d.confidence + '%';
    ring.style.strokeDashoffset = 326.7;
  } else {
    ringBox.style.display = 'none';
  }

  const bars = document.getElementById('bars');
  bars.innerHTML = '';
  (d.probs || []).forEach(p=>{
    const row = document.createElement('div');
    row.className = 'bar-row';
    row.innerHTML = '<span>'+p.label+'</span><div class="track"><div class="fill" style="background:'+p.color+'"></div></div><span>'+p.value+'%</span>';
    bars.appendChild(row);
  });

  box.style.display='block';
  box.scrollIntoView({behavior:'smooth', block:'center'});
  setTimeout(()=>{
    if(d.confidence !== null){ ring.style.strokeDashoffset = 326.7 * (1 - d.confidence/100); }
    document.querySelectorAll('.fill').forEach((f,i)=>{ f.style.width = d.probs[i].value + '%'; });
  }, 80);
}

function showError(msg){
  document.getElementById('errtext').textContent = msg;
  document.getElementById('errbox').style.display='block';
}

updateProgress();
</script>
</body>
</html>
"""


# --------------------------------------------------------------------------
# 5. ROUTES
# --------------------------------------------------------------------------
@app.route("/")
def home():
    defaults = [f["default"] for f in FEATURES]
    n_estimators = len(getattr(model, "estimators_", [])) or "-"
    n_classes = len(getattr(model, "classes_", [])) or "-"
    return render_template_string(
        PAGE,
        features=FEATURES,
        defaults=defaults,
        n_estimators=n_estimators,
        n_classes=n_classes,
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        values = request.get_json(force=True).get("values", [])
        if len(values) != len(FEATURE_NAMES):
            return jsonify({"error": f"Expected {len(FEATURE_NAMES)} values."}), 400

        if USE_NAMES:
            X = pd.DataFrame([values], columns=FEATURE_NAMES)
        else:
            X = np.array(values, dtype=float).reshape(1, -1)

        pred = model.predict(X)[0]
        classes = list(model.classes_) if hasattr(model, "classes_") else [pred]
        idx = classes.index(pred) if pred in classes else 0

        probs, confidence = [], None
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(X)[0]
            confidence = round(float(p[idx]) * 100, 1)
            probs = [
                {
                    "label": label_for(c),
                    "value": round(float(p[i]) * 100, 1),
                    "color": RESULT_COLORS[i % len(RESULT_COLORS)],
                }
                for i, c in enumerate(classes)
            ]

        return jsonify(
            {
                "label": label_for(pred),
                "color": RESULT_COLORS[idx % len(RESULT_COLORS)],
                "confidence": confidence,
                "probs": probs,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)