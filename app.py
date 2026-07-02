import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import io
import os
import json
import lime.lime_tabular
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="EduRisk — UTM Student Risk Dashboard",
    page_icon=":material/school:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS  — clean, clinical, data-first
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg:        #f6f8fb;
    --surface:   #ffffff;
    --surface2:  #ffffff;
    --surface3:  #f1f5f9;
    --border:    #dbe3ee;
    --border2:   #b8c4d6;
    --accent:    #2563eb;
    --accent-lo: rgba(37,99,235,0.10);
    --accent-hi: #1d4ed8;
    --danger:    #dc2626;
    --danger-lo: rgba(220,38,38,0.10);
    --safe:      #16a34a;
    --safe-lo:   rgba(22,163,74,0.10);
    --warn:      #d97706;
    --warn-lo:   rgba(217,119,6,0.12);
    --text:      #172033;
    --text2:     #475569;
    --text3:     #64748b;
    --mono:      'JetBrains Mono', monospace;
    --sans:      'Inter', sans-serif;
    --radius:    8px;
    --radius-lg: 12px;
    --shadow:    0 10px 30px rgba(15,23,42,0.06);
}

html, body, [class*="css"] {
    font-family: var(--sans);
    background: var(--bg);
    color: var(--text);
    -webkit-font-smoothing: antialiased;
}

.stApp { background: var(--bg); }
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1220px; }

/* ── Sidebar ── */
div[data-testid="stSidebar"],
div[data-testid="stSidebar"] > div,
div[data-testid="stSidebar"] > div:first-child,
section[data-testid="stSidebar"] {
    background: #ffffff !important;
    background-color: #ffffff !important;
    border-right: 1px solid var(--border) !important;
    box-shadow: 8px 0 30px rgba(15,23,42,0.04);
}
div[data-testid="stSidebarContent"] {
    background: #ffffff !important;
    background-color: #ffffff !important;
}d="stSidebar"] .stMarkdown p,
div[data-testid="stSidebar"] label { font-size: 0.82rem; }
div[data-testid="stSidebar"] hr { border-color: var(--border); }

[data-baseweb="radio"] div,
[data-baseweb="select"] div,
.stTextInput input,
.stFileUploader section {
    border-color: var(--border) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: var(--accent) !important;
    color: #fff !important;
    border: none !important;
    border-radius: var(--radius) !important;
    font-family: var(--sans) !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0 !important;
    padding: 0.5rem 1.2rem !important;
    width: 100% !important;
    box-shadow: 0 8px 18px rgba(37,99,235,.18) !important;
    transition: transform .15s, box-shadow .15s, background .15s !important;
}
.stButton > button:hover { background:#1d4ed8 !important; transform:translateY(-1px); }
.stDownloadButton > button {
    border-radius: var(--radius) !important;
    border-color: var(--border) !important;
    background: var(--surface) !important;
    color: var(--text) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-lg) !important;
    padding: .25rem !important;
    gap: .25rem !important;
    box-shadow: var(--shadow);
}
.stTabs [data-baseweb="tab"] {
    color: var(--text3) !important;
    font-size: 0.82rem !important;
    font-weight: 650 !important;
    padding: 0.65rem 1rem !important;
    border-radius: var(--radius) !important;
    border: 0 !important;
    background: transparent !important;
}
.stTabs [aria-selected="true"] {
    color: var(--accent-hi) !important;
    background: var(--accent-lo) !important;
}

/* ── KPI row ── */
.kpi-row { display:flex; gap:.75rem; margin-bottom:1.75rem; flex-wrap:wrap; }
.kpi-card {
    flex:1; min-width:120px;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1rem 1.2rem;
    box-shadow: var(--shadow);
}
.kpi-val   { font-size:1.9rem; font-weight:800; line-height:1; margin-bottom:.35rem; font-variant-numeric: tabular-nums; }
.kpi-label { color:var(--text3); font-size:0.68rem; font-weight:600; text-transform:uppercase; letter-spacing:.08em; }
.kpi-danger .kpi-val { color:var(--danger); }
.kpi-safe   .kpi-val { color:var(--safe);   }
.kpi-warn   .kpi-val { color:var(--warn);   }
.kpi-blue   .kpi-val { color:var(--accent-hi); }

/* main page website sections */
.page-hero {
    background: linear-gradient(135deg, #ffffff 0%, #eef4ff 100%);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 1.45rem 1.6rem;
    margin-bottom: 1.15rem;
    box-shadow: var(--shadow);
}
.hero-grid {
    display: grid;
    grid-template-columns: minmax(0, 1.5fr) minmax(260px, .8fr);
    gap: 1.2rem;
    align-items: center;
}
.hero-eyebrow {
    color: var(--accent-hi);
    font-size: .72rem;
    font-weight: 800;
    letter-spacing: .09em;
    text-transform: uppercase;
    margin-bottom: .45rem;
}
.hero-title {
    color: var(--text);
    font-size: 2.15rem;
    font-weight: 850;
    line-height: 1.08;
    margin: 0;
}
.hero-copy {
    color: var(--text2);
    font-size: .96rem;
    line-height: 1.65;
    margin: .65rem 0 0;
    max-width: 680px;
}
.status-panel {
    background: rgba(255,255,255,.82);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1rem;
}
.status-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: .75rem;
    border-bottom: 1px solid var(--border);
    padding: .65rem 0;
    color: var(--text2);
    font-size: .84rem;
}
.status-item:first-child { padding-top: 0; }
.status-item:last-child { border-bottom: 0; padding-bottom: 0; }
.status-pill {
    border-radius: 999px;
    padding: .18rem .6rem;
    font-size: .7rem;
    font-weight: 800;
    white-space: nowrap;
}
.status-ready { background: var(--safe-lo); color: var(--safe); }
.status-wait { background: var(--warn-lo); color: var(--warn); }
.main-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1rem 1.1rem;
    box-shadow: var(--shadow);
    height: 100%;
}
.main-card-title {
    color: var(--text);
    font-weight: 800;
    font-size: .98rem;
    margin-bottom: .4rem;
}
.main-card-body {
    color: var(--text2);
    font-size: .84rem;
    line-height: 1.6;
}
.step-list {
    display: grid;
    gap: .75rem;
}
.step-item {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: .9rem 1rem;
    box-shadow: var(--shadow);
}
.step-num {
    color: var(--accent-hi);
    font-size: .7rem;
    font-weight: 800;
    letter-spacing: .08em;
    text-transform: uppercase;
    margin-bottom: .25rem;
}

@media (max-width: 760px) {
    .hero-grid { grid-template-columns: 1fr; }
    .hero-title { font-size: 1.65rem; }
}

/* ── Badges ── */
.badge {
    display:inline-block; padding:.18rem .65rem;
    border-radius:999px; font-size:.7rem; font-weight:700; letter-spacing:.04em;
}
.badge-danger { background:var(--danger-lo); color:var(--danger); border:1px solid rgba(239,68,68,.25); }
.badge-safe   { background:var(--safe-lo);   color:var(--safe);   border:1px solid rgba(34,197,94,.25); }
.badge-warn   { background:var(--warn-lo);   color:var(--warn);   border:1px solid rgba(245,158,11,.25); }

/* ── Risk bar ── */
.risk-bar-wrap { background:var(--border); border-radius:999px; height:4px; width:100%; margin-bottom:3px; }
.risk-bar-fill { height:4px; border-radius:999px; }

/* ── Section label ── */
.sec-label {
    font-size:.68rem; font-weight:700; text-transform:uppercase;
    letter-spacing:.1em; color:var(--text3); margin-bottom:.6rem;
}

/* ── Risk table ── */
.risk-table { width:100%; border-collapse:collapse; font-size:.82rem; }
.risk-table th {
    background:var(--surface3); color:var(--text3);
    font-size:.67rem; font-weight:700; text-transform:uppercase; letter-spacing:.08em;
    padding:.55rem .85rem; border-bottom:1px solid var(--border); text-align:left;
}
.risk-table td { padding:.55rem .85rem; border-bottom:1px solid var(--border); }
.risk-table tr:last-child td { border-bottom:none; }
.risk-table tr:hover td { background:#f8fafc; }

/* ── Student card ── */
.stu-card {
    background:var(--surface2); border:1px solid var(--border);
    border-radius:var(--radius-lg); padding:1.25rem 1.4rem; height:100%;
    box-shadow: var(--shadow);
}

/* ── AI panel ── */
.ai-section {
    background:var(--surface2); border:1px solid var(--border);
    border-radius:var(--radius-lg); padding:1.25rem 1.4rem; margin-top:1rem;
    box-shadow: var(--shadow);
}
.ai-section-title {
    font-size:.7rem; font-weight:700; text-transform:uppercase;
    letter-spacing:.1em; color:var(--accent-hi); margin-bottom:.6rem;
    display:flex; align-items:center; gap:.4rem;
}
.ai-section-body { font-size:.86rem; color:var(--text2); line-height:1.75; }
.ai-section-body li { margin-bottom:.35rem; }
.ai-section-body strong { color:var(--text); font-weight:600; }

/* ── CP card (landing) ── */
.cp-card { border-radius:var(--radius-lg); padding:.9rem 1.1rem; margin-bottom:.5rem; border:1px solid; }
.cp-active   { background:var(--accent-lo); border-color:rgba(37,99,235,.28); }
.cp-inactive { background:var(--surface2);  border-color:var(--border); }

/* ── Info box ── */
.info-box {
    background:var(--accent-lo); border-left:3px solid var(--accent);
    border-radius:0 var(--radius) var(--radius) 0;
    padding:.85rem 1rem; margin-bottom:1.25rem; font-size:.83rem; color:var(--text2);
}

/* ── Dataframe tweaks ── */
.stDataFrame { border-radius:var(--radius) !important; }
[data-testid="stDataFrameResizable"] { border:1px solid var(--border) !important; border-radius:var(--radius) !important; }

hr { border-color: var(--border); }
code {
    color: var(--accent-hi);
    background: #eef4ff;
    border-radius: 5px;
    padding: .1rem .28rem;
}

/* hide streamlit chrome */
#MainMenu, footer, header { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# CONSTANTS
# ============================================================
CAMPUS_IP_PATTERN = re.compile(r'^(192\.168\.|10\.)')
THRESHOLD = 0.45
# AI provider — swap base URL + model to use any OpenAI-compatible API
# Groq (free):       https://api.groq.com/openai/v1  |  llama-3.3-70b-versatile
# OpenRouter (free): https://openrouter.ai/api/v1    |  meta-llama/llama-3.3-70b-instruct:free
# Together (free):   https://api.together.xyz/v1     |  meta-llama/Llama-3.3-70B-Instruct-Turbo
AI_BASE_URL = "https://openrouter.ai/api/v1"
AI_MODELS = [
    "openrouter/auto",          # auto-picks best available free model
    "meta-llama/llama-3.3-70b-instruct:free",
    "meta-llama/llama-3.1-8b-instruct:free",
    "mistralai/mistral-7b-instruct:free",
]

FEATURE_LABELS = {
    'total_clicks': 'Total Moodle Clicks', 
    'Online_C':     'On-Campus Clicks',
    'Online_O':     'Off-Campus Clicks',
    'CW1':          'Coursework 1 Mark',
    'CW2':          'Coursework 2 Mark',
}

CP_DESCRIPTIONS = {
    'CP1': {
        'title': 'Checkpoint 1 — Engagement Only',
        'desc':  'Moodle log data only. Earliest possible warning, before any marks are available.',
        'feats': ['total_clicks', 'Online_C', 'Online_O'],
    },
    'CP2': {
        'title': 'Checkpoint 2 — Engagement + CW1',
        'desc':  'Refines the prediction after Coursework 1 marks are released.',
        'feats': ['total_clicks', 'Online_C', 'Online_O', 'CW1'],
    },
    'CP3': {
        'title': 'Checkpoint 3 — Engagement + CW1 + CW2',
        'desc':  'Most complete picture. Still time to intervene before the final exam.',
        'feats': ['total_clicks', 'Online_C', 'Online_O', 'CW1', 'CW2'],
    },
}

# ============================================================
# MOODLE LOG PARSER
# ============================================================
def parse_moodle_log(uploaded_file):
    try:
        content = uploaded_file.read()
        df = None
        for enc in ['utf-8', 'utf-8-sig', 'latin-1']:
            try:
                df = pd.read_csv(io.BytesIO(content), encoding=enc)
                break
            except UnicodeDecodeError:
                continue
        if df is None:
            st.error("Could not decode Moodle log file.")
            return None

        df.columns = df.columns.str.strip()

        if 'Description' not in df.columns:
            st.error("Moodle log is missing the 'Description' column.")
            return None

        def extract_moodle_id(desc):
            if not isinstance(desc, str):
                return None
            # Match both numeric IDs ('23') and alphanumeric matrix numbers ('A22EC9998')
            m = re.search(r"user(?:\s+with)?\s+id\s*['\"]?([\w]+)['\"]?", desc, re.IGNORECASE)
            return m.group(1).upper() if m else None

        df['StudentID'] = df['Description'].apply(extract_moodle_id)

        user_col = next(
            (c for c in ['User full name', 'Full name', 'Name', 'User'] if c in df.columns),
            None
        )
        if user_col:
            df['_display_name'] = df[user_col].astype(str).str.strip().str.upper()
            df['_display_name'] = df.apply(
                lambda r: r['StudentID'] if str(r['_display_name']) in ('', '-', 'NAN') else r['_display_name'],
                axis=1
            )

        df = df.dropna(subset=['StudentID'])

        def is_campus(ip):
            if not isinstance(ip, str): return False
            return bool(CAMPUS_IP_PATTERN.match(ip.strip()))

        ip_col = 'IP address' if 'IP address' in df.columns else None
        df['is_campus'] = df[ip_col].apply(is_campus) if ip_col else False

        engagement = (
            df.groupby('StudentID')
            .agg(total_clicks=('StudentID', 'count'), Online_C=('is_campus', 'sum'))
            .reset_index()
        )
        engagement['Online_O'] = engagement['total_clicks'] - engagement['Online_C']

        if '_display_name' in df.columns:
            name_map = (
                df[df['_display_name'] != df['StudentID']]
                .drop_duplicates('StudentID')
                .set_index('StudentID')['_display_name']
            )
            engagement['student_name'] = engagement['StudentID'].map(name_map).fillna(engagement['StudentID'])
        else:
            engagement['student_name'] = engagement['StudentID']

        return engagement

    except Exception as e:
        st.error(f"Failed to parse Moodle log: {e}")
        return None

# ============================================================
# MARKS PARSER
# ============================================================
def parse_marks(uploaded_file):
    try:
        content = uploaded_file.read()
        df = None
        for enc in ['utf-8', 'utf-8-sig', 'latin-1']:
            try:
                df = pd.read_csv(io.BytesIO(content), encoding=enc)
                break
            except UnicodeDecodeError:
                continue
        if df is None:
            st.error("Could not decode marks file.")
            return None

        df.columns = df.columns.str.strip()
        user_col = None
        for alias in ['RollNumber', 'StudentID', 'student_id', 'ID', 'id',
                       'Name', 'StudentName', 'ApplicantName', 'User full name']:
            if alias in df.columns:
                user_col = alias
                break
        if not user_col:
            st.error("Marks CSV must have an ID or Name column.")
            return None

        df['StudentID'] = (df[user_col].astype(str)
                           .str.upper().str.replace(r'\.0$', '', regex=True).str.strip())
        df.drop_duplicates(subset='StudentID', keep='first', inplace=True)
        return df

    except Exception as e:
        st.error(f"Failed to parse marks file: {e}")
        return None

def merge_raw_files(engagement, marks):
    try:
        return pd.merge(engagement, marks, on='StudentID', how='inner')
    except Exception as e:
        st.error(f"Merge error: {e}")
        return None

# ============================================================
# LOAD MODELS
# ============================================================
@st.cache_resource
def load_models():
    path = 'model_backend/checkpoint_models.pkl'
    if not os.path.exists(path):
        return None
    return joblib.load(path)

models   = load_models()
assets_ok = models is not None

# ============================================================
# PREDICTION
# ============================================================
def predict_batch(df_in, cp):
    if not assets_ok: return None
    cp_data = models.get(cp)
    if not cp_data: return None
    feats   = CP_DESCRIPTIONS[cp]['feats']
    missing = [f for f in feats if f not in df_in.columns]
    if missing: return None
    X = df_in[feats].values.astype(float)
    return cp_data['model'].predict_proba(X)[:, 1]

# ============================================================
# LIME
# ============================================================
def lime_figure(instance, cp):
    if not assets_ok: return None, None
    cp_data = models.get(cp)
    if not cp_data: return None, None
    feats = CP_DESCRIPTIONS[cp]['feats']
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=cp_data['explainer_data'],
        feature_names=feats,
        class_names=['Safe', 'At-Risk'],
        mode='classification',
        random_state=42
    )
    exp = explainer.explain_instance(
        instance, cp_data['model'].predict_proba,
        num_features=len(feats), num_samples=3000
    )
    fw = exp.as_list()

    def resolve_label(s):
        for raw, friendly in FEATURE_LABELS.items():
            if raw in s:
                return friendly
        return s

    labels  = [resolve_label(fw[i][0]) for i in range(len(fw)-1, -1, -1)]
    weights = [fw[i][1]                for i in range(len(fw)-1, -1, -1)]
    colors  = ['#dc2626' if w > 0 else '#16a34a' for w in weights]

    fig, ax = plt.subplots(figsize=(9, max(3.5, len(labels) * 0.7)))
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')
    bars = ax.barh(labels, weights, color=colors, height=0.48, zorder=3)
    ax.axvline(0, color='#b8c4d6', linewidth=1, zorder=2)
    ax.grid(axis='x', color='#dbe3ee', linewidth=0.5, zorder=1)
    ax.tick_params(colors='#64748b', labelsize=9.5)
    for sp in ax.spines.values(): sp.set_color('#dbe3ee')
    ax.set_xlabel('← Reduces risk   |   Increases risk →', color='#64748b', fontsize=9)
    ax.set_title(f'Feature Influence — {CP_DESCRIPTIONS[cp]["title"].split("—")[0].strip()}',
                 color='#172033', fontsize=10, pad=10, loc='left')
    plt.tight_layout()

    readable = [(resolve_label(fw[i][0]), fw[i][1]) for i in range(len(fw))]
    return fig, readable

# ============================================================
# HELPERS
# ============================================================
def risk_level(prob):
    if prob >= 0.65:      return 'High Risk'
    if prob >= THRESHOLD: return 'Borderline'
    return 'On Track'

def risk_badge(prob):
    lvl = risk_level(prob)
    cls = {'High Risk':'badge-danger','Borderline':'badge-warn','On Track':'badge-safe'}[lvl]
    return f'<span class="badge {cls}">{lvl}</span>'

def risk_bar_html(prob):
    pct = prob * 100
    col = '#dc2626' if prob >= 0.65 else ('#d97706' if prob >= THRESHOLD else '#16a34a')
    return (f'<div class="risk-bar-wrap">'
            f'<div class="risk-bar-fill" style="width:{pct:.0f}%;background:{col}"></div>'
            f'</div><small style="color:#64748b;font-size:.72rem">{pct:.0f}%</small>')

def checkpoint_short_title(title):
    for sep in ['—', 'â€”', 'Ã¢â‚¬â€', '-']:
        if sep in title:
            return title.split(sep, 1)[1].strip()
    return title

# ============================================================
# GROQ LLM — AI INSIGHT GENERATOR
# ============================================================
def call_ai(prompt: str, api_key: str) -> str:
    import urllib.request
    for model in AI_MODELS:
        payload = json.dumps({
            "model": model,
            #"max_tokens": 1500,
            "temperature": 0.4,
            "messages": [{"role": "user", "content": prompt}],
        }).encode()
        req = urllib.request.Request(
            f"{AI_BASE_URL}/chat/completions",
            data=payload,
            headers={
                "Content-Type":  "application/json",
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer":  "https://edurisk.app",
                "X-Title":       "EduRisk",
            },
            method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                body = json.loads(resp.read())
            return body["choices"][0]["message"]["content"].strip()
        except Exception:
            continue
    return "[AI unavailable: all models failed]"

def build_ai_prompt(name, prob, cp, lime_weights, raw_data):
    risk_str  = risk_level(prob)
    feats_str = "\n".join(
        f"  - {feat}: weight {w:+.4f} ({'increases' if w>0 else 'reduces'} risk)"
        for feat, w in (lime_weights or [])
    )
    data_str = "\n".join(f"  - {k}: {v}" for k, v in raw_data.items())

    return f"""You are an academic advisor assistant. A student risk model has flagged a student.
Provide a concise educator-facing report with EXACTLY these three sections, each as a JSON key:

{{
  "profile_summary": "contextualize the student's current standing based on the raw data.",
  "risk_analysis": ["analysis 1", "analysis 2", "analysis 3", "analysis 4", make min 3 analysis — make lecturers understand (even non technical ones) the factors contributing to the risk score, and what they mean in practical terms." make it in numbers],
  "interventions": ["action 1", "action 2", "action 3", "action 4", make min 4 actions max 8 actions— practical, actionable steps for the educator to help the student succeed."]
}}

Student: {name}
Risk score: {prob*100:.1f}% ({risk_str})
Checkpoint: {cp} — {CP_DESCRIPTIONS[cp]['desc']}

Raw data:
{data_str}

LIME feature weights (positive = pushes toward At-Risk):
{feats_str if feats_str else '  (not available)'}

Return ONLY the JSON object. No markdown fences, no preamble."""


def generate_ai_insight(name, prob, cp, lime_weights, raw_data, groq_key):
    prompt   = build_ai_prompt(name, prob, cp, lime_weights, raw_data)
    raw_resp = call_ai(prompt, groq_key)
    try:
        clean = re.sub(r"```(?:json)?|```", "", raw_resp).strip()
        return json.loads(clean)
    except Exception:
        return {"profile_summary": raw_resp, "risk_analysis": "", "interventions": []}

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## EduRisk")
    st.markdown('<p style="color:#64748b;font-size:.8rem;margin-top:-.3rem">UTM · Student Risk Dashboard</p>',
                unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<p class="sec-label">Checkpoint</p>', unsafe_allow_html=True)
    cp_choice = st.radio(
        "checkpoint",
        ['CP1', 'CP2', 'CP3'],
        format_func=lambda x: {
            'CP1': '1 — Engagement only',
            'CP2': '2 — Engagement + CW1',
            'CP3': '3 — Engagement + CW1 + CW2',
        }[x],
        label_visibility='collapsed'
    )
    cp_info = CP_DESCRIPTIONS[cp_choice]
    st.markdown(f'<div class="info-box"><strong>{cp_info["title"]}</strong><br>'
                f'<span style="font-size:.8rem">{cp_info["desc"]}</span></div>',
                unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p class="sec-label">Data</p>', unsafe_allow_html=True)

    moodle_file = st.file_uploader("Moodle Log CSV", type=["csv"],
                                   help="Course → Administration → Logs → Download table as CSV")
    marks_file  = st.file_uploader("Marks CSV  (Name/ID | CW1 | CW2)", type=["csv"],
                                   help="Gradebook. CW2 required only for CP3.")

    analyse_btn = st.button("Analyse Class", icon=":material/bar_chart:")

    # Load Groq key from .streamlit/secrets.toml
    groq_key = st.secrets.get("AI_API_KEY", "") if hasattr(st, "secrets") else ""
    st.markdown("---")

    if not assets_ok:
        st.markdown("""
        <div style="background:rgba(245,158,11,.08);border:1px solid rgba(245,158,11,.25);
        border-radius:8px;padding:.6rem .85rem;font-size:.76rem;color:#d97706;margin-top:.75rem">
        <strong>Models not loaded.</strong><br>
        Run <code>fyp2_pipeline.py</code> to generate<br>
        <code>model_backend/checkpoint_models.pkl</code>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# MAIN WEBSITE HERO
# ============================================================
active_cp_info = CP_DESCRIPTIONS[cp_choice]
moodle_ready = bool(moodle_file)
marks_ready = bool(marks_file)
model_ready = bool(assets_ok)

st.markdown(f"""
<section class="page-hero">
  <div class="hero-grid">
    <div>
      <div class="hero-eyebrow">UTM Student Risk Dashboard</div>
      <h1 class="hero-title">EduRisk Dashboard</h1>
      <p class="hero-copy">
        Review Moodle engagement, coursework marks, class-level risk, and student-level explanations
        from one clean workspace.
      </p>
    </div>
    <div class="status-panel">
      <div class="status-item">
        <span>Moodle log</span>
        <span class="status-pill {'status-ready' if moodle_ready else 'status-wait'}">{'Ready' if moodle_ready else 'Needed'}</span>
      </div>
      <div class="status-item">
        <span>Marks CSV</span>
        <span class="status-pill {'status-ready' if marks_ready else 'status-wait'}">{'Ready' if marks_ready else 'Needed'}</span>
      </div>
      <div class="status-item">
        <span>Model backend</span>
        <span class="status-pill {'status-ready' if model_ready else 'status-wait'}">{'Ready' if model_ready else 'Demo mode'}</span>
      </div>
      <div class="status-item">
        <span>Selected checkpoint</span>
        <strong style="color:#1d4ed8">{cp_choice}</strong>
      </div>
    </div>
  </div>
</section>
""", unsafe_allow_html=True)

# LANDING PAGE  (shown before first analysis)
# ============================================================
if 'result_df' not in st.session_state:
    home_tab, help_tab = st.tabs(["Dashboard", "How It Works"])

    with home_tab:
        st.markdown('<p class="sec-label">Analysis workspace</p>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div class="main-card">
              <div class="main-card-title">Current Checkpoint</div>
              <div class="main-card-body">
                <strong style="color:#1d4ed8">{active_cp_info['title']}</strong><br>
                {active_cp_info['desc']}
              </div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="main-card">
              <div class="main-card-title">Uploaded Data</div>
              <div class="main-card-body">
                Moodle log: <strong>{'Ready' if moodle_ready else 'Upload needed'}</strong><br>
                Marks CSV: <strong>{'Ready' if marks_ready else 'Upload needed'}</strong>
              </div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="main-card">
              <div class="main-card-title">Next Action</div>
              <div class="main-card-body">
                {'Click Analyse Class in the sidebar to generate the class dashboard.' if moodle_ready and marks_ready else 'Upload both CSV files in the sidebar to begin the analysis.'}
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<p class="sec-label" style="margin-top:1.25rem">Checkpoint coverage</p>',
                    unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        for col, cp in [(col1, 'CP1'), (col2, 'CP2'), (col3, 'CP3')]:
            info = CP_DESCRIPTIONS[cp]
            active = cp == cp_choice
            with col:
                feats_html = "".join(
                    f'<li style="color:#475569;font-size:.8rem">{FEATURE_LABELS.get(f,f)}</li>'
                    for f in info['feats']
                )
                st.markdown(f"""
                <div class="cp-card cp-{'active' if active else 'inactive'}">
                  <div style="font-size:.68rem;color:#1d4ed8;font-weight:700;text-transform:uppercase;
                              letter-spacing:.1em;margin-bottom:.35rem">{cp}</div>
                  <div style="font-weight:700;font-size:.88rem;margin-bottom:.4rem;color:#172033">
                    {checkpoint_short_title(info['title'])}
                  </div>
                  <div style="font-size:.78rem;color:#64748b;margin-bottom:.5rem">{info['desc']}</div>
                  <ul style="padding-left:1rem;margin:0">{feats_html}</ul>
                </div>
                """, unsafe_allow_html=True)

    with help_tab:
        st.markdown('<p class="sec-label">How it works</p>', unsafe_allow_html=True)
        st.markdown("""
        <div class="step-list">
          <div class="step-item">
            <div class="step-num">Step 1</div>
            <div class="main-card-title">Export the Moodle log</div>
            <div class="main-card-body">Course administration logs should be downloaded as a CSV file.</div>
          </div>
          <div class="step-item">
            <div class="step-num">Step 2</div>
            <div class="main-card-title">Prepare the marks CSV</div>
            <div class="main-card-body">Use a student name or ID column with CW1 and CW2 marks where available.</div>
          </div>
          <div class="step-item">
            <div class="step-num">Step 3</div>
            <div class="main-card-title">Choose the checkpoint</div>
            <div class="main-card-body">Select CP1, CP2, or CP3 based on the data available at that semester stage.</div>
          </div>
          <div class="step-item">
            <div class="step-num">Step 4</div>
            <div class="main-card-title">Analyse the class</div>
            <div class="main-card-body">Run the analysis to view class risk, engagement charts, student profiles, LIME, and AI insights.</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<p class="sec-label" style="margin-top:1rem">Sample marks format</p>',
                    unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            'StudentName': ['STUDENT A', 'STUDENT B', 'STUDENT C'],
            'CW1': [29, 18, 16], 'CW2': [34, 36, 18],
        }), use_container_width=True, hide_index=True)

    if not analyse_btn:
        st.stop()


# ============================================================
# PROCESS ON BUTTON CLICK
# ============================================================
if analyse_btn:
    if not moodle_file:
        st.sidebar.error("Upload the Moodle log CSV first.")
        st.stop()
    if cp_choice in ['CP2', 'CP3'] and not marks_file:
        st.sidebar.error("Upload the Marks CSV first.")
        st.stop()

    with st.spinner("Parsing files…"):
        engagement = parse_moodle_log(moodle_file)
        marks = parse_marks(marks_file) if marks_file else None

    if engagement is None or (cp_choice in ['CP2', 'CP3'] and marks is None):
        st.stop()

    if marks is not None:
        with st.spinner("Merging…"):
            df = merge_raw_files(engagement, marks)
        if df is None or df.empty:
            st.error("No students matched. Check that Student IDs in your Marks file match the Moodle log.")
            st.stop()
    else:
        df = engagement.copy()

    if 'student_name' not in df.columns or df['student_name'].isna().all():
        df['student_name'] = df['StudentID']

    for col in ['total_clicks', 'Online_C', 'Online_O']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    if 'CW1' in df.columns:
        df['CW1'] = pd.to_numeric(df['CW1'], errors='coerce').fillna(0)
    if 'CW2' in df.columns:
        df['CW2'] = pd.to_numeric(df['CW2'], errors='coerce').fillna(0)

    if assets_ok:
        probs = predict_batch(df, cp_choice)
        if probs is None:
            st.error(f"Model prediction failed for {cp_choice}. Columns: {list(df.columns)}")
            st.stop()
        df['risk_probability'] = probs
    else:
        st.info("Demo mode — models not loaded. Showing placeholder scores.")
        df['risk_probability'] = 0.5

    df['risk_level'] = df['risk_probability'].apply(risk_level)
    df = df.sort_values('risk_probability', ascending=False).reset_index(drop=True)

    cp_feats = [f for f in CP_DESCRIPTIONS[cp_choice]['feats'] if f in df.columns]
    df['_encoded'] = [df.iloc[i][cp_feats].values.astype(float) for i in range(len(df))]

    st.session_state['result_df']  = df
    st.session_state['active_cp']  = cp_choice
    st.session_state['active_tab'] = 0   # jump to Overview tab

    # Scroll to top of results
    st.markdown('<script>window.scrollTo(0,0);</script>', unsafe_allow_html=True)

# ============================================================
# RESULTS
# ============================================================
if 'result_df' not in st.session_state:
    st.stop()

result_df = st.session_state['result_df']
active_cp = st.session_state.get('active_cp', cp_choice)

if active_cp != cp_choice:
    st.warning(f"Showing results for **{active_cp}**. Click **Analyse Class** to re-run with {cp_choice}.")

n_total  = len(result_df)
n_high   = int((result_df['risk_probability'] >= 0.65).sum())
n_border = int(((result_df['risk_probability'] >= THRESHOLD) &
                (result_df['risk_probability'] < 0.65)).sum())
n_safe   = int((result_df['risk_probability'] < THRESHOLD).sum())
avg_risk = result_df['risk_probability'].mean() * 100

st.markdown(f"""
<div class="kpi-row">
  <div class="kpi-card kpi-blue">
    <div class="kpi-val">{n_total}</div>
    <div class="kpi-label">Students</div>
  </div>
  <div class="kpi-card kpi-danger">
    <div class="kpi-val">{n_high}</div>
    <div class="kpi-label">High Risk</div>
  </div>
  <div class="kpi-card kpi-warn">
    <div class="kpi-val">{n_border}</div>
    <div class="kpi-label">Borderline</div>
  </div>
  <div class="kpi-card kpi-safe">
    <div class="kpi-val">{n_safe}</div>
    <div class="kpi-label">On Track</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-val" style="color:#1d4ed8">{avg_risk:.0f}%</div>
    <div class="kpi-label">Avg Risk</div>
  </div>
</div>
""", unsafe_allow_html=True)

cp_info = CP_DESCRIPTIONS[active_cp]
st.markdown(f"""
<div style="background:rgba(37,99,235,.08);border:1px solid rgba(37,99,235,.20);
border-radius:8px;padding:.65rem 1rem;margin-bottom:1.5rem;font-size:.82rem;">
<strong style="color:#1d4ed8">{cp_info['title']}</strong>
<span style="color:#64748b;margin-left:.5rem">— {cp_info['desc']}</span>
</div>
""", unsafe_allow_html=True)

TAB_LABELS = ["Class Overview", "Engagement & Marks", "Student Profile + LIME"]
active_tab = st.session_state.get('active_tab', 0)

tab1, tab2, tab3 = st.tabs(TAB_LABELS, key="main_tabs", default=TAB_LABELS[active_tab])

# ── TAB 1 ──────────────────────────────────────────────────
with tab1:
    if tab1.open:
        st.session_state['active_tab'] = 0
    nc = 'student_name'
    st.markdown('<p class="sec-label">Student risk list</p>', unsafe_allow_html=True)

    display_df = result_df.copy()

    # Build display columns
    display_df['Risk %'] = (display_df['risk_probability'] * 100).round(1)
    display_df['Level']  = display_df['risk_level']
    display_df['Clicks'] = display_df['total_clicks'].astype(int)

    col_map = {'student_name': 'Name', 'StudentID': 'ID'}
    if 'CW1' in display_df.columns:
        display_df['CW1'] = display_df['CW1'].round(1)
    if 'CW2' in display_df.columns:
        display_df['CW2'] = display_df['CW2'].round(1)

    show_cols = ['student_name', 'StudentID', 'Risk %', 'Level', 'Clicks']
    if 'CW1' in display_df.columns: show_cols.append('CW1')
    if 'CW2' in display_df.columns: show_cols.append('CW2')

    table_df = display_df[show_cols].rename(columns=col_map)

    def color_level(val):
        colors = {
            'High Risk':   'color: #dc2626; font-weight: 700',
            'Borderline':  'color: #d97706; font-weight: 700',
            'On Track':    'color: #16a34a; font-weight: 700',
        }
        return colors.get(val, '')

    def color_risk(val):
        if val >= 65:   return 'color: #dc2626; font-weight: 700'
        if val >= 45:   return 'color: #d97706; font-weight: 700'
        return 'color: #16a34a; font-weight: 700'

    styled = (
        table_df.style
        .applymap(color_level, subset=['Level'])
        .applymap(color_risk,  subset=['Risk %'])
        .format({'Risk %': '{:.1f}%'})
    )

    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        height=min(38 * len(table_df) + 38, 600),
        column_config={
            'Name':    st.column_config.TextColumn('Name',   width='medium'),
            'ID':      st.column_config.TextColumn('ID',     width='small'),
            'Risk %':  st.column_config.NumberColumn('Risk %', format='%.1f%%', width='small'),
            'Level':   st.column_config.TextColumn('Level',  width='small'),
            'Clicks':  st.column_config.NumberColumn('Clicks', width='small'),
            'CW1':     st.column_config.NumberColumn('CW1',   width='small'),
            'CW2':     st.column_config.NumberColumn('CW2',   width='small'),
        }
    )

    st.markdown("<br>", unsafe_allow_html=True)
    csv_data = result_df.drop(columns=['_encoded'], errors='ignore').to_csv(index=False)
    st.download_button("Download results CSV", csv_data, "edurisk_results.csv",
                       "text/csv", icon=":material/download:")

# ── TAB 2 ──────────────────────────────────────────────────
with tab2:
    if tab2.open:
        st.session_state['active_tab'] = 1
    nc = 'student_name'

    fig1, ax1 = plt.subplots(figsize=(10, 3.5))
    fig1.patch.set_facecolor('#ffffff')
    ax1.set_facecolor('#ffffff')
    colors_bar = ['#dc2626' if p >= 0.65 else ('#d97706' if p >= THRESHOLD else '#16a34a')
                  for p in result_df['risk_probability']]
    ax1.bar(range(len(result_df)), result_df['risk_probability'] * 100,
            color=colors_bar, width=0.7, zorder=3)
    ax1.axhline(THRESHOLD * 100, color='#d97706', linewidth=1, linestyle='--', zorder=4)
    ax1.axhline(65, color='#dc2626', linewidth=1, linestyle='--', zorder=4)
    ax1.set_ylabel('Risk %', color='#64748b', fontsize=9)
    ax1.set_xlabel('Students (sorted by risk)', color='#64748b', fontsize=9)
    ax1.tick_params(colors='#64748b', labelsize=8)
    ax1.set_xticks([])
    ax1.grid(axis='y', color='#dbe3ee', linewidth=0.5)
    for sp in ax1.spines.values(): sp.set_color('#dbe3ee')
    plt.tight_layout()
    st.pyplot(fig1, use_container_width=True)
    plt.close(fig1)

    if 'CW1' in result_df.columns:
        fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(10, 3.5))
        fig2.patch.set_facecolor('#ffffff')
        for ax in (ax2, ax3): ax.set_facecolor('#ffffff')

        sc = ax2.scatter(result_df['total_clicks'], result_df['CW1'],
                         c=result_df['risk_probability'],
                         cmap='RdYlGn_r', vmin=0, vmax=1, alpha=0.8, s=50)
        ax2.set_xlabel('Total Clicks', color='#64748b', fontsize=9)
        ax2.set_ylabel('CW1 Mark', color='#64748b', fontsize=9)
        ax2.set_title('Clicks vs CW1', color='#172033', fontsize=10)
        ax2.tick_params(colors='#64748b', labelsize=8)
        for sp in ax2.spines.values(): sp.set_color('#dbe3ee')

        risk_counts = result_df['risk_level'].value_counts()
        pie_colors  = {'High Risk': '#dc2626', 'Borderline': '#d97706', 'On Track': '#16a34a'}
        ax3.pie(risk_counts.values,
                labels=risk_counts.index,
                colors=[pie_colors.get(l, '#1d4ed8') for l in risk_counts.index],
                autopct='%1.0f%%', startangle=90,
                textprops={'color': '#475569', 'fontsize': 9})
        ax3.set_title('Risk Distribution', color='#172033', fontsize=10)
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

    if 'Online_C' in result_df.columns and 'Online_O' in result_df.columns:
        fig3, ax4 = plt.subplots(figsize=(10, 3.5))
        fig3.patch.set_facecolor('#ffffff')
        ax4.set_facecolor('#ffffff')
        show = result_df.head(30).reset_index(drop=True)
        idx  = np.arange(len(show))
        ax4.bar(idx, show['Online_C'], label='On-Campus',  color='#2563eb', alpha=0.85)
        ax4.bar(idx, show['Online_O'], bottom=show['Online_C'], label='Off-Campus', color='#16a34a', alpha=0.85)
        ax4.set_xticks(idx)
        ax4.set_xticklabels([f"S{i+1}" for i in idx], fontsize=6, color='#64748b')
        ax4.set_ylabel('Clicks', color='#64748b', fontsize=9)
        ax4.set_title('On/Off-Campus Breakdown — top 30 by risk', color='#172033', fontsize=10, loc='left')
        ax4.legend(fontsize=8, labelcolor='#475569', facecolor='#ffffff')
        ax4.tick_params(colors='#64748b', labelsize=8)
        ax4.grid(axis='y', color='#dbe3ee', linewidth=0.5)
        for sp in ax4.spines.values(): sp.set_color('#dbe3ee')
        plt.tight_layout()
        st.pyplot(fig3, use_container_width=True)
        plt.close(fig3)

# ── TAB 3 ──────────────────────────────────────────────────
with tab3:
    if tab3.open:
        st.session_state['active_tab'] = 2
    nc = 'student_name'

    st.markdown("""
    <div class="info-box">
    Select a student to see their LIME explanation and AI-generated analysis.&nbsp;
    <span style="color:#dc2626">●</span> Red = pushes toward At-Risk &nbsp;
    <span style="color:#16a34a">●</span> Green = reduces risk.
    </div>
    """, unsafe_allow_html=True)

    select_mode = st.radio("Find student", ["Browse by Risk Rank", "Search by Name"],
                           horizontal=True, label_visibility="collapsed")
    final_idx = 0

    if select_mode == "Search by Name":
        drill_search = st.text_input("Name", placeholder="Type a name…", label_visibility="collapsed")
        if drill_search.strip():
            matches = result_df[result_df[nc].astype(str).str.contains(
                drill_search.strip(), case=False, na=False)]
            if not matches.empty:
                if len(matches) > 1:
                    chosen    = st.selectbox("Multiple matches:", matches[nc].tolist())
                    final_idx = int(matches[matches[nc] == chosen].index[0])
                else:
                    final_idx = int(matches.index[0])
            else:
                st.warning("No match found.")
    else:
        rank_options = [
            f"#{i+1}  {row[nc]}  —  {row['risk_probability']*100:.1f}%  [{risk_level(row['risk_probability'])}]"
            for i, row in result_df.iterrows()
        ]
        chosen_label = st.selectbox("Select", rank_options, label_visibility="collapsed")
        final_idx    = rank_options.index(chosen_label)

    st.markdown("---")
    selected  = result_df.iloc[final_idx]
    name_disp = selected[nc]
    prob_val  = selected['risk_probability']
    prob_col  = '#dc2626' if prob_val >= 0.65 else ('#d97706' if prob_val >= THRESHOLD else '#16a34a')

    # ── Student card + data table ──
    left, right = st.columns([1, 2], gap="large")
    with left:
        st.markdown(f"""
        <div class="stu-card">
          <div class="sec-label">Student</div>
          <div style="font-size:1.15rem;font-weight:800;color:#172033">{name_disp}</div>
          <div style="font-size:.78rem;color:#64748b;font-family:var(--mono);margin-top:.2rem">
            {selected.get('StudentID','—')}
          </div>
          <div style="border-top:1px solid #dbe3ee;margin-top:1rem;padding-top:1rem">
            <div class="sec-label">Risk Score</div>
            <div style="font-size:2.6rem;font-weight:800;line-height:1;color:{prob_col};
                        font-variant-numeric:tabular-nums">{prob_val*100:.1f}%</div>
            <div style="margin-top:.5rem">{risk_badge(prob_val)}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with right:
        prof_rows = []
        for col, label in [('total_clicks','Total Moodle Clicks'),('Online_C','On-Campus Clicks'),
                            ('Online_O','Off-Campus Clicks'),('CW1','CW1 Mark'),('CW2','CW2 Mark')]:
            if col in selected.index:
                try:    val = round(float(selected[col]), 1)
                except: val = selected[col]
                prof_rows.append({'Feature': label, 'Value': val})
        if prof_rows:
            st.dataframe(pd.DataFrame(prof_rows), use_container_width=True, hide_index=True, height=215)

    # ── LIME chart ──
    st.markdown("<br>", unsafe_allow_html=True)
    encoded = selected.get('_encoded', None)
    lime_weights = None

    if encoded is not None and not isinstance(encoded, float):
        with st.spinner("Computing LIME explanation…"):
            if assets_ok:
                fig_lime, lime_weights = lime_figure(np.array(encoded), active_cp)
                if fig_lime:
                    st.pyplot(fig_lime, use_container_width=True)
                    plt.close(fig_lime)
            else:
                st.info("LIME requires trained models. Run `fyp2_pipeline.py` first.")
    else:
        st.warning("Feature data not available for this student.")

    # ── AI Insight panels ──
    st.markdown("<br>", unsafe_allow_html=True)

    raw_data = {}
    for col, label in [('total_clicks','Total Moodle Clicks'),('Online_C','On-Campus Clicks'),
                        ('Online_O','Off-Campus Clicks'),('CW1','CW1 Mark'),('CW2','CW2 Mark')]:
        if col in selected.index:
            try:    raw_data[label] = round(float(selected[col]), 1)
            except: raw_data[label] = selected[col]

    use_groq = bool(groq_key and groq_key.strip())

    if use_groq:
        cache_key = f"ai_{name_disp}_{active_cp}_{prob_val:.3f}"
        if cache_key not in st.session_state:
            with st.spinner("Generating AI insight…"):
                st.session_state[cache_key] = generate_ai_insight(
                    name_disp, prob_val, active_cp, lime_weights, raw_data, groq_key.strip()
                )
        insight = st.session_state[cache_key]

        def format_ai_html(content):
            if isinstance(content, list):
                return "".join(f"<p>{item}</p>" for item in content)
            return f"<p>{content}</p>"

        def ai_panel(icon, title, body_html, border_color):
            st.markdown(f"""
            <div class="ai-section" style="border-color:{border_color}40">
              <div class="ai-section-title" style="color:{border_color}">
                {icon} {title}
              </div>
              <div class="ai-section-body">{body_html}</div>
            </div>
            """, unsafe_allow_html=True)

        ai_panel("◈", "Profile Summary",
                 format_ai_html(insight.get('profile_summary','—')),
                 "#1d4ed8")

        ai_panel("◉", "Risk Analysis — Why?",
                 format_ai_html(insight.get('risk_analysis','—')),
                 "#d97706")

        interventions = insight.get('interventions', [])
        if isinstance(interventions, list) and interventions:
            items = "".join(f"<li>{it}</li>" for it in interventions)
        else:
            items = f"<li>{interventions}</li>" if interventions else "<li>No suggestions generated.</li>"
        ai_panel("◎", "Suggested Interventions",
                 f"<ol style='padding-left:1.1rem;margin:0'>{items}</ol>",
                 "#16a34a")

    else:
        # Rule-based fallback using LIME weights (no API key needed)
        def rule_based_insight(name, prob, lime_weights, raw_data):
            risk_str = risk_level(prob)

            # Profile summary from raw data
            parts = []
            if 'Total Moodle Clicks' in raw_data:
                clicks = raw_data['Total Moodle Clicks']
                if clicks < 200:   parts.append("very low Moodle engagement")
                elif clicks < 380: parts.append("below-average Moodle engagement")
                else:              parts.append("adequate Moodle engagement")
            if 'CW1 Mark' in raw_data:
                cw1 = raw_data['CW1 Mark']
                if cw1 < 30:   parts.append("a critically low CW1 mark")
                elif cw1 < 50: parts.append("a below-average CW1 mark")
                else:          parts.append("a satisfactory CW1 mark")
            if 'CW2 Mark' in raw_data:
                cw2 = raw_data['CW2 Mark']
                if cw2 < 30:   parts.append("a critically low CW2 mark")
                elif cw2 < 50: parts.append("a below-average CW2 mark")
                else:          parts.append("a satisfactory CW2 mark")
            summary = f"{name} is currently flagged as <strong>{risk_str}</strong> with {', and '.join(parts)}." if parts else f"{name} is flagged as <strong>{risk_str}</strong>."

            # Risk analysis from LIME weights — pick top positive driver
            analysis = ""
            if lime_weights:
                top_risks    = [(f, w) for f, w in lime_weights if w > 0]
                top_reducing = [(f, w) for f, w in lime_weights if w < 0]
                top_risks.sort(key=lambda x: x[1], reverse=True)
                top_reducing.sort(key=lambda x: x[1])
                if top_risks:
                    top_feat = top_risks[0][0]
                    analysis += f"The primary risk driver is <strong>{top_feat}</strong>. "
                if len(top_risks) > 1:
                    analysis += f"<strong>{top_risks[1][0]}</strong> is also contributing to the elevated risk. "
                if top_reducing:
                    analysis += f"<strong>{top_reducing[0][0]}</strong> is currently working in the student's favour."
            if not analysis:
                analysis = "Insufficient feature data to determine primary risk drivers."

            # Interventions from LIME weights
            interventions = []
            weight_map = dict(lime_weights) if lime_weights else {}
            if weight_map.get('Total Moodle Clicks', 0) > 0:
                interventions.append("Send a direct message via Moodle encouraging the student to log in and engage with course materials regularly.")
            if weight_map.get('On-Campus Clicks', 0) > 0:
                interventions.append("Encourage the student to attend on-campus sessions or lab hours to increase face-to-face engagement.")
            if weight_map.get('Off-Campus Clicks', 0) > 0:
                interventions.append("Follow up to check whether the student has reliable off-campus internet access for remote study.")
            if weight_map.get('Coursework 1 Mark', 0) > 0:
                interventions.append("Schedule a one-to-one academic review session to address weaknesses identified in Coursework 1.")
            if weight_map.get('Coursework 2 Mark', 0) > 0:
                interventions.append("Provide targeted feedback on Coursework 2 and connect the student with peer study groups.")
            if not interventions:
                interventions.append("Monitor the student's engagement weekly and flag for personal tutor review if no improvement is observed.")

            return summary, analysis, interventions

        summary, analysis, interventions = rule_based_insight(
            name_disp, prob_val, lime_weights, raw_data
        )

        def fallback_panel(icon, title, body_html, border_color):
            st.markdown(f"""
            <div class="ai-section" style="border-color:{border_color}40">
              <div class="ai-section-title" style="color:{border_color}">
                {icon} {title}
              </div>
              <div class="ai-section-body">{body_html}</div>
            </div>
            """, unsafe_allow_html=True)

        fallback_panel("◈", "Profile Summary",
                       f"<p>{summary}</p>", "#1d4ed8")
        fallback_panel("◉", "Risk Analysis — Why?",
                       f"<p>{analysis}</p>", "#d97706")
        items = "".join(f"<li>{i}</li>" for i in interventions)
        fallback_panel("◎", "Suggested Interventions",
                       f"<ol style='padding-left:1.1rem;margin:0'>{items}</ol>", "#16a34a")
