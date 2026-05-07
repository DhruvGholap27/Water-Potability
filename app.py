import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import json
import datetime
import tempfile
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# WHO / EPA safe limits for each parameter
WHO_LIMITS = {
    "ph":              {"min": 6.5,  "max": 8.5,   "unit": "",       "label": "pH Level"},
    "Hardness":        {"min": 0,    "max": 300.0,  "unit": "mg/L",   "label": "Hardness"},
    "Solids":          {"min": 0,    "max": 500.0,  "unit": "ppm",    "label": "Total Solids"},
    "Chloramines":     {"min": 0,    "max": 4.0,    "unit": "ppm",    "label": "Chloramines"},
    "Sulfate":         {"min": 0,    "max": 250.0,  "unit": "mg/L",   "label": "Sulfate"},
    "Conductivity":    {"min": 0,    "max": 400.0,  "unit": "μS/cm",  "label": "Conductivity"},
    "Organic_carbon":  {"min": 0,    "max": 2.0,    "unit": "ppm",    "label": "Organic Carbon"},
    "Trihalomethanes": {"min": 0,    "max": 80.0,   "unit": "μg/L",   "label": "Trihalomethanes"},
    "Turbidity":       {"min": 0,    "max": 5.0,    "unit": "NTU",    "label": "Turbidity"},
}

TREATMENT_TIPS = {
    "ph":              ("Adjust pH with lime (if acidic) or CO₂ injection / acid dosing (if alkaline).", "Low-cost chemical treatment"),
    "Hardness":        ("Use ion-exchange water softener or reverse osmosis (RO) filtration.", "Household water softener"),
    "Solids":          ("Install an RO system or distillation unit to remove dissolved solids.", "RO system (~$150–400)"),
    "Chloramines":     ("Use activated carbon / GAC filter; standard carbon filters remove chloramines.", "Carbon filter cartridge"),
    "Sulfate":         ("RO filtration or distillation effectively removes sulfates.", "RO system"),
    "Conductivity":    ("High conductivity indicates excessive ions — use RO or deionization.", "RO or DI system"),
    "Organic_carbon":  ("Activated carbon adsorption + UV disinfection removes organic compounds.", "Carbon + UV filter"),
    "Trihalomethanes": ("Activated carbon block filter (NSF/ANSI 53 certified) removes THMs effectively.", "Certified carbon filter"),
    "Turbidity":       ("Use a sediment pre-filter (5 micron) or ceramic filter to reduce turbidity.", "Sediment filter (~$20–50)"),
}


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Water Potability Prediction",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CUSTOM CSS FOR PREMIUM LOOK
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ═══════════════════════════════════════════
   GLOBAL RESET & TYPOGRAPHY
   ═══════════════════════════════════════════ */
*,html,body,h1,h2,h3,h4,h5,p,span,li,label,
div:not([class*="stIcon"]),
input,textarea,select,button {
    font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif !important;
}

/* ═══════════════════════════════════════════
   OLED BLACK CANVAS
   ═══════════════════════════════════════════ */
.stApp,[data-testid="stAppViewContainer"],
section[data-testid="stMain"]{background:#000!important}
header[data-testid="stHeader"]{display:none!important}
[data-testid="stToolbar"]{display:none!important}
[data-testid="stDecoration"]{display:none!important}
#MainMenu{visibility:hidden!important}
footer{visibility:hidden!important}

/* ═══════════════════════════════════════════
   ANIMATIONS
   ═══════════════════════════════════════════ */
@keyframes fadeInUp{
    from{opacity:0;transform:translateY(24px)}
    to{opacity:1;transform:translateY(0)}
}
@keyframes glowPulse{
    0%,100%{box-shadow:0 0 0 rgba(255,255,255,0)}
    50%{box-shadow:0 0 20px rgba(255,255,255,0.04)}
}

.main .block-container{
    animation:fadeInUp .7s cubic-bezier(.22,1,.36,1) both;
    max-width:1060px;
    padding:5rem 2rem 4rem;
}

/* ═══════════════════════════════════════════
   FLOATING NAVBAR
   ═══════════════════════════════════════════ */
.pro-navbar{
    position:fixed;top:0;left:0;right:0;z-index:999999;
    display:flex;align-items:center;justify-content:space-between;
    padding:.85rem 2.5rem;
    background:rgba(0,0,0,.65);
    backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px);
    border-bottom:1px solid rgba(255,255,255,.06);
}
.pro-navbar .logo{
    font-size:1.15rem;font-weight:800;color:#fff;
    letter-spacing:1.5px;text-transform:uppercase;
    display:flex;align-items:center;gap:.5rem;
}
.pro-navbar .logo span{font-weight:300;color:#666;letter-spacing:0}
.pro-navbar .nav-links{
    display:flex;gap:2rem;align-items:center;
}
.pro-navbar .nav-links a{
    color:#777;font-size:.78rem;font-weight:500;
    text-decoration:none;text-transform:uppercase;
    letter-spacing:1.8px;
    transition:color .25s ease;
}
.pro-navbar .nav-links a:hover{color:#fff}
.pro-navbar .nav-links a.active{color:#fff}
.pro-navbar .nav-cta{
    background:#fff;color:#000;
    padding:.45rem 1.3rem;border-radius:6px;
    font-size:.75rem;font-weight:700;
    text-transform:uppercase;letter-spacing:1.5px;
    text-decoration:none;
    transition:all .2s ease;
}
.pro-navbar .nav-cta:hover{
    background:#e0e0e0;transform:translateY(-1px);
}
@media(max-width:768px){
    .pro-navbar .nav-links{display:none}
    .pro-navbar{padding:.7rem 1.2rem}
}

/* ═══════════════════════════════════════════
   GLASS CARD  — THE CORE COMPONENT
   ═══════════════════════════════════════════ */
.glass-card{
    background:rgba(255,255,255,.02);
    backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);
    border:1px solid #1a1a1a;
    border-radius:20px;
    padding:2.2rem 2rem;
    margin-bottom:1.8rem;
    transition:all .3s cubic-bezier(.22,1,.36,1);
    animation:fadeInUp .6s cubic-bezier(.22,1,.36,1) both;
}
.glass-card:hover{
    border-color:#333;
    box-shadow:0 0 40px rgba(255,255,255,.025);
}

/* ═══════════════════════════════════════════
   HERO / PAGE HEADER
   ═══════════════════════════════════════════ */
.main-header{
    background:transparent;
    border:none;
    padding:2.5rem 0 1.5rem;
    margin-bottom:1rem;
    color:#fff;text-align:center;
    position:relative;
}
.main-header h1{
    font-size:3rem;font-weight:900;margin:0;
    letter-spacing:-1.5px;line-height:1.1;
    color:#fff;-webkit-text-fill-color:#fff;background:none;
}
.main-header p{
    font-size:1rem;color:#555;margin-top:.8rem;
    font-weight:400;letter-spacing:1.5px;
    text-transform:uppercase;
}

/* ═══════════════════════════════════════════
   SECTION TITLES
   ═══════════════════════════════════════════ */
.section-label{
    font-size:.7rem;font-weight:600;color:#555;
    text-transform:uppercase;letter-spacing:3px;
    margin-bottom:.6rem;
}

/* ═══════════════════════════════════════════
   RESULT CARDS
   ═══════════════════════════════════════════ */
.result-card{
    padding:2.5rem 2rem;border-radius:20px;
    text-align:center;margin:1rem 0;
    animation:fadeInUp .5s ease both;
    backdrop-filter:blur(10px);
}
.potable{
    background:rgba(255,255,255,.03);
    border:1px solid rgba(255,255,255,.12);color:#fff;
}
.not-potable{
    background:rgba(255,255,255,.03);
    border:1px solid rgba(255,255,255,.08);color:#fff;
}
.result-card h2{
    font-size:2rem;margin:0;font-weight:800;
    letter-spacing:-1px;color:#fff;
}
.result-card p{
    font-size:.95rem;margin-top:.6rem;color:#666;
    letter-spacing:.5px;
}

/* ═══════════════════════════════════════════
   METRIC CARDS
   ═══════════════════════════════════════════ */
.metric-card{
    background:rgba(255,255,255,.02);
    backdrop-filter:blur(10px);
    padding:1.6rem 1.2rem;border-radius:16px;
    text-align:center;
    border:1px solid #1a1a1a;
    transition:all .3s cubic-bezier(.22,1,.36,1);
}
.metric-card:hover{
    transform:translateY(-4px);border-color:#333;
    box-shadow:0 8px 30px rgba(255,255,255,.03);
}
.metric-card h3{
    font-size:2.2rem;color:#fff;margin:0;
    font-weight:800;letter-spacing:-1px;
}
.metric-card p{
    color:#555;font-size:.7rem;margin:.4rem 0 0;
    font-weight:600;text-transform:uppercase;
    letter-spacing:2px;
}

/* ═══════════════════════════════════════════
   FEATURE INFO
   ═══════════════════════════════════════════ */
.feature-info{
    background:rgba(255,255,255,.02);
    backdrop-filter:blur(10px);
    padding:1.2rem 1.5rem;border-radius:14px;
    border:1px solid #1a1a1a;
    border-left:3px solid #fff;
    margin:.7rem 0;font-size:.88rem;color:#aaa;
    transition:border-color .2s;
}
.feature-info:hover{border-color:#333}

/* ═══════════════════════════════════════════
   CONFIDENCE BAR
   ═══════════════════════════════════════════ */
.confidence-bar{
    height:6px;border-radius:3px;overflow:hidden;
    background:#111;margin:.8rem 0;
}
.confidence-fill{
    height:100%;border-radius:3px;
    background:#fff!important;
    transition:width 1s cubic-bezier(.22,1,.36,1);
}

/* ═══════════════════════════════════════════
   SIDEBAR (FALLBACK NAV)
   ═══════════════════════════════════════════ */
div[data-testid="stSidebar"]{
    background:#000!important;
    border-right:1px solid #111;
}
div[data-testid="stSidebar"] [data-testid="stMarkdown"]{color:#888}

/* ═══════════════════════════════════════════
   BUTTONS — FLAT, PREMIUM
   ═══════════════════════════════════════════ */
.stButton>button{
    background:#fff!important;color:#000!important;
    border:none!important;border-radius:8px!important;
    font-weight:700!important;font-size:.82rem!important;
    padding:.7rem 2rem!important;
    letter-spacing:1px!important;text-transform:uppercase!important;
    box-shadow:none!important;
    transition:all .25s cubic-bezier(.22,1,.36,1)!important;
}
.stButton>button:hover{
    background:#e0e0e0!important;
    transform:translateY(-2px)!important;
    box-shadow:0 8px 25px rgba(255,255,255,.06)!important;
}
.stButton>button:active{transform:translateY(0)!important}

.stDownloadButton>button{
    background:transparent!important;color:#fff!important;
    border:1px solid #222!important;border-radius:8px!important;
    font-weight:600!important;letter-spacing:1px!important;
    text-transform:uppercase!important;font-size:.78rem!important;
}
.stDownloadButton>button:hover{
    border-color:#666!important;background:#0a0a0a!important;
}

/* ═══════════════════════════════════════════
   SLIDERS
   ═══════════════════════════════════════════ */
.stSlider>div>div{color:#fff}
[data-testid="stSlider"] label{color:#888!important;font-size:.85rem!important}

/* ═══════════════════════════════════════════
   TABS
   ═══════════════════════════════════════════ */
.stTabs [data-baseweb="tab-list"]{
    gap:0;border-bottom:1px solid #1a1a1a;
}
.stTabs [data-baseweb="tab"]{
    background:transparent;color:#555;border:none;
    border-bottom:2px solid transparent;
    padding:.9rem 1.6rem;font-weight:600;
    font-size:.75rem;text-transform:uppercase;
    letter-spacing:1.5px;transition:all .2s;
}
.stTabs [data-baseweb="tab"]:hover{color:#aaa}
.stTabs [aria-selected="true"]{
    color:#fff!important;border-bottom-color:#fff!important;
    background:transparent!important;
}

/* ═══════════════════════════════════════════
   DATAFRAMES, CHAT, EXPANDER, METRICS
   ═══════════════════════════════════════════ */
[data-testid="stDataFrame"]{
    border:1px solid #1a1a1a;border-radius:14px;overflow:hidden;
}
[data-testid="stChatMessage"]{
    background:rgba(255,255,255,.02)!important;
    border:1px solid #1a1a1a;border-radius:16px;
    padding:1.2rem!important;margin-bottom:.8rem;
}
[data-testid="stExpander"]{
    border:1px solid #1a1a1a!important;border-radius:14px!important;
    background:rgba(255,255,255,.02);
}
[data-testid="stMetric"]{
    background:rgba(255,255,255,.02);
    border:1px solid #1a1a1a;border-radius:14px;padding:1.2rem;
}
[data-testid="stMetric"] label{color:#555!important;
    text-transform:uppercase;letter-spacing:1.5px;font-size:.7rem!important}
[data-testid="stMetric"] [data-testid="stMetricValue"]{
    color:#fff!important;font-weight:800!important}

/* ═══════════════════════════════════════════
   RADIO NAV IN SIDEBAR
   ═══════════════════════════════════════════ */
[data-testid="stSidebar"] .stRadio label{
    color:#666!important;transition:color .2s;
    font-size:.85rem!important;letter-spacing:.5px;
}
[data-testid="stSidebar"] .stRadio label:hover{color:#fff!important}

/* ═══════════════════════════════════════════
   TYPOGRAPHY
   ═══════════════════════════════════════════ */
hr{border-color:#111!important}
.stMarkdown h1,.stMarkdown h2,.stMarkdown h3,.stMarkdown h4{
    color:#fff!important;letter-spacing:-.5px;font-weight:700;
}
.stMarkdown h3{
    font-size:1.1rem!important;text-transform:uppercase;
    letter-spacing:1.5px!important;font-weight:600!important;
    color:#888!important;margin-top:2rem!important;
}
.stMarkdown p,.stMarkdown li{color:#999;line-height:1.7}
.stMarkdown strong{color:#fff}
.stMarkdown code{
    background:#111!important;color:#aaa!important;
    border:1px solid #222;border-radius:4px;padding:.15rem .5rem;
}
[data-testid="stAlert"]{border-radius:14px!important;border:1px solid #1a1a1a!important}
.stSpinner>div{color:#555!important}

/* ═══════════════════════════════════════════
   RESPONSIVE
   ═══════════════════════════════════════════ */
@media(max-width:768px){
    .main .block-container{padding:4.5rem 1rem 3rem}
    .main-header h1{font-size:2rem}
    .metric-card h3{font-size:1.5rem}
    .result-card h2{font-size:1.4rem}
    .glass-card{padding:1.4rem 1.2rem;border-radius:14px}
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL & DATA
# ============================================================
@st.cache_resource
def load_model():
    """Load the trained Random Forest model"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "models", "model.pkl")
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f)
    return None

@st.cache_resource
def load_imputer():
    """Load the trained imputer rules for missing inference data"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    imputer_path = os.path.join(base_dir, "models", "imputer.pkl")
    if os.path.exists(imputer_path):
        with open(imputer_path, "rb") as f:
            return pickle.load(f)
    return {}


@st.cache_data
def load_dataset():
    """Load the original dataset for statistics and visualization"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "src", "data", "water_potability.csv")
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    return None


@st.cache_data
def load_metrics():
    """Load model evaluation metrics"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    metrics_path = os.path.join(base_dir, "reports", "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            return json.load(f)
    return None


@st.cache_data
def load_comparison():
    """Load model comparison results"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    comparison_path = os.path.join(base_dir, "reports", "model_comparison.json")
    if os.path.exists(comparison_path):
        with open(comparison_path, "r") as f:
            return json.load(f)
    return None


# ============================================================
# PREDICTION CACHE (survives query-param page navigation)
# ============================================================
_PREDICTION_CACHE_PATH = os.path.join(
    tempfile.gettempdir(), "aquapure_prediction_cache.json"
)


def _save_prediction_cache(params, prediction, confidence):
    """Persist prediction data to a temp JSON file so it survives page reruns."""
    try:
        payload = {
            "params": params,
            "prediction": prediction,
            "confidence": confidence,
        }
        with open(_PREDICTION_CACHE_PATH, "w") as f:
            json.dump(payload, f)
    except Exception:
        pass


def _load_prediction_cache():
    """Load cached prediction data from disk. Returns (params, prediction, confidence) or Nones."""
    try:
        if os.path.exists(_PREDICTION_CACHE_PATH):
            with open(_PREDICTION_CACHE_PATH, "r") as f:
                data = json.load(f)
            return data.get("params"), data.get("prediction"), data.get("confidence")
    except Exception:
        pass
    return None, None, None


def _clear_prediction_cache():
    """Remove cached prediction file."""
    try:
        if os.path.exists(_PREDICTION_CACHE_PATH):
            os.remove(_PREDICTION_CACHE_PATH)
    except Exception:
        pass


# ============================================================
# GEMINI HELPER
# ============================================================
def get_gemini_model():
    """Initialise and return Gemini flash model, or None if key missing."""
    if not GEMINI_AVAILABLE:
        return None
    try:
        # Step 1: Natively attempt to capture Render Cloud Environment variables
        api_key = os.environ.get("GEMINI_API_KEY", "")
        
        # Step 2: Fallback explicitly to local Streamlit dicts correctly bypassing KeyError limits
        if not api_key:
            try:
                api_key = st.secrets.get("GEMINI_API_KEY", "")
            except Exception:
                pass
                
        if not api_key:
            return None
            
        genai.configure(api_key=api_key)
        return genai.GenerativeModel("gemini-2.5-flash")
    except Exception:
        return None


def param_status(key, value):
    """Return (emoji, status_str) for a parameter value vs WHO limits."""
    lim = WHO_LIMITS[key]
    if lim["min"] <= value <= lim["max"]:
        return "🟢", "Safe"
    overshoot = abs(value - lim["max"]) / lim["max"] if value > lim["max"] else abs(value - lim["min"]) / max(lim["min"], 0.001)
    if overshoot < 0.25:
        return "🟡", "Borderline"
    return "🔴", "Unsafe"


# ============================================================
# SESSION STATE INITIALISATION
# ============================================================
if "last_params" not in st.session_state:
    st.session_state["last_params"] = None
if "last_prediction" not in st.session_state:
    st.session_state["last_prediction"] = None
if "last_confidence" not in st.session_state:
    st.session_state["last_confidence"] = None
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# --- Restore prediction from disk cache if session_state is empty ---
if st.session_state["last_params"] is None:
    _cached_params, _cached_pred, _cached_conf = _load_prediction_cache()
    if _cached_params is not None:
        st.session_state["last_params"] = _cached_params
        st.session_state["last_prediction"] = _cached_pred
        st.session_state["last_confidence"] = _cached_conf


# ============================================================
# NAVIGATION SYSTEM (query-param driven)
# ============================================================
_PAGE_LIST = [
    "🔮 Predict Water Quality",
    "📋 Health Report",
    "🤖 Ask the Water Bot",
    "💊 How to Make it Potable",
    "📊 Data Exploration",
    "🏆 Model Comparison",
    "ℹ️ About Project",
]
_NAV_MAP = {
    "Predict":   0,
    "Report":    1,
    "Chatbot":   2,
    "Treatment": 3,
    "Data":      4,
    "Models":    5,
    "About":     6,
}

# Read query param to determine which page to show
_qp = st.query_params.get("nav", "Predict")
_default_idx = _NAV_MAP.get(_qp, 0)

page = st.sidebar.radio(
    "Nav",
    _PAGE_LIST,
    index=_default_idx,
    label_visibility="collapsed"
)

# Keep query param in sync with sidebar selection
_reverse_map = {v: k for k, v in _NAV_MAP.items()}
_current_idx = _PAGE_LIST.index(page)
_current_key = _reverse_map.get(_current_idx, "Predict")
if st.query_params.get("nav") != _current_key:
    st.query_params["nav"] = _current_key

# ============================================================
# FLOATING NAVBAR
# ============================================================
_nav_links_html = ""
for short_label, idx in _NAV_MAP.items():
    _active = ' class="active"' if _current_idx == idx else ""
    _nav_links_html += f'<a href="?nav={short_label}" target="_self"{_active}>{short_label}</a>'

st.markdown(f"""
<div class="pro-navbar">
    <div class="logo">💧 AQUA<span>PURE</span></div>
    <div class="nav-links">{_nav_links_html}</div>
    <a class="nav-cta" href="?nav=Predict" target="_self">Analyze Water</a>
</div>
""", unsafe_allow_html=True)


# ============================================================
# PAGE 1: PREDICT WATER QUALITY
# ============================================================
if page == "🔮 Predict Water Quality":

    st.markdown("""
    <div class="main-header">
        <p class="section-label">Water Analysis</p>
        <h1>Potability Prediction</h1>
        <p>Enter water quality parameters below</p>
    </div>
    """, unsafe_allow_html=True)

    model = load_model()
    imputer = load_imputer()
    data = load_dataset()

    if model is None:
        st.error("⚠️ Model not found! Please train the model first by running `python src/model/model_building.py`")
        st.stop()

    if data is not None:
        stats = data.describe()

    # Input form with two columns
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📝 Enter Water Parameters")

    col1, col2, col3 = st.columns(3)

    with col1:
        ph = st.slider("🧪 pH Level", 0.0, 14.0, 7.0, 0.1,
                        help="WHO recommends 6.5-8.5 for drinking water")
        hardness = st.slider("💎 Hardness (mg/L)", 47.0, 324.0, 196.0, 1.0,
                             help="Calcium & Magnesium content")
        solids = st.slider("🔬 Total Solids (ppm)", 320.0, 61228.0, 20927.0, 1.0,
                           help="Total Dissolved Solids. WHO limit: 500 ppm")

    with col2:
        chloramines = st.slider("🧴 Chloramines (ppm)", 0.35, 13.13, 7.12, 0.01,
                                help="Disinfectant. Safe up to 4 ppm (EPA)")
        sulfate = st.slider("⚗️ Sulfate (mg/L)", 129.0, 481.0, 333.0, 1.0,
                            help="WHO suggests limit of 250 mg/L")
        conductivity = st.slider("⚡ Conductivity (μS/cm)", 181.0, 753.0, 421.0, 1.0,
                                 help="Electrical conductivity — measures ion concentration")

    with col3:
        organic_carbon = st.slider("🌿 Organic Carbon (ppm)", 2.2, 28.3, 14.28, 0.01,
                                   help="Total organic carbon — organic pollutants")
        trihalomethanes = st.slider("☢️ Trihalomethanes (μg/L)", 0.74, 124.0, 66.40, 0.01,
                                    help="Chlorine byproduct. Carcinogenic above 80 μg/L")
        turbidity = st.slider("🌊 Turbidity (NTU)", 1.45, 6.74, 3.97, 0.01,
                              help="Cloudiness. WHO limit: 5 NTU")

    # Predict button
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🔮 Predict Water Quality", use_container_width=True, type="primary"):
        # Create input dataframe
        input_data = pd.DataFrame({
            'ph': [ph],
            'Hardness': [hardness],
            'Solids': [solids],
            'Chloramines': [chloramines],
            'Sulfate': [sulfate],
            'Conductivity': [conductivity],
            'Organic_carbon': [organic_carbon],
            'Trihalomethanes': [trihalomethanes],
            'Turbidity': [turbidity]
        })

        # Rigorously apply MLOps Imputation constraints mimicking pipeline
        if imputer:
            for col, mean_val in imputer.items():
                if col in input_data.columns:
                    input_data[col] = input_data[col].fillna(mean_val)

        # Make prediction
        prediction = model.predict(input_data)[0]

        # Get prediction probabilities if available
        try:
            proba = model.predict_proba(input_data)[0]
            confidence = max(proba) * 100
        except Exception:
            confidence = None

        # Display result
        col_result1, col_result2 = st.columns([2, 1])

        with col_result1:
            if prediction == 1:
                st.markdown("""
                <div class="result-card potable">
                    <h2>✅ Water is POTABLE</h2>
                    <p>This water is predicted to be safe for drinking</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="result-card not-potable">
                    <h2>❌ Water is NOT POTABLE</h2>
                    <p>This water is predicted to be unsafe for drinking</p>
                </div>
                """, unsafe_allow_html=True)

        with col_result2:
            if confidence is not None:
                st.markdown(f"""
                <div class="metric-card" style="margin-top: 1rem;">
                    <h3>{confidence:.1f}%</h3>
                    <p>Confidence</p>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: {confidence}%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Show input summary
        st.markdown("### 📋 Your Input Summary")
        st.dataframe(input_data.T.rename(columns={0: 'Value'}), use_container_width=True)

        # --- persist for AI pages ---
        _pred_params = {
            'ph': ph, 'Hardness': hardness, 'Solids': solids,
            'Chloramines': chloramines, 'Sulfate': sulfate,
            'Conductivity': conductivity, 'Organic_carbon': organic_carbon,
            'Trihalomethanes': trihalomethanes, 'Turbidity': turbidity
        }
        st.session_state["last_params"] = _pred_params
        st.session_state["last_prediction"] = int(prediction)
        st.session_state["last_confidence"] = confidence

        # --- also persist to disk so it survives page navigation ---
        _save_prediction_cache(_pred_params, int(prediction), confidence)

        st.info("✨ Prediction saved! Visit **📋 Health Report**, **🤖 Ask the Water Bot**, or **💊 How to Make it Potable** from the sidebar.")


# ============================================================
# PAGE 2: DATA EXPLORATION
# ============================================================
elif page == "📊 Data Exploration":

    st.markdown("""
    <div class="main-header">
        <p class="section-label">Dataset</p>
        <h1>Data Exploration</h1>
        <p>Discover patterns in the potability dataset</p>
    </div>
    """, unsafe_allow_html=True)

    data = load_dataset()

    if data is None:
        st.error("⚠️ Dataset not found!")
        st.stop()

    # Dataset overview
    st.markdown("### 📋 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(f'<div class="metric-card"><h3>{data.shape[0]}</h3><p>Total Samples</p></div>', unsafe_allow_html=True)
    col2.markdown(f'<div class="metric-card"><h3>{data.shape[1] - 1}</h3><p>Features</p></div>', unsafe_allow_html=True)
    col3.markdown(f'<div class="metric-card"><h3>{(data.Potability == 1).sum()}</h3><p>Potable Samples</p></div>', unsafe_allow_html=True)
    col4.markdown(f'<div class="metric-card"><h3>{(data.Potability == 0).sum()}</h3><p>Not Potable</p></div>', unsafe_allow_html=True)

    st.markdown("---")

    # Tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Correlation Heatmap",
        "📈 Feature Distributions",
        "📦 Box Plots",
        "❓ Missing Values",
        "🎯 Target Distribution"
    ])

    with tab1:
        st.markdown("#### Correlation Heatmap")
        st.markdown("Shows relationships between all features. Values close to **+1 or -1** indicate strong correlation.")
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor('#000000')
        ax.set_facecolor('#000000')
        corr = data.corr()
        sns.heatmap(corr, annot=True, fmt='.3f', cmap='Greys', center=0,
                    square=True, linewidths=0.5, ax=ax, vmin=-1, vmax=1,
                    annot_kws={'color': '#ccc', 'fontsize': 8})
        ax.set_title('Correlation Heatmap', fontsize=14, fontweight='bold', color='#fff')
        ax.tick_params(colors='#aaa')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab2:
        st.markdown("#### Feature Distributions by Potability")
        st.markdown("Histograms showing how each feature distributes across potable vs non-potable water.")
        features = data.columns.drop('Potability')
        fig, axes = plt.subplots(3, 3, figsize=(14, 12))
        fig.patch.set_facecolor('#000000')
        colors = ['#555555', '#ffffff']
        labels = ['Not Potable', 'Potable']
        for idx, feature in enumerate(features):
            ax = axes[idx // 3, idx % 3]
            ax.set_facecolor('#0A0A0A')
            for cls in [0, 1]:
                subset = data[data['Potability'] == cls][feature].dropna()
                ax.hist(subset, bins=30, alpha=0.7, color=colors[cls], label=labels[cls], edgecolor='#222')
            ax.set_title(feature, fontsize=11, fontweight='bold', color='#fff')
            ax.legend(fontsize=7, facecolor='#111', edgecolor='#333', labelcolor='#ccc')
            ax.grid(axis='y', alpha=0.15, color='#333')
            ax.tick_params(colors='#aaa')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab3:
        st.markdown("#### Box Plots by Potability Class")
        st.markdown("Box plots help detect **outliers** and compare value ranges between classes.")
        features = data.columns.drop('Potability')
        plot_data = data.copy()
        plot_data['Potability'] = plot_data['Potability'].astype(str)
        fig, axes = plt.subplots(3, 3, figsize=(14, 12))
        fig.patch.set_facecolor('#000000')
        palette = {'0': '#555555', '1': '#ffffff'}
        for idx, feature in enumerate(features):
            ax = axes[idx // 3, idx % 3]
            ax.set_facecolor('#0A0A0A')
            sns.boxplot(x='Potability', y=feature, data=plot_data.dropna(subset=[feature]),
                        palette=palette, ax=ax)
            ax.set_title(feature, fontsize=11, fontweight='bold', color='#fff')
            ax.grid(axis='y', alpha=0.15, color='#333')
            ax.tick_params(colors='#aaa')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab4:
        st.markdown("#### Missing Values Analysis")
        nulls = data.isnull().sum()
        null_pct = (nulls / len(data) * 100).round(1)
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor('#000000')
        ax.set_facecolor('#0A0A0A')
        colors = ['#ffffff' if n > 0 else '#333333' for n in nulls]
        bars = ax.bar(nulls.index, nulls.values, color=colors, edgecolor='#222')
        for bar, count, pct in zip(bars, nulls.values, null_pct.values):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
                        f'{count}\n({pct}%)', ha='center', fontsize=9, fontweight='bold', color='#ccc')
        ax.set_title('Missing Values per Feature', fontsize=14, fontweight='bold', color='#fff')
        plt.xticks(rotation=45, ha='right', color='#aaa')
        ax.tick_params(colors='#aaa')
        ax.grid(axis='y', alpha=0.15, color='#333')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("**Missing columns:** ph, Sulfate, Trihalomethanes — handled with **Mean Imputation**")

    with tab5:
        st.markdown("#### Target Variable Distribution")
        col1, col2 = st.columns(2)
        counts = data['Potability'].value_counts()
        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor('#000000')
            ax.set_facecolor('#0A0A0A')
            bars = ax.bar(['Not Potable (0)', 'Potable (1)'], counts.values,
                          color=['#555555', '#ffffff'], edgecolor='#222')
            for bar, count in zip(bars, counts.values):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 10,
                        str(count), ha='center', fontsize=12, fontweight='bold', color='#ccc')
            ax.set_title('Count', fontsize=13, fontweight='bold', color='#fff')
            ax.grid(axis='y', alpha=0.15, color='#333')
            ax.tick_params(colors='#aaa')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor('#000000')
            ax.pie(counts.values, labels=['Not Potable', 'Potable'],
                   colors=['#555555', '#ffffff'], autopct='%1.1f%%', startangle=90,
                   textprops={'color': '#ccc'})
            ax.set_title('Percentage', fontsize=13, fontweight='bold', color='#fff')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    # Raw data view
    st.markdown("---")
    with st.expander("📋 View Raw Dataset"):
        st.dataframe(data, use_container_width=True)
        st.markdown(f"**Shape:** {data.shape[0]} rows × {data.shape[1]} columns")


# ============================================================
# PAGE 3: MODEL COMPARISON
# ============================================================
elif page == "🏆 Model Comparison":

    st.markdown("""
    <div class="main-header">
        <p class="section-label">Performance</p>
        <h1>Model Comparison</h1>
        <p>Random Forest vs SVM vs Logistic Regression</p>
    </div>
    """, unsafe_allow_html=True)

    comparison = load_comparison()
    metrics = load_metrics()

    if comparison is None:
        st.error("⚠️ Model comparison results not found! Run `python src/model/model_comparison.py` first.")
        st.stop()

    # Comparison table
    st.markdown("### 📊 Performance Metrics")

    comp_df = pd.DataFrame(comparison).T
    comp_df.index.name = 'Model'

    # Highlight the best in each column
    st.dataframe(
        comp_df.style.highlight_max(axis=0, color='#333333')
                     .format('{:.4f}'),
        use_container_width=True
    )

    # Visual comparison
    st.markdown("### 📈 Visual Comparison")

    models = list(comparison.keys())
    metric_names = ['accuracy', 'precision', 'recall', 'f1_score']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#000000')
    ax.set_facecolor('#0A0A0A')
    x = np.arange(len(metric_names))
    width = 0.25
    colors = ['#ffffff', '#999999', '#444444']

    for i, model_name in enumerate(models):
        values = [comparison[model_name][m] for m in metric_names]
        bars = ax.bar(x + i * width, values, width, label=model_name, color=colors[i], edgecolor='#222')
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#ccc')

    ax.set_xlabel('Metrics', fontsize=12, color='#aaa')
    ax.set_ylabel('Score', fontsize=12, color='#aaa')
    ax.set_title('Model Comparison', fontsize=14, fontweight='bold', color='#fff')
    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_labels, fontsize=11, color='#aaa')
    ax.legend(fontsize=11, facecolor='#111', edgecolor='#333', labelcolor='#ccc')
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.15, color='#333')
    ax.tick_params(colors='#aaa')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Why Random Forest
    st.markdown("---")
    st.markdown("### 🏅 Why Random Forest is the Best Choice")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **✅ Advantages of Random Forest:**
        - **Best F1-Score** — balances precision and recall
        - **Highest Recall** — catches more potable water correctly
        - **No scaling needed** — simpler pipeline
        - **Handles non-linear data** — captures complex feature interactions
        - **Robust to outliers** — common in water quality data
        """)

    with col2:
        st.markdown("""
        **❌ Why NOT the others:**
        - **SVM**: Slightly higher accuracy but lower recall & F1. Misses more potable water.
        - **Logistic Regression**: Completely fails — predicts everything as "Not Potable" (0 recall).
          This proves the data has non-linear relationships that linear models can't handle.
        """)


# ============================================================
# PAGE 4: ABOUT PROJECT
# ============================================================
elif page == "ℹ️ About Project":

    st.markdown("""
    <div class="main-header">
        <p class="section-label">Overview</p>
        <h1>About This Project</h1>
        <p>Machine Learning with MLOps Pipeline</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### 🎯 Project Overview
    This project predicts whether water is **safe to drink (potable)** based on 9 water quality
    parameters using Machine Learning. The model takes water quality readings as input and outputs
    whether the water is potable (1) or not potable (0).
    """)

    st.markdown("### 📊 Features Used")
    features_info = {
        "ph": ("Acidity/basicity (0-14)", "6.5 – 8.5"),
        "Hardness": ("Calcium + Magnesium (mg/L)", "Up to 300 mg/L"),
        "Solids": ("Total Dissolved Solids (ppm)", "< 500 ppm"),
        "Chloramines": ("Disinfectant level (ppm)", "< 4 ppm"),
        "Sulfate": ("Sulfate ions (mg/L)", "< 250 mg/L"),
        "Conductivity": ("Electrical conductivity (μS/cm)", "< 400 μS/cm"),
        "Organic_carbon": ("Organic pollutants (ppm)", "< 2 ppm"),
        "Trihalomethanes": ("Chlorine byproduct (μg/L)", "< 80 μg/L"),
        "Turbidity": ("Cloudiness (NTU)", "< 5 NTU")
    }

    feat_df = pd.DataFrame({
        'Feature': features_info.keys(),
        'Description': [v[0] for v in features_info.values()],
        'WHO/EPA Safe Limit': [v[1] for v in features_info.values()]
    })
    st.dataframe(feat_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 🔄 End-to-End Pipeline Flow")
    st.markdown("""
    ```
    1. Data Collection      → Load CSV, split 80/20 train-test
    2. Data Preprocessing   → Mean imputation for missing values
    3. Visualization        → Generate correlation heatmaps, distributions
    4. Model Building       → Train Random Forest (n_estimators=100)
    5. Model Evaluation     → Calculate Accuracy, Precision, Recall, F1
    6. Model Comparison     → Compare RF vs SVM vs Logistic Regression
    7. Streamlit UI         → Interactive prediction web app (this page!)
    ```
    """)

    st.markdown("---")
    st.markdown("### 🛠️ MLOps with DVC")
    st.markdown("""
    | Component | Purpose |
    |-----------|---------|
    | `dvc.yaml` | Pipeline definition — 6 reproducible stages |
    | `params.yaml` | Centralized hyperparameters (test_size, n_estimators) |
    | `dvc.lock` | MD5 hashes for reproducibility |
    | `reports/metrics.json` | DVC-tracked model metrics |
    | `dvc repro` | Re-run only changed stages automatically |
    """)

    st.markdown("---")
    st.markdown("### 📝 Key Technical Decisions")
    st.markdown("""
    - **No Encoding Needed** — All features are already numerical (float64)
    - **No Label Encoding** — Target is already binary (0/1)
    - **Mean Imputation** — Used for missing values in ph, Sulfate, Trihalomethanes
    - **No Feature Engineering** — All 9 features are WHO/EPA water quality standards
    - **StandardScaler** — Applied only for SVM and LR during model comparison (not for RF)
    """)

    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 1.5rem; color: #555;'>
        <p>Built with ❤️ using Python, Scikit-Learn, DVC & Streamlit</p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# PAGE 5: HEALTH REPORT
# ============================================================
elif page == "\U0001f4cb Health Report":
    st.markdown("""
    <div class="main-header">
        <p class="section-label">AI Powered</p>
        <h1>Water Health Report</h1>
        <p>Comprehensive safety assessment of your sample</p>
    </div>
    """, unsafe_allow_html=True)

    params = st.session_state.get("last_params")
    prediction = st.session_state.get("last_prediction")
    confidence = st.session_state.get("last_confidence")

    if params is None:
        st.warning("\u26a0\ufe0f No prediction found. Please go to **\U0001f52e Predict Water Quality** first and click Predict.")
        st.stop()

    st.markdown("### \U0001f52c Parameter Safety Analysis")
    rows = []
    safe_count = 0
    for key, val in params.items():
        lim = WHO_LIMITS[key]
        emoji, status = param_status(key, val)
        safe_limit = f"{lim['min']}\u2013{lim['max']} {lim['unit']}".strip()
        if status == "Safe":
            safe_count += 1
        rows.append({
            "Parameter": lim["label"],
            "Your Value": f"{val:.2f} {lim['unit']}".strip(),
            "WHO / EPA Limit": safe_limit,
            "Status": f"{emoji} {status}"
        })
    report_df = pd.DataFrame(rows)
    st.dataframe(report_df, use_container_width=True, hide_index=True)

    safety_score = int((safe_count / len(params)) * 100)
    score_color = "#ffffff" if safety_score >= 78 else ("#999999" if safety_score >= 45 else "#555555")
    st.markdown(f"""
    <div class="metric-card" style="max-width:280px; margin: 1rem auto;">
        <h3 style="color:{score_color};">{safety_score}/100</h3>
        <p>Overall Safety Score</p>
    </div>
    """, unsafe_allow_html=True)

    verdict = "\u2705 POTABLE" if prediction == 1 else "\u274c NOT POTABLE"
    conf_text = f"{confidence:.1f}% confidence" if confidence else ""
    st.markdown(f"**Model Verdict:** {verdict} &nbsp;|&nbsp; {conf_text}")
    st.markdown("---")

    st.markdown("### \U0001f916 AI Health Narrative")
    model_ai = get_gemini_model()
    if model_ai is None:
        st.error("\U0001f511 Gemini API key not configured. Add `GEMINI_API_KEY` to `.streamlit/secrets.toml`.")
    else:
        unsafe_params = [
            WHO_LIMITS[k]["label"] + f" ({v:.2f} {WHO_LIMITS[k]['unit']})"
            for k, v in params.items() if param_status(k, v)[1] != "Safe"
        ]
        param_lines = "\n".join(
            f"- {WHO_LIMITS[k]['label']}: {v:.2f} {WHO_LIMITS[k]['unit']}"
            for k, v in params.items()
        )
        prompt = (
            f"You are an expert water quality scientist. A water sample has been analysed.\n\n"
            f"Prediction: {'POTABLE (safe for drinking)' if prediction == 1 else 'NOT POTABLE (unsafe for drinking)'}\n"
            f"Confidence: {conf_text}\nSafety Score: {safety_score}/100\n"
            f"Out-of-range parameters: {', '.join(unsafe_params) if unsafe_params else 'None'}\n\n"
            f"Water readings:\n{param_lines}\n\n"
            f"Write a professional Water Quality Health Report with these sections:\n"
            f"1. Executive Summary (2 sentences)\n"
            f"2. Key Risk Factors (bullet list, be specific)\n"
            f"3. Health Implications (for adults, children, elderly)\n"
            f"4. Short-term Recommendations\n\n"
            f"Use plain language. Format with markdown headers."
        )
        with st.spinner("\U0001f916 Generating AI health narrative..."):
            try:
                response = model_ai.generate_content(prompt)
                narrative = response.text
                st.markdown(narrative)

                report_text = (
                    f"WATER QUALITY HEALTH REPORT\n"
                    f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
                    f"{'='*60}\n\nMODEL VERDICT: {verdict}\nSAFETY SCORE: {safety_score}/100\n\n"
                    f"PARAMETER READINGS\n{'='*60}\n"
                )
                for key, val in params.items():
                    lim = WHO_LIMITS[key]
                    _, status = param_status(key, val)
                    report_text += f"{lim['label']:22s}: {val:8.2f} {lim['unit']:8s}  [{status}]\n"
                report_text += f"\nAI NARRATIVE\n{'='*60}\n{narrative}"
                st.download_button(
                    "\u2b07\ufe0f Download Full Report (.txt)",
                    data=report_text,
                    file_name="water_health_report.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"AI generation failed: {e}")


# ============================================================
# PAGE 6: ASK THE WATER BOT
# ============================================================
elif page == "\U0001f916 Ask the Water Bot":
    st.markdown("""
    <div class="main-header">
        <p class="section-label">AI Assistant</p>
        <h1>Ask the Water Bot</h1>
        <p>Chat with an expert about your test results</p>
    </div>
    """, unsafe_allow_html=True)

    params = st.session_state.get("last_params")
    prediction = st.session_state.get("last_prediction")
    confidence = st.session_state.get("last_confidence")

    if params is None:
        st.warning("\u26a0\ufe0f No prediction found. Please run a prediction first on the **\U0001f52e Predict Water Quality** page.")
        st.stop()

    model_ai = get_gemini_model()
    if model_ai is None:
        st.error("\U0001f511 Gemini API key not configured. Add `GEMINI_API_KEY` to `.streamlit/secrets.toml`.")
        st.stop()

    conf_text = f"{confidence:.1f}%" if confidence else "N/A"
    verdict = "POTABLE" if prediction == 1 else "NOT POTABLE"
    param_lines = "\n".join(
        f"- {WHO_LIMITS[k]['label']}: {v:.2f} {WHO_LIMITS[k]['unit']} (limit: {WHO_LIMITS[k]['min']}\u2013{WHO_LIMITS[k]['max']})"
        for k, v in params.items()
    )
    system_context = (
        f"You are WaterBot, an expert water quality analyst AI assistant.\n"
        f"The user tested their water. Results:\n"
        f"Prediction: {verdict} | Confidence: {conf_text}\n"
        f"{param_lines}\n"
        f"Answer questions concisely, referencing these specific readings. Keep responses under 200 words unless detail is requested."
    )

    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask anything about your water results\u2026")
    if user_input:
        st.session_state["chat_history"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    history_text = "\n".join(
                        f"{m['role'].capitalize()}: {m['content']}"
                        for m in st.session_state["chat_history"][:-1]
                    )
                    full_prompt = f"{system_context}\n\nConversation:\n{history_text}\nUser: {user_input}\nWaterBot:"
                    response = model_ai.generate_content(full_prompt)
                    reply = response.text
                except Exception as e:
                    reply = f"\u26a0\ufe0f Error: {e}"
            st.markdown(reply)
            st.session_state["chat_history"].append({"role": "assistant", "content": reply})

    if st.button("\U0001f5d1\ufe0f Clear Chat History", use_container_width=True):
        st.session_state["chat_history"] = []
        st.rerun()


# ============================================================
# PAGE 7: HOW TO MAKE IT POTABLE
# ============================================================
elif page == "\U0001f48a How to Make it Potable":
    st.markdown("""
    <div class="main-header">
        <p class="section-label">Treatment</p>
        <h1>Water Treatment Guide</h1>
        <p>AI-generated plan to make your water safe</p>
    </div>
    """, unsafe_allow_html=True)

    params = st.session_state.get("last_params")
    prediction = st.session_state.get("last_prediction")

    if params is None:
        st.warning("\u26a0\ufe0f No prediction found. Please run a prediction first on the **\U0001f52e Predict Water Quality** page.")
        st.stop()

    unsafe = {k: v for k, v in params.items() if param_status(k, v)[1] == "Unsafe"}
    borderline = {k: v for k, v in params.items() if param_status(k, v)[1] == "Borderline"}
    safe_p = {k: v for k, v in params.items() if param_status(k, v)[1] == "Safe"}

    if prediction == 1 and not unsafe:
        st.success("\u2705 Your water is already potable with all parameters within WHO/EPA limits. No treatment needed!")
        st.markdown(
            "**Keep it safe by:**\n"
            "- Storing in clean, covered containers\n"
            "- Re-testing every 6\u201312 months\n"
            "- Checking for source contamination seasonally"
        )
        st.stop()

    col1, col2, col3 = st.columns(3)
    col1.metric("\U0001f534 Unsafe Parameters", len(unsafe))
    col2.metric("\U0001f7e1 Borderline Parameters", len(borderline))
    col3.metric("\U0001f7e2 Safe Parameters", len(safe_p))

    if unsafe:
        st.markdown("### \U0001f534 Parameters Requiring Treatment")
        for key, val in unsafe.items():
            lim = WHO_LIMITS[key]
            tip, cost = TREATMENT_TIPS[key]
            st.markdown(f"""
<div class="feature-info">
<b>{lim['label']}</b>: Your value <code>{val:.2f} {lim['unit']}</code> \u2014 limit is <code>{lim['max']} {lim['unit']}</code><br>
\U0001f4a1 <b>Treatment:</b> {tip}<br>
\U0001f4b0 <b>Cost category:</b> {cost}
</div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### \U0001f916 AI Treatment Plan")

    model_ai = get_gemini_model()
    if model_ai is None:
        st.error("\U0001f511 Gemini API key not configured. Add `GEMINI_API_KEY` to `.streamlit/secrets.toml`.")
    else:
        unsafe_desc = "\n".join(
            f"- {WHO_LIMITS[k]['label']}: {v:.2f} {WHO_LIMITS[k]['unit']} (limit: {WHO_LIMITS[k]['max']})"
            for k, v in unsafe.items()
        ) or "None"
        borderline_desc = "\n".join(
            f"- {WHO_LIMITS[k]['label']}: {v:.2f} {WHO_LIMITS[k]['unit']}"
            for k, v in borderline.items()
        ) or "None"
        prompt = (
            f"You are a water treatment engineer. A water sample needs treatment.\n\n"
            f"OUT-OF-RANGE (must fix):\n{unsafe_desc}\n\n"
            f"BORDERLINE (monitor):\n{borderline_desc}\n\n"
            f"Create a practical numbered treatment plan:\n"
            f"1. List steps in order of priority (most critical first)\n"
            f"2. For each step: what to do, equipment/chemical needed, approximate cost, and time required\n"
            f"3. Estimate total time until water is safe to drink\n"
            f"4. Add a brief 'Maintenance Tips' section\n\n"
            f"Be specific and practical. Format with markdown."
        )
        with st.spinner("\U0001f916 Generating treatment plan..."):
            try:
                response = model_ai.generate_content(prompt)
                st.markdown(response.text)
            except Exception as e:
                st.error(f"AI generation failed: {e}")
