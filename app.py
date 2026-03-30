import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import json
import datetime
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
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS FOR PREMIUM LOOK
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Fixed Material Icon Bug: explicitly excluded st- class brute-forcing */
    html, body, h1, h2, h3, p, span, li, div:not([class*="stIcon"]) {
        font-family: 'Inter', sans-serif;
    }

    /* Dark SaaS Overall Background Elements */
    .main-header {
        background: linear-gradient(135deg, #151521, #1e1e2d, #252538);
        border: 1px solid rgba(255,255,255,0.05);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
    }

    .main-header h1 {
        font-size: 2.4rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
        background: -webkit-linear-gradient(45deg, #fff, #00E5FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .main-header p {
        font-size: 1.1rem;
        opacity: 0.8;
        margin-top: 0.5rem;
    }

    .result-card {
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        animation: fadeIn 0.5s ease-in;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .potable {
        background: linear-gradient(135deg, rgba(67, 160, 71, 0.8), rgba(102, 187, 106, 0.8));
        backdrop-filter: blur(10px);
        border: 1px solid rgba(102, 187, 106, 0.3);
        color: white;
    }

    .not-potable {
        background: linear-gradient(135deg, rgba(229, 57, 53, 0.8), rgba(239, 83, 80, 0.8));
        backdrop-filter: blur(10px);
        border: 1px solid rgba(239, 83, 80, 0.3);
        color: white;
    }

    .result-card h2 {
        font-size: 2rem;
        margin: 0;
        font-weight: 700;
    }

    .result-card p {
        font-size: 1.2rem;
        margin-top: 0.5rem;
        opacity: 0.9;
    }

    .metric-card {
        background: rgba(30, 30, 46, 0.6);
        padding: 1.2rem;
        border-radius: 12px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 229, 255, 0.15);
        border: 1px solid rgba(0, 229, 255, 0.3);
    }

    .metric-card h3 {
        font-size: 1.8rem;
        color: #00E5FF;
        margin: 0;
        font-weight: 700;
        text-shadow: 0 0 10px rgba(0, 229, 255, 0.3);
    }

    .metric-card p {
        color: #a6accd;
        font-size: 0.85rem;
        margin: 0.3rem 0 0 0;
        font-weight: 500;
    }

    .feature-info {
        background: rgba(30, 30, 46, 0.6);
        padding: 1rem 1.2rem;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-left: 4px solid #00E5FF;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        color: #e0e0e0;
    }

    .stSlider > div > div { color: #00E5FF; }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #151521 0%, #1a1a2e 100%);
        border-right: 1px solid rgba(255,255,255,0.05);
    }

    .confidence-bar {
        height: 30px;
        border-radius: 15px;
        overflow: hidden;
        background: rgba(0,0,0,0.3);
        border: 1px solid rgba(255,255,255,0.05);
        margin: 0.5rem 0;
    }

    .confidence-fill {
        height: 100%;
        border-radius: 15px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        color: white;
        font-size: 0.85rem;
        transition: width 0.8s ease;
        box-shadow: inset 0 0 10px rgba(255,255,255,0.2);
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
# GEMINI HELPER
# ============================================================
def get_gemini_model():
    """Initialise and return Gemini flash model, or None if key missing."""
    if not GEMINI_AVAILABLE:
        return None
    try:
        api_key = st.secrets.get("GEMINI_API_KEY", "")
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


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
st.sidebar.markdown("## 💧 Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "🔮 Predict Water Quality",
        "📋 Health Report",
        "🤖 Ask the Water Bot",
        "💊 How to Make it Potable",
        "📊 Data Exploration",
        "🏆 Model Comparison",
        "ℹ️ About Project",
    ],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='text-align: center; opacity: 0.7; font-size: 0.8rem;'>
    <p>Water Potability Prediction<br>Powered by Machine Learning</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# PAGE 1: PREDICT WATER QUALITY
# ============================================================
if page == "🔮 Predict Water Quality":

    st.markdown("""
    <div class="main-header">
        <h1>💧 Water Potability Prediction</h1>
        <p>Enter water quality parameters to predict if the water is safe to drink</p>
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
    st.markdown("### 📝 Enter Water Parameters")

    col1, col2, col3 = st.columns(3)

    with col1:
        ph = st.slider("🧪 pH Level", 0.0, 14.0, 7.0, 0.1,
                        help="WHO recommends 6.5-8.5 for drinking water")
        hardness = st.slider("💎 Hardness (mg/L)", 47.0, 324.0, 196.0, 1.0,
                             help="Calcium & Magnesium content")
        solids = st.slider("🔬 Total Solids (ppm)", 320.0, 61228.0, 20927.0, 100.0,
                           help="Total Dissolved Solids. WHO limit: 500 ppm")

    with col2:
        chloramines = st.slider("🧴 Chloramines (ppm)", 0.35, 13.13, 7.12, 0.1,
                                help="Disinfectant. Safe up to 4 ppm (EPA)")
        sulfate = st.slider("⚗️ Sulfate (mg/L)", 129.0, 481.0, 333.0, 1.0,
                            help="WHO suggests limit of 250 mg/L")
        conductivity = st.slider("⚡ Conductivity (μS/cm)", 181.0, 753.0, 421.0, 1.0,
                                 help="Electrical conductivity — measures ion concentration")

    with col3:
        organic_carbon = st.slider("🌿 Organic Carbon (ppm)", 2.2, 28.3, 14.28, 0.1,
                                   help="Total organic carbon — organic pollutants")
        trihalomethanes = st.slider("☢️ Trihalomethanes (μg/L)", 0.74, 124.0, 66.4, 0.5,
                                    help="Chlorine byproduct. Carcinogenic above 80 μg/L")
        turbidity = st.slider("🌊 Turbidity (NTU)", 1.45, 6.74, 3.97, 0.1,
                              help="Cloudiness. WHO limit: 5 NTU")

    # Predict button
    st.markdown("---")

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
                color = "#43A047" if prediction == 1 else "#E53935"
                st.markdown(f"""
                <div class="metric-card" style="margin-top: 1rem;">
                    <h3>{confidence:.1f}%</h3>
                    <p>Prediction Confidence</p>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: {confidence}%; background: {color};">
                            {confidence:.1f}%
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Show input summary
        st.markdown("### 📋 Your Input Summary")
        st.dataframe(input_data.T.rename(columns={0: 'Value'}), use_container_width=True)

        # --- persist for AI pages ---
        st.session_state["last_params"] = {
            'ph': ph, 'Hardness': hardness, 'Solids': solids,
            'Chloramines': chloramines, 'Sulfate': sulfate,
            'Conductivity': conductivity, 'Organic_carbon': organic_carbon,
            'Trihalomethanes': trihalomethanes, 'Turbidity': turbidity
        }
        st.session_state["last_prediction"] = int(prediction)
        st.session_state["last_confidence"] = confidence
        st.info("✨ Prediction saved! Visit **📋 Health Report**, **🤖 Ask the Water Bot**, or **💊 How to Make it Potable** from the sidebar.")


# ============================================================
# PAGE 2: DATA EXPLORATION
# ============================================================
elif page == "📊 Data Exploration":

    st.markdown("""
    <div class="main-header">
        <h1>📊 Data Exploration</h1>
        <p>Explore the water potability dataset and discover patterns</p>
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
        corr = data.corr()
        sns.heatmap(corr, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                    square=True, linewidths=0.5, ax=ax, vmin=-1, vmax=1)
        ax.set_title('Correlation Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab2:
        st.markdown("#### Feature Distributions by Potability")
        st.markdown("Histograms showing how each feature distributes across potable vs non-potable water.")
        features = data.columns.drop('Potability')
        fig, axes = plt.subplots(3, 3, figsize=(14, 12))
        colors = ['#E53935', '#43A047']
        labels = ['Not Potable', 'Potable']
        for idx, feature in enumerate(features):
            ax = axes[idx // 3, idx % 3]
            for cls in [0, 1]:
                subset = data[data['Potability'] == cls][feature].dropna()
                ax.hist(subset, bins=30, alpha=0.6, color=colors[cls], label=labels[cls], edgecolor='white')
            ax.set_title(feature, fontsize=11, fontweight='bold')
            ax.legend(fontsize=7)
            ax.grid(axis='y', alpha=0.3)
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
        palette = {'0': '#E53935', '1': '#43A047'}
        for idx, feature in enumerate(features):
            ax = axes[idx // 3, idx % 3]
            sns.boxplot(x='Potability', y=feature, data=plot_data.dropna(subset=[feature]),
                        palette=palette, ax=ax)
            ax.set_title(feature, fontsize=11, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab4:
        st.markdown("#### Missing Values Analysis")
        nulls = data.isnull().sum()
        null_pct = (nulls / len(data) * 100).round(1)
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ['#E53935' if n > 0 else '#43A047' for n in nulls]
        bars = ax.bar(nulls.index, nulls.values, color=colors, edgecolor='white')
        for bar, count, pct in zip(bars, nulls.values, null_pct.values):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
                        f'{count}\n({pct}%)', ha='center', fontsize=9, fontweight='bold')
        ax.set_title('Missing Values per Feature', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
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
            bars = ax.bar(['Not Potable (0)', 'Potable (1)'], counts.values,
                          color=['#E53935', '#43A047'], edgecolor='white')
            for bar, count in zip(bars, counts.values):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 10,
                        str(count), ha='center', fontsize=12, fontweight='bold')
            ax.set_title('Count', fontsize=13, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.pie(counts.values, labels=['Not Potable', 'Potable'],
                   colors=['#E53935', '#43A047'], autopct='%1.1f%%', startangle=90)
            ax.set_title('Percentage', fontsize=13, fontweight='bold')
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
        <h1>🏆 Model Comparison</h1>
        <p>Comparing Random Forest vs SVM vs Logistic Regression</p>
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
        comp_df.style.highlight_max(axis=0, color='#c8e6c9')
                     .format('{:.4f}'),
        use_container_width=True
    )

    # Visual comparison
    st.markdown("### 📈 Visual Comparison")

    models = list(comparison.keys())
    metric_names = ['accuracy', 'precision', 'recall', 'f1_score']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(metric_names))
    width = 0.25
    colors = ['#2196F3', '#FF9800', '#4CAF50']

    for i, model_name in enumerate(models):
        values = [comparison[model_name][m] for m in metric_names]
        bars = ax.bar(x + i * width, values, width, label=model_name, color=colors[i])
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)
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
        <h1>ℹ️ About This Project</h1>
        <p>Water Potability Prediction using Machine Learning with MLOps</p>
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
    <div style='text-align: center; padding: 1rem; opacity: 0.6;'>
        <p>Built with ❤️ using Python, Scikit-Learn, DVC & Streamlit</p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# PAGE 5: HEALTH REPORT
# ============================================================
elif page == "\U0001f4cb Health Report":
    st.markdown("""
    <div class="main-header">
        <h1>\U0001f4cb AI Water Health Report</h1>
        <p>Comprehensive safety assessment of your water sample powered by AI</p>
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
    score_color = "#43A047" if safety_score >= 78 else ("#FF9800" if safety_score >= 45 else "#E53935")
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
        <h1>\U0001f916 Ask the Water Bot</h1>
        <p>Chat with an AI expert about your specific water test results</p>
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
        <h1>\U0001f48a Water Treatment Guide</h1>
        <p>AI-generated step-by-step plan to make your water safe for drinking</p>
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
