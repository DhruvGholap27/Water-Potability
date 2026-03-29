import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns


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

    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #0077B6, #00B4D8, #48CAE4);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 119, 182, 0.3);
    }

    .main-header h1 {
        font-size: 2.4rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }

    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }

    .result-card {
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        animation: fadeIn 0.5s ease-in;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .potable {
        background: linear-gradient(135deg, #43A047, #66BB6A);
        color: white;
    }

    .not-potable {
        background: linear-gradient(135deg, #E53935, #EF5350);
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
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        padding: 1.2rem;
        border-radius: 12px;
        text-align: center;
        border: 1px solid #dee2e6;
        transition: transform 0.2s ease;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }

    .metric-card h3 {
        font-size: 1.8rem;
        color: #0077B6;
        margin: 0;
        font-weight: 700;
    }

    .metric-card p {
        color: #495057;
        font-size: 0.85rem;
        margin: 0.3rem 0 0 0;
        font-weight: 500;
    }

    .feature-info {
        background: #f0f7ff;
        padding: 1rem 1.2rem;
        border-radius: 10px;
        border-left: 4px solid #0077B6;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }

    .stSlider > div > div { color: #0077B6; }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }

    .confidence-bar {
        height: 30px;
        border-radius: 15px;
        overflow: hidden;
        background: #e9ecef;
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
# SIDEBAR NAVIGATION
# ============================================================
st.sidebar.markdown("## 💧 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🔮 Predict Water Quality", "📊 Data Exploration", "🏆 Model Comparison", "ℹ️ About Project"],
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
