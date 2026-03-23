import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ScoreScope",
    page_icon="🎯",
    layout="centered"
)

# ── Custom CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800;900&family=Space+Grotesk:wght@400;500;600;700&display=swap');

* { font-family: 'Space Grotesk', sans-serif !important; }

html, body, [data-testid="stAppViewContainer"] {
    background-color: #F0F4FF !important;
    color: #1A1A2E !important;
}

[data-testid="stAppViewContainer"] {
    background-image:
        radial-gradient(ellipse 60% 40% at 15% 10%, rgba(99,102,241,0.10) 0%, transparent 60%),
        radial-gradient(ellipse 50% 35% at 85% 85%, rgba(236,72,153,0.08) 0%, transparent 55%),
        radial-gradient(ellipse 40% 30% at 70% 20%, rgba(16,185,129,0.07) 0%, transparent 50%);
}

[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"] { display: none !important; }

h2, h3 {
    font-family: 'Nunito', sans-serif !important;
    font-weight: 800 !important;
    color: #3730A3 !important;
    font-size: 1.1rem !important;
    letter-spacing: -0.01em !important;
    margin-top: 1.5rem !important;
}

p, label { color: #374151 !important; }

[data-testid="stCaptionContainer"] p {
    color: #9CA3AF !important;
    font-size: 0.78rem !important;
}

hr {
    border: none !important;
    border-top: 2px dashed #C7D2FE !important;
    margin: 1.8rem 0 !important;
}

[data-testid="stSlider"] label p {
    color: #4B5563 !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em !important;
    text-transform: uppercase !important;
}

[data-baseweb="slider"] div[role="slider"] {
    background-color: #6366F1 !important;
    border-color: #6366F1 !important;
    width: 18px !important;
    height: 18px !important;
}

[data-baseweb="slider"] [data-testid="stSliderTrackFill"] {
    background: linear-gradient(90deg, #6366F1, #EC4899) !important;
}

[data-testid="stButton"] button[kind="primary"] {
    background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 50%, #EC4899 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 50px !important;
    font-family: 'Nunito', sans-serif !important;
    font-weight: 800 !important;
    font-size: 1rem !important;
    letter-spacing: 0.02em !important;
    padding: 0.7rem 2.8rem !important;
    box-shadow: 0 4px 20px rgba(99,102,241,0.35) !important;
    transition: all 0.25s ease !important;
}

[data-testid="stButton"] button[kind="primary"]:hover {
    box-shadow: 0 8px 30px rgba(99,102,241,0.5) !important;
    transform: translateY(-2px) scale(1.02) !important;
}

[data-testid="stAlert"] {
    border-radius: 16px !important;
    border: none !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
}

[data-testid="stProgressBar"] > div {
    background-color: #E0E7FF !important;
    border-radius: 50px !important;
    height: 10px !important;
}
[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #6366F1, #EC4899) !important;
    border-radius: 50px !important;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #EEF2FF 0%, #FAF5FF 100%) !important;
    border-right: 2px solid #E0E7FF !important;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    font-family: 'Nunito', sans-serif !important;
    font-weight: 800 !important;
    color: #4F46E5 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] li {
    color: #6B7280 !important;
    font-size: 0.85rem !important;
    line-height: 1.9 !important;
}

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #F0F4FF; }
::-webkit-scrollbar-thumb { background: #C7D2FE; border-radius: 50px; }
</style>
""", unsafe_allow_html=True)

# ── Load model & scaler ───────────────────────────────────────────────────
@st.cache_resource
def load_models():
    rf     = joblib.load("models/rf_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return rf, scaler

rf, scaler = load_models()

# ── Header ────────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding-top:2rem;margin-bottom:0.5rem'>
  <span style='font-family:Nunito,sans-serif;font-size:2.4rem;font-weight:900;
    background:linear-gradient(135deg,#6366F1 0%,#EC4899 60%,#F59E0B 100%);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    background-clip:text;display:inline-block'>ScoreScope 🎯</span>
</div>
<p style='color:#6B7280;font-size:0.92rem;margin-top:0.2rem;
  font-family:Space Grotesk,sans-serif;font-weight:500'>
  Find out if a student is likely to
  <strong style='color:#6366F1'>pass</strong> or
  <strong style='color:#EC4899'>fail</strong> — powered by Machine Learning
</p>
""", unsafe_allow_html=True)
st.markdown("---")

# ── Input sliders ─────────────────────────────────────────────────────────
st.subheader("📋 Student Profile")

col1, col2 = st.columns(2)
with col1:
    study   = st.slider("Study hours per day",  1.0, 10.0, 5.0, 0.5)
    attend  = st.slider("Attendance hours (sem)", 10.0, 50.0, 30.0, 1.0)
with col2:
    assign  = st.slider("Assignments completed", 0, 10, 5)
    past    = st.slider("Past score (weak feature)", 30.0, 100.0, 60.0, 1.0)

st.caption("💡 Tip: Past score is intentionally treated as a weak predictor.")

# ── Predict ───────────────────────────────────────────────────────────────
if st.button("✨  Predict Outcome", type="primary"):
    X = pd.DataFrame([[study, attend, assign, past]],
                     columns=["study_hours","attendance_hours",
                              "assignments","past_score"])
    X_sc   = scaler.transform(X)
    pred   = rf.predict(X_sc)[0]
    prob   = rf.predict_proba(X_sc)[0]

    st.markdown("---")
    st.subheader("🎯 Prediction Result")

    if pred == 1:
        st.success(f"🏆  PASS — Confidence: {prob[1]*100:.1f}%")
    else:
        st.error(f"⚠️  FAIL — Confidence: {prob[0]*100:.1f}%")

    st.markdown(
        "<p style='color:#6366F1;font-size:0.8rem;font-weight:700;"
        "letter-spacing:0.06em;text-transform:uppercase;"
        "margin-bottom:0.3rem;font-family:Nunito,sans-serif'>"
        "Pass Probability</p>",
        unsafe_allow_html=True
    )
    st.progress(float(prob[1]))

    st.markdown("---")
    st.subheader("📊 Feature Contribution")

    weights = [0.40, 0.30, 0.25, 0.05]
    feat_names = ["Study hours", "Attendance", "Assignments", "Past score"]
    raw = [study/10, attend/50, assign/10, (past-30)/70]
    contributions = [w * v for w, v in zip(weights, raw)]

    fig, ax = plt.subplots(figsize=(6, 3))
    fig.patch.set_facecolor("#FAFBFF")
    ax.set_facecolor("#FAFBFF")

    bar_colors = ["#D1D5DB" if n == "Past score" else "#6366F1" for n in feat_names]
    bars = ax.barh(feat_names, contributions, color=bar_colors,
                   height=0.52, edgecolor="none")

    for bar, val in zip(bars, contributions):
        ax.text(val + 0.003, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va='center', ha='left',
                fontsize=8, color="#6B7280")

    ax.set_xlabel("Weighted contribution", color="#9CA3AF", fontsize=8, labelpad=8)
    ax.set_title("How each feature influenced this prediction",
                 color="#374151", fontsize=9, pad=10, fontweight='bold')
    ax.axvline(x=0, color="#E5E7EB", linewidth=1)
    ax.tick_params(colors="#6B7280", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#E5E7EB")
    plt.tight_layout()
    st.pyplot(fig)

    st.caption("🟣 Purple = strong features   ·   ⬜ Grey = weak feature (past score)")

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<p style='color:#4F46E5;font-size:0.75rem;font-weight:800;"
        "letter-spacing:0.1em;text-transform:uppercase;"
        "font-family:Nunito,sans-serif;margin-bottom:0.5rem'>🤖 Model Info</p>",
        unsafe_allow_html=True
    )
    st.markdown("---")
    st.write("*Algorithm:* Random Forest Classifier")
    st.write("*Dataset:* 500 synthetic students")
    st.write("*Train/Test split:* 80/20")
    st.write("*CV Accuracy:* ~92%")
    st.markdown("---")
    st.markdown(
        "<p style='color:#4F46E5;font-size:0.75rem;font-weight:800;"
        "letter-spacing:0.1em;text-transform:uppercase;"
        "font-family:Nunito,sans-serif;margin-bottom:0.5rem'>⚖️ Feature Weights</p>",
        unsafe_allow_html=True
    )
    st.write("- Study hours: *40%*")
    st.write("- Attendance: *30%*")
    st.write("- Assignments: *25%*")
    st.write("- Past score: *5%* ← weak")
    st.markdown("---")
    st.markdown(
        "<p style='color:#9CA3AF;font-size:0.72rem;text-align:center;"
        "font-family:Space Grotesk,sans-serif'>ScoreScope · Made with ❤️ for students</p>",
        unsafe_allow_html=True
    )
    