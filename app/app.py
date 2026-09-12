"""Streamlit single-page app: estimates a firm's expected AI-driven productivity increase."""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
SAMPLE_AVERAGE = 11.0  # mean of ai_f2a2 in the modeling sample
SCALE_MAX = 35  # visual gauge scale, %

# Human-readable labels for the 25 model features, used in the "what drove your
# result" chart and callouts -- feature codes alone aren't readable in a chart axis.
FEATURE_LABELS = {
    "reskilling_need_share": "Reskilling need",
    "workers_wanting_more_hours_share": "Workers wanting more hours",
    "idle_time_share": "Idle production time",
    "admin_worker_share": "Admin worker share",
    "share_clerical": "Clerical staff share",
    "sales_seasonality_ratio": "Sales seasonality",
    "policy_support_score": "Policy support score",
    "ai_b3a3_flag": "AI used for drafting",
    "share_professionals": "Professional staff share",
    "ai_b2a4_flag": "Generative AI use",
    "share_managers": "Manager staff share",
    "worker_seasonality_ratio": "Worker seasonality",
    "uses_genai": "Generative AI use",
    "share_computer_use": "Computer use share",
    "log_workers": "Firm size",
    "task_breadth": "AI task breadth",
    "finance_digitization": "Finance digitization",
    "ai_g1b_flag": "Workforce-training policy support",
    "customer_digitization": "Customer digitization",
    "ai_b3a2_flag": "AI used for summarizing",
    "can_absorb_demand_shock": "Demand-shock capacity",
    "supplychain_digitization": "Supply chain digitization",
    "ai_breadth_score": "AI technology breadth",
    "ai_g1f_flag": "Graduate-skills policy support",
    "has_website": "Has a website",
}


def feature_label(f: str) -> str:
    return FEATURE_LABELS.get(f, f)


@st.cache_data
def load_permutation_importance():
    df = pd.read_csv(ROOT / "reports" / "permutation_importance.csv")
    return df.set_index("feature")["importance_mean"].to_dict()


@st.cache_data
def load_modeling_sample():
    # Deliberately NOT data/processed/model_data.csv: that directory is gitignored
    # (regeneratable pipeline intermediate) and doesn't exist on a fresh clone/deploy.
    # models/sample_outcomes.csv is the app-facing subset (just the 2 columns the
    # charts below need), staged into the tracked models/ dir by
    # src/07_compare_and_select.py.
    return pd.read_csv(MODELS_DIR / "sample_outcomes.csv")

st.set_page_config(page_title="Your AI Productivity Edge", page_icon="📈", layout="centered")

# Fields not collected from the user -- fixed at these defaults (roughly the modeling
# sample's typical/median values) per an explicit request to cut the form down to just:
# reskilling need %, idle time %, workers-wanting-more-hours %, AI technologies, AI
# tasks, and 1-2 digitization questions.
DEFAULTS = {
    "total_workers": 25,
    "pct_managers": 10,
    "pct_professionals": 15,
    "pct_clerical": 10,
    "pct_admin": 10,
    "has_website": "Yes",
    "pct_computer_use": 40,
    "supplychain_digitization": 2,
    "can_absorb_shock": "Yes",
    "peak_sales": 10.0,
    "low_sales": 8.0,
    "peak_workers": 25,
    "low_workers": 22,
    "policy_support": [],
}

# ---------------------------------------------------------------------------
# Model / preprocessing loading
# ---------------------------------------------------------------------------


@st.cache_resource
def load_artifacts():
    with open(MODELS_DIR / "final_model_meta.json") as fh:
        meta = json.load(fh)
    # Not data/processed/feature_list.json -- gitignored, absent on a fresh clone.
    # models/feature_list.json is the tracked copy staged by 07_compare_and_select.py.
    with open(MODELS_DIR / "feature_list.json") as fh:
        feature_spec = json.load(fh)

    imputer = joblib.load(MODELS_DIR / "imputer.joblib")
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")

    if meta["kind"] == "keras":
        from tensorflow import keras
        model = keras.models.load_model(MODELS_DIR / meta["artifact"])
        if meta.get("target_scaled"):
            y_scaler = joblib.load(MODELS_DIR / "target_scaler.joblib")
            def predict_fn(X):
                pred_scaled = model.predict(X, verbose=0).flatten()[0]
                return float(y_scaler.inverse_transform([[pred_scaled]])[0, 0])
        else:
            predict_fn = lambda X: float(model.predict(X, verbose=0).flatten()[0])
    else:
        model = joblib.load(MODELS_DIR / meta["artifact"])
        predict_fn = lambda X: float(model.predict(X)[0])

    return meta, feature_spec["features"], imputer, scaler, predict_fn


def predict(features: dict) -> float:
    meta, feature_names, imputer, scaler, predict_fn = load_artifacts()
    row = pd.DataFrame([[features.get(f, np.nan) for f in feature_names]], columns=feature_names)
    imputed = imputer.transform(row)
    if meta["feature_space"] == "scaled":
        X = scaler.transform(imputed)
    else:
        X = imputed
    pred = predict_fn(X)
    return max(pred, 0.0), meta


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

CSS = """
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root {
    --bg: #0D0F26;
    --panel: #191C3D;
    --panel-2: #232750;
    --border: #33366B;
    --text: #F7F7FB;
    --text-dim: #C2C4E8;
    --text-faint: #8C8FC0;
    --accent-purple: #C77DFF;
    --accent-yellow: #FFD966;
    --gradient: linear-gradient(135deg, #C77DFF 0%, #FFD966 100%);
}
html { scroll-behavior: smooth; }
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background:
      radial-gradient(circle at 15% 20%, rgba(199, 125, 255, 0.18) 0%, transparent 45%),
      radial-gradient(circle at 85% 75%, rgba(255, 217, 102, 0.12) 0%, transparent 45%),
      radial-gradient(circle at 50% 100%, rgba(199, 125, 255, 0.10) 0%, transparent 60%),
      #0D0F26 !important;
    min-height: 100vh;
}
.stApp { background-attachment: fixed; color: var(--text); font-family: 'Inter', sans-serif; }
/* neutralize any competing background Streamlit's own theme sets on these --
   this is what was boxing the gradient into block-container's bounds */
[data-testid="stHeader"], .block-container, [data-testid="stMainBlockContainer"] {
    background: transparent !important;
}
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; color: var(--text) !important; }
p, span, label, div { color: var(--text); }
.block-container { max-width: 720px; margin: 0 auto !important; padding-top: 1.4rem; padding-bottom: 2.5rem; font-size: 1.18rem; }

/* Streamlit's own widget labels/captions don't pick up rem-based sizing on ancestors
   the way our custom classes do (Streamlit sets its own explicit font-size) --
   target them directly so slider/pills labels actually get bigger too. */
div[data-testid="stWidgetLabel"] p { font-size: 1.15rem !important; }
div[data-testid="stMarkdownContainer"] p { font-size: 1.15rem !important; }

.section-heading { font-size: 1.85rem; font-weight: 700; font-family: 'Space Grotesk', sans-serif; margin: 1.8rem 0 0.2rem; }
.section-sub { color: var(--text-dim); font-size: 1.35rem; margin-bottom: 0.9rem; }

div[data-testid="stButton"] > button {
  background: var(--gradient);
  color: #201202;
  font-weight: 700;
  border: none;
  border-radius: 10px; padding: 0.55rem 1.4rem;
}
div[data-testid="stButton"] > button:hover { filter: brightness(1.08); color: #201202; }

/* sliders: force purple thumb/track, overriding any leftover theme orange */
div[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
  background-color: var(--accent-purple) !important;
  border-color: var(--accent-purple) !important;
}
div[data-testid="stSlider"] [data-baseweb="slider"] > div > div {
  background: var(--accent-purple) !important;
}

/* gauge */
.gauge-wrap { display:flex; justify-content:center; margin: 0.6rem 0; }
.gauge-ring {
    width: 220px; height: 220px; border-radius: 50%;
    background: conic-gradient(var(--accent-purple) 0deg, var(--accent-yellow) calc(var(--fill-deg) * 1deg), var(--panel-2) calc(var(--fill-deg) * 1deg) 360deg);
    /* --panel-2 (the unfilled track) is close in tone to the page's own
       background gradient -- without an edge, the ring reads as a floating
       arc rather than a full ring. A thin border makes the boundary legible
       regardless of fill/backdrop contrast, without altering the conic-gradient. */
    border: 1px solid var(--border);
    display: flex; align-items: center; justify-content: center;
}
.gauge-inner {
  width: 168px; height: 168px; border-radius: 50%; background: var(--panel);
  display:flex; flex-direction:column; align-items:center; justify-content:center;
}
.gauge-num { font-family:'Space Grotesk', sans-serif; font-size: 2.3rem; font-weight:700; color: var(--accent-purple); }
.gauge-lbl { font-size: 0.9rem; color: var(--text-faint); }

.compare-row { margin-bottom: 0.6rem; }
.compare-label { display:flex; justify-content:space-between; font-size:1.05rem; color: var(--text-dim); margin-bottom:4px; }
.compare-track { background: var(--panel-2); border-radius: 8px; height: 10px; overflow:hidden; }
.compare-fill { height: 100%; border-radius: 8px; }

.footnote { color: var(--text-faint); font-size: 0.85rem; margin-top: 1.1rem; line-height:1.5; }
.insight { background: var(--panel-2); border-left: 3px solid var(--accent-purple); border-radius: 8px; padding: 0.7rem 1rem; margin: 0.7rem 0; font-size: 1.1rem; }
/* self-contained insight cards used for the "In plain terms / Compared to similar
   firms / Biggest driver / Next step / Keep in mind" breakdown -- each one is its
   own card, not sub-sections of one big block */
.insight-card { position: relative; background: var(--panel); border-radius: 10px; padding: 0.85rem 1.1rem 0.85rem 1.4rem; margin: 0.7rem 0; overflow: hidden; font-size: 1.1rem; line-height: 1.5; }
.insight-card-bar { position: absolute; left: 0; top: 0; bottom: 0; width: 4px; background: var(--gradient); }
.insight-card-note { color: var(--text-faint); font-size: 0.85rem; margin-top: 0.4rem; line-height: 1.4; }
.chart-caption { font-size: 0.85rem; color: var(--text-faint); margin: -0.3rem 0 0.9rem; }

/* hero */
div.st-key-hero_wrap { text-align: center; padding: 1rem 0 0.3rem; }
.hero-content { max-width: 700px; margin: 0 auto; text-align: center; }
.hero-title {
    background: var(--gradient); -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(2.2rem, 4vw, 3rem); line-height: 1.1; font-weight: 700; margin-bottom: 0.6rem;
    text-align: center; white-space: nowrap;
}
.hero-sub { color: var(--text-dim); font-size: clamp(0.8rem, 2vw, 1rem); margin-bottom: 1.5rem; white-space: nowrap; }
.hero-stats { display: flex; justify-content: center; gap: 2.8rem; margin-bottom: 1.2rem; }
.hero-stat { text-align: center; }
.hero-stat-icon { margin: 0 auto 0.35rem; display: block; }
.hero-stat-num {
    background: var(--gradient); -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; font-family: 'Space Grotesk', sans-serif;
    font-size: 2.2rem; font-weight: 700;
}
.hero-stat-label { font-size: 0.9rem; color: var(--text-faint); margin-top: 0.2rem; }
.hero-divider { width: 80px; height: 3px; border-radius: 2px; background: var(--gradient); margin: 0 auto 1rem; }
.hero-features { display: flex; justify-content: center; gap: 1.6rem; margin-bottom: 1.1rem; flex-wrap: wrap; }
.hero-feature { font-size: 1rem; color: var(--text-dim); white-space: nowrap; }
.hero-feature-check { color: var(--accent-purple); font-weight: 700; margin-right: 0.3rem; }
.jump-link { display: inline-block; margin-top: 0.6rem; color: var(--text-dim); font-size: 0.85rem; text-decoration: none; border-bottom: 1px dashed var(--border); padding-bottom: 2px; }
.jump-link:hover { color: var(--accent-purple); border-color: var(--accent-purple); }

/* Intro-page hero only (div.st-key-intro_hero_wrap) -- bigger/more prominent than
   the results-page hero, which still uses the base .hero-* sizing above via its
   own div.st-key-hero_wrap container. These overrides win on specificity
   (container class + element class) without touching the shared base rules. */
div.st-key-intro_hero_wrap { padding: 2rem 0 0.7rem; }
div.st-key-intro_hero_wrap .hero-content { max-width: 720px; }
/* Max bound kept below the ~3.8rem originally tried -- at that size "Your AI
   Productivity Edge" overflowed the 720px container and got clipped by
   white-space:nowrap (confirmed visually, not just estimated). 3.2rem is the
   largest that reliably fits this exact title on one line at this width. */
div.st-key-intro_hero_wrap .hero-title { font-size: clamp(2.8rem, 5vw, 3.2rem); }
div.st-key-intro_hero_wrap .hero-sub { font-size: clamp(0.95rem, 2.3vw, 1.15rem); margin-bottom: 2rem; }
div.st-key-intro_hero_wrap .hero-stats { gap: 3.4rem; margin-bottom: 1.6rem; }
div.st-key-intro_hero_wrap .hero-stat-num { font-size: 3.1rem; }
div.st-key-intro_hero_wrap .hero-stat-label { font-size: 1rem; }
div.st-key-intro_hero_wrap .hero-features { gap: 1.9rem; margin-bottom: 0.85rem; }
div.st-key-intro_hero_wrap .hero-feature { font-size: 1.1rem; }

/* "About" block -- 5 explanatory sections between the hero and the first input
   section. Reuses .section-heading's type scale but with a tighter top margin
   so the 5 read as one cohesive group rather than 5 separate page sections.
   Deliberately more compact than the hero above it -- supporting/reference
   material, not the visual anchor of the page. */
.about-heading { font-size: 1.85rem; font-weight: 700; font-family: 'Space Grotesk', sans-serif; margin: 0.75rem 0 0.15rem; }
.about-card { background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 0.8rem 0.95rem; margin: 0; font-size: 0.92rem; line-height: 1.45; }
.stat-highlight { color: var(--accent-purple); font-weight: 700; }

.about-flow { display: flex; align-items: center; justify-content: center; flex-wrap: wrap; gap: 0.4rem; margin-top: 1rem; }
.about-flow-step {
    background: var(--panel-2); border: 1px solid var(--accent-purple);
    border-radius: 10px; padding: 0.5rem 0.5rem; font-size: 0.8rem; font-weight: 600;
    color: var(--text-dim); text-align: center; flex: 1 1 90px; min-width: 80px;
}
.about-flow-step.alt { background: var(--panel); border-color: var(--accent-yellow); }
.about-flow-arrow { color: var(--accent-purple); font-size: 1.1rem; font-weight: 700; flex: 0 0 auto; }
@media (max-width: 600px) {
  .about-flow-arrow { display: none; }
  .about-flow-step { flex: 1 1 45%; }
}
.about-stats { display: flex; justify-content: center; gap: 2.4rem; margin-top: 1rem; flex-wrap: wrap; }

/* big centered CTA at the bottom of the input page */
div.st-key-cta_wrap { display: flex; justify-content: center; margin: 1.6rem 0 1rem; }
div.st-key-cta_wrap div[data-testid="stButton"] > button {
    padding: 1rem 2.8rem !important; font-size: 1.25rem !important;
}
/* small outlined "back" button at the top of the results page */
div.st-key-edit_wrap div[data-testid="stButton"] > button {
    background: transparent !important; color: var(--text-dim) !important;
    border: 1px solid var(--border) !important; font-weight: 600 !important;
    padding: 0.45rem 1.1rem !important; font-size: 1rem !important;
}
</style>
"""

st.html(CSS)

_ICON_TARGET = """<svg class="hero-stat-icon" width="20" height="20" viewBox="0 0 24 24" fill="none">
<circle cx="12" cy="12" r="9" stroke="#C77DFF" stroke-width="2"/><circle cx="12" cy="12" r="4" stroke="#C77DFF" stroke-width="2"/>
</svg>"""
_ICON_CHART = """<svg class="hero-stat-icon" width="20" height="20" viewBox="0 0 24 24" fill="none">
<rect x="4" y="12" width="4" height="8" fill="#C77DFF"/><rect x="10" y="7" width="4" height="13" fill="#C77DFF"/><rect x="16" y="3" width="4" height="17" fill="#C77DFF"/>
</svg>"""
_ICON_CHECK = """<svg class="hero-stat-icon" width="20" height="20" viewBox="0 0 24 24" fill="none">
<circle cx="12" cy="12" r="9" stroke="#C77DFF" stroke-width="2"/><path d="M8 12l3 3 5-6" stroke="#C77DFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
</svg>"""

if "page" not in st.session_state:
    st.session_state.page = "intro"  # "intro" | "inputs" | "results"
# Explicit persisted store for input values, independent of widget key state.
# Streamlit does NOT reliably keep a widget's session_state[key] value when the
# widget is omitted from one or more reruns and then re-instantiated later
# (confirmed empirically: st.pills reset to its `default=` after a round trip
# through the results page) -- so inputs are saved here on "See my results"
# and fed back in as each widget's initial value on return, rather than relying
# on Streamlit to remember them on its own.
if "saved" not in st.session_state:
    st.session_state.saved = {
        "ai_tech": [], "ai_tasks": [], "finance_dig": 2, "customer_dig": 2,
        "reskilling": 15, "idle_time": 10, "more_hours": 10,
    }

meta, _, _, _, _ = load_artifacts()
total_firms = meta["n_train"] + meta["n_test"]

# ===========================================================================
# PAGE 1: INTRO -- hero + the 5 "About" cards. Only this page's content shows
# while st.session_state.page == "intro".
# ===========================================================================

if st.session_state.page == "intro":

    with st.container(key="intro_hero_wrap"):
        st.html(f"""
        <div class="hero-content">
          <div class="hero-title">Your AI Productivity Edge</div>
          <div class="hero-sub">A data-driven estimate of your firm's AI upside ; built from a survey of {total_firms} Indian businesses.</div>
          <div class="hero-stats">
            <div class="hero-stat">{_ICON_TARGET}<div class="hero-stat-num">{meta['r2']:.2f}</div><div class="hero-stat-label">Test R&sup2;</div></div>
            <div class="hero-stat">{_ICON_CHART}<div class="hero-stat-num">{total_firms}</div><div class="hero-stat-label">Firms</div></div>
            <div class="hero-stat">{_ICON_CHECK}<div class="hero-stat-num">{meta['mae']:.1f}pp</div><div class="hero-stat-label">Average absolute error</div></div>
          </div>
          <div class="hero-divider"></div>
          <div class="hero-features">
            <div class="hero-feature"><span class="hero-feature-check">&#10003;</span>25 predictive features</div>
            <div class="hero-feature"><span class="hero-feature-check">&#10003;</span>Instant estimate</div>
            <div class="hero-feature"><span class="hero-feature-check">&#10003;</span>Built on real survey data</div>
          </div>
        </div>
        """)

    # -----------------------------------------------------------------------
    # About block -- 5 explanatory sections, below the hero on this same page.
    # -----------------------------------------------------------------------

    flow_steps = ["Your inputs", "Preprocessing", "25 features", "Extra Trees model", "Expected productivity gain"]
    flow_html = '<div class="about-flow">'
    for i, step in enumerate(flow_steps):
        step_cls = "about-flow-step alt" if i % 2 == 1 else "about-flow-step"
        flow_html += f'<div class="{step_cls}">{step}</div>'
        if i < len(flow_steps) - 1:
            flow_html += '<div class="about-flow-arrow">&rarr;</div>'
    flow_html += "</div>"

    st.html("""
    <div class="about-heading">What is this project?</div>
    <div class="about-card">This project uses machine learning and statistical regression
    techniques to estimate a firm's expected productivity gain from AI adoption. The model
    analyses data from 986 Indian firms across AI adoption, workforce characteristics, digital
    maturity, and operating conditions to identify patterns associated with firms' expected
    productivity gains.</div>
    """)

    st.html(f"""
    <div class="about-heading">How does the model work?</div>
    <div class="about-card">Your responses are processed through the same preprocessing
    pipeline used during model development. The model then evaluates 25 predictive features
    covering AI usage, task breadth, digitization maturity, workforce characteristics, and
    operating conditions before generating an estimate of your firm's expected productivity gain.
    {flow_html}
    </div>
    """)

    st.html(f"""
    <div class="about-heading">About the dataset</div>
    <div class="about-card">The study uses a stratified survey of 1,355 Indian firms. After
    filtering for valid responses to the expected productivity-gain measure, {total_firms} firms
    were retained for modelling. The dataset captures information on AI adoption, workforce
    composition, digitization maturity, firm characteristics, and operating conditions.
    <div class="about-stats">
      <div class="hero-stat"><div class="hero-stat-num">1,355</div><div class="hero-stat-label">Firms surveyed</div></div>
      <div class="hero-stat"><div class="hero-stat-num">{total_firms}</div><div class="hero-stat-label">Valid modelling responses</div></div>
      <div class="hero-stat"><div class="hero-stat-num">25</div><div class="hero-stat-label">Predictive features</div></div>
    </div>
    </div>
    """)

    st.html("""
    <div class="about-heading">Why measure AI productivity?</div>
    <div class="about-card">AI adoption is changing how businesses perform everyday tasks, make
    decisions, and allocate resources. Measuring its potential productivity impact can help
    organisations understand where AI may create value and where additional investment or
    organisational changes may be required.</div>
    """)

    st.html(f"""
    <div class="about-heading">Understanding the model performance</div>
    <div class="about-card">The deployed Extra Trees model achieves a test R&sup2; of
    <span class="stat-highlight">{meta['r2']:.2f}</span>, indicating that it explains a substantial
    share of the variation in firms' reported expected productivity gains.</div>
    """)

    # -----------------------------------------------------------------------
    # Section 6: What does the research show? -- model comparison table.
    # -----------------------------------------------------------------------
    st.html("""
    <div class="about-heading">What does the research show?</div>
    <div class="about-card">Ten regression approaches were compared, spanning linear models,
    tuned tree ensembles, a stacking ensemble, and two deep-learning architectures. The deployed
    Extra Trees model achieved the strongest result.</div>
    """)
    comparison_df = pd.DataFrame({
        "Model": [
            "Extra Trees (deployed)", "Stacking", "Random Forest", "Gradient Boosting",
            "Linear Regression", "Ridge", "Lasso", "Feedforward Neural Network",
            "Wide & Deep Hybrid", "Mean baseline",
        ],
        "Test R²": [0.621, 0.621, 0.617, 0.607, 0.481, 0.480, 0.487, 0.465, 0.395, -0.003],
    })

    def _highlight_winner(row):
        if row["Model"] == "Extra Trees (deployed)":
            return ["background-color: #232750; color: #C77DFF; font-weight: 700;"] * len(row)
        return [""] * len(row)

    styled_comparison = comparison_df.style.apply(_highlight_winner, axis=1).format({"Test R²": "{:.3f}"})
    st.dataframe(styled_comparison, hide_index=True, width="stretch")

    # -----------------------------------------------------------------------
    # Section: Key research finding.
    # -----------------------------------------------------------------------
    st.html("""
    <div class="about-heading">Key research finding</div>
    <div class="about-card">
    <strong class="stat-highlight" style="font-size: 1.05em;">AI adoption isn't the whole story.</strong><br><br>
    Our analysis suggests that firms' expected productivity gains are strongly associated with
    their existing operating conditions. Factors such as reskilling needs and underutilised
    production capacity were stronger predictors than AI adoption breadth alone.
    <div class="about-stats">
      <div class="hero-stat"><div class="hero-stat-num">0.128</div><div class="hero-stat-label">Reskilling need (&Delta;R&sup2;)</div></div>
      <div class="hero-stat"><div class="hero-stat-num">0.098</div><div class="hero-stat-label">Workers wanting more hours (&Delta;R&sup2;)</div></div>
      <div class="hero-stat"><div class="hero-stat-num">0.072</div><div class="hero-stat-label">Idle production time (&Delta;R&sup2;)</div></div>
    </div>
    </div>
    """)

    # -----------------------------------------------------------------------
    # Section 7: About the research -- paper citation + summary stats. Final
    # section on this page, directly above the button into the input form.
    # -----------------------------------------------------------------------
    st.html("""
    <div class="about-heading">About the research</div>
    <div class="about-card">This application accompanies the research paper "Predicting Firms'
    Expected Productivity Gains from Generative AI: A Machine Learning and Deep Learning
    Comparison Using Indian Firm-Level Data."
    <div class="about-stats">
      <div class="hero-stat"><div class="hero-stat-num">986 / 1,355</div><div class="hero-stat-label">Valid responses / surveyed firms</div></div>
      <div class="hero-stat"><div class="hero-stat-num">25</div><div class="hero-stat-label">Validated predictors</div></div>
      <div class="hero-stat"><div class="hero-stat-num">10</div><div class="hero-stat-label">Models compared (ML + DL)</div></div>
      <div class="hero-stat"><div class="hero-stat-num">Extra Trees</div><div class="hero-stat-label">Deployed model (R&sup2; = 0.621)</div></div>
    </div>
    </div>
    """)

    def _go_to_inputs():
        st.session_state.page = "inputs"

    with st.container(key="cta_wrap"):
        st.button("Get started →", on_click=_go_to_inputs)

# ===========================================================================
# PAGE 2: INPUTS -- all input sections. Only shows while page == "inputs".
# ===========================================================================

elif st.session_state.page == "inputs":

    def _go_to_intro():
        st.session_state.page = "intro"

    with st.container(key="edit_wrap"):
        st.button("← Back", on_click=_go_to_intro)

    saved = st.session_state.saved

    st.html('<div class="section-heading">Your AI footprint</div><div class="section-sub">What AI tools and tasks are already part of your workflow?</div>')
    with st.container(key="pills_wrap", gap=None):
        st.pills(
            "AI technologies in use",
            ["Machine learning", "Chatbots", "AI agents", "Generative AI", "Automation", "Autonomous systems"],
            selection_mode="multi", default=saved["ai_tech"],
            help="Select all that apply", key="ai_tech_pills",
        )
        st.pills(
            "Tasks AI is used for",
            ["Finding info", "Summarizing", "Drafting", "Coding", "Translation", "Analysis", "Process control", "Customer interaction"],
            selection_mode="multi", default=saved["ai_tasks"],
            help="Select all that apply", key="ai_tasks_pills",
        )

    st.html('<div class="section-heading">Digitization maturity</div><div class="section-sub">How advanced are these processes?</div>')
    dig_col1, dig_col2 = st.columns(2)
    with dig_col1:
        st.slider(
            "Finance / accounting digitization", 0, 7, saved["finance_dig"],
            help="0 = none, 7 = fully digitized", key="finance_dig",
        )
    with dig_col2:
        st.slider(
            "Customer-facing digitization", 0, 8, saved["customer_dig"],
            help="0 = none, 8 = fully digitized", key="customer_dig",
        )

    st.html('<div class="section-heading">What matters most</div><div class="section-sub">Capacity and reskilling needs</div>')
    wm_col1, wm_col2 = st.columns(2)
    with wm_col1:
        st.slider("% of workforce needing reskilling", 0, 100, saved["reskilling"], key="reskilling")
    with wm_col2:
        st.slider("% of time production sits idle", 0, 100, saved["idle_time"], key="idle_time")
    st.slider("% of workers wanting more hours", 0, 100, saved["more_hours"], key="more_hours")

    def _go_to_results():
        st.session_state.saved = {
            "ai_tech": st.session_state["ai_tech_pills"],
            "ai_tasks": st.session_state["ai_tasks_pills"],
            "finance_dig": st.session_state["finance_dig"],
            "customer_dig": st.session_state["customer_dig"],
            "reskilling": st.session_state["reskilling"],
            "idle_time": st.session_state["idle_time"],
            "more_hours": st.session_state["more_hours"],
        }
        st.session_state.page = "results"

    with st.container(key="cta_wrap"):
        st.button("See my results →", on_click=_go_to_results)

# ===========================================================================
# PAGE 3: RESULTS -- shown instead of (not alongside) the other two pages.
# ===========================================================================

elif st.session_state.page == "results":
    def _go_back_to_inputs():
        st.session_state.page = "inputs"

    with st.container(key="edit_wrap"):
        st.button("← Edit my inputs", on_click=_go_back_to_inputs)

    # Widgets aren't instantiated on this page -- read the values captured into
    # st.session_state.saved when "See my results" was clicked.
    saved = st.session_state.saved
    ai_tech = saved["ai_tech"]
    ai_tasks = saved["ai_tasks"]
    finance_digitization = saved["finance_dig"]
    customer_digitization = saved["customer_dig"]
    reskilling_pct = saved["reskilling"]
    idle_time_pct = saved["idle_time"]
    more_hours_pct = saved["more_hours"]

    d = DEFAULTS
    sales_ratio = min((d["peak_sales"] / max(d["low_sales"], 1e-6)), 20)
    worker_ratio = min((d["peak_workers"] / max(d["low_workers"], 1e-6)), 20)

    features = {
        "reskilling_need_share": reskilling_pct / 100,
        "workers_wanting_more_hours_share": more_hours_pct / 100,
        "idle_time_share": idle_time_pct / 100,
        "admin_worker_share": d["pct_admin"] / 100,
        "share_clerical": d["pct_clerical"] / 100,
        "sales_seasonality_ratio": sales_ratio,
        "policy_support_score": len(d["policy_support"]),
        "ai_b3a3_flag": int("Drafting" in ai_tasks),
        "share_professionals": d["pct_professionals"] / 100,
        "ai_b2a4_flag": int("Generative AI" in ai_tech),
        "share_managers": d["pct_managers"] / 100,
        "worker_seasonality_ratio": worker_ratio,
        "uses_genai": int("Generative AI" in ai_tech),
        "share_computer_use": d["pct_computer_use"] / 100,
        "log_workers": np.log1p(d["total_workers"]),
        "task_breadth": len(ai_tasks),
        "finance_digitization": finance_digitization,
        "ai_g1b_flag": int("Workforce training" in d["policy_support"]),
        "customer_digitization": customer_digitization,
        "ai_b3a2_flag": int("Summarizing" in ai_tasks),
        "can_absorb_demand_shock": int(d["can_absorb_shock"] == "Yes"),
        "supplychain_digitization": d["supplychain_digitization"],
        "ai_breadth_score": len(ai_tech),
        "ai_g1f_flag": int("Graduate skills programs" in d["policy_support"]),
        "has_website": int(d["has_website"] == "Yes"),
    }

    pred, meta = predict(features)

    with st.container(key="hero_wrap"):
        st.html(f"""
        <div class="hero-content">
          <div class="hero-title">Your Result Is In</div>
          <div class="hero-sub">Based on what you told us, here's your AI productivity estimate.</div>
          <div class="hero-stats">
            <div class="hero-stat">{_ICON_TARGET}<div class="hero-stat-num">{meta['r2']:.2f}</div><div class="hero-stat-label">Test R&sup2;</div></div>
            <div class="hero-stat">{_ICON_CHART}<div class="hero-stat-num">{total_firms}</div><div class="hero-stat-label">Firms</div></div>
            <div class="hero-stat">{_ICON_CHECK}<div class="hero-stat-num">{meta['mae']:.1f}pp</div><div class="hero-stat-label">Average absolute error</div></div>
          </div>
          <div class="hero-divider"></div>
        </div>
        """)

    frac = min(pred / SCALE_MAX, 1.0)
    fill_deg = frac * 360
    st.html(f"""
    <div class="gauge-wrap">
      <div class="gauge-ring" style="--fill-deg: {fill_deg};">
        <div class="gauge-inner">
          <div class="gauge-num">{pred:.1f}%</div>
          <div class="gauge-lbl">of {SCALE_MAX}% scale</div>
        </div>
      </div>
    </div>
    """)

    you_pct = min(pred / SCALE_MAX, 1.0) * 100
    avg_pct = min(SAMPLE_AVERAGE / SCALE_MAX, 1.0) * 100
    st.html(f"""
    <div class="compare-row">
      <div class="compare-label"><span>Your firm</span><span>{pred:.1f}%</span></div>
      <div class="compare-track"><div class="compare-fill" style="width:{you_pct}%; background: var(--accent-purple);"></div></div>
    </div>
    <div class="compare-row">
      <div class="compare-label"><span>Sample average</span><span>{SAMPLE_AVERAGE:.1f}%</span></div>
      <div class="compare-track"><div class="compare-fill" style="width:{avg_pct}%; background: var(--accent-purple);"></div></div>
    </div>
    """)

    if pred > SAMPLE_AVERAGE * 1.15:
        insight = "You're ahead of the curve &mdash; your AI usage and digitization point to above-average gains."
    elif pred < SAMPLE_AVERAGE * 0.85:
        insight = "There's room to grow &mdash; broader AI and digitization adoption could lift this estimate."
    else:
        insight = "Right in line with the pack &mdash; your estimate tracks close to the sample average."
    st.html(f'<div class="insight">{insight}</div>')

    # --- What drove this result: z-score of each input vs. the training sample,
    # weighted by that feature's permutation importance on the deployed model ---
    _, feature_names, _, scaler, _ = load_artifacts()
    imp_map = load_permutation_importance()
    sample_df = load_modeling_sample()

    means = dict(zip(feature_names, scaler.mean_))
    stds = dict(zip(feature_names, scaler.scale_))
    contributions = []
    for f in feature_names:
        val = features.get(f, np.nan)
        std = stds[f] if stds[f] > 0 else 1.0
        z = (val - means[f]) / std
        contributions.append((f, z * imp_map.get(f, 0.0)))
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    top5 = contributions[:5]

    st.html('<div class="section-heading">What drove your result</div><div class="section-sub">Your top inputs, weighted by how much this model relies on each one</div>')
    labels = [feature_label(f) for f, _ in reversed(top5)]
    vals = [v for _, v in reversed(top5)]
    # color by direction, not rank: purple = pulled the estimate down (negative),
    # yellow = pushed it up (positive) -- makes the color meaningful rather than decorative
    bar_colors = ["#FFD966" if v >= 0 else "#C77DFF" for v in vals]
    fig1, ax1 = plt.subplots(figsize=(6.4, 2.1))
    fig1.patch.set_facecolor("#0D0F26")
    ax1.set_facecolor("#0D0F26")
    ax1.barh(labels, vals, color=bar_colors)
    ax1.axvline(0, color="#8C8FC0", linewidth=0.8)
    ax1.tick_params(colors="#C2C4E8", labelsize=9)
    for spine in ax1.spines.values():
        spine.set_visible(False)
    ax1.set_xlabel("Pull on your estimate", color="#C2C4E8", fontsize=9)
    fig1.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)
    st.html('<div class="chart-caption">Bars extending left pulled your estimate down; bars extending right pushed it up. Longer bars = bigger effect.</div>')

    # --- Where you land in the full sample's distribution ---
    st.html('<div class="section-heading">Where you land</div><div class="section-sub">Your estimate against all 986 firms\' expected gains</div>')
    fig2, ax2 = plt.subplots(figsize=(6.4, 1.8))
    fig2.patch.set_facecolor("#0D0F26")
    ax2.set_facecolor("#0D0F26")
    ax2.hist(sample_df["ai_f2a2"], bins=20, color="#C77DFF", alpha=0.75, edgecolor="#0D0F26")
    ax2.axvline(pred, color="#FFD966", linewidth=2.2)
    ax2.annotate(
        f"You: {pred:.1f}%", xy=(pred, ax2.get_ylim()[1]), xytext=(5, -12),
        textcoords="offset points", color="#FFD966", fontsize=9, fontweight="bold",
    )
    ax2.tick_params(colors="#C2C4E8", labelsize=9)
    ax2.set_yticks([])
    for spine in ax2.spines.values():
        spine.set_visible(False)
    ax2.set_xlabel("Expected productivity increase (%)", color="#C2C4E8", fontsize=9)
    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)
    st.html('<div class="chart-caption">Each bar shows how many firms in the sample landed at that expected gain &mdash; your result is marked in yellow.</div>')

    # --- Peer comparison + single biggest driver ---
    user_breadth = len(ai_tech)
    peer_mask = (sample_df["ai_breadth_score"] - user_breadth).abs() <= 1
    peer_n = int(peer_mask.sum())
    if peer_n > 0:
        peer_mean = sample_df.loc[peer_mask, "ai_f2a2"].mean()
        vs_word = "above" if pred > peer_mean else "below" if pred < peer_mean else "in line with"
        peer_line = (
            f"Among the {peer_n} firms using a similar breadth of AI technology "
            f"({user_breadth} in use), the average expected gain is {peer_mean:.1f}% "
            f"&mdash; your estimate is {vs_word} that peer group."
        )
    else:
        peer_line = ""

    top_feature, top_contrib = top5[0]
    direction = "pushed it up" if top_contrib > 0 else "pulled it down"
    driver_line = (
        f"Of everything you entered, <strong>{feature_label(top_feature)}</strong> had the single "
        f"biggest pull on your result &mdash; it {direction} the most relative to a typical firm."
    )

    # --- Conclusion ---
    if idle_time_pct >= 20:
        next_step = (
            f"Your biggest lever is probably not AI itself: at {idle_time_pct}% idle production time, "
            "closing that gap would likely do more for realized productivity than any AI tool alone."
        )
    elif len(ai_tech) <= 1:
        next_step = (
            "Your AI usage is still narrow. Broadening into more of the technologies above "
            "&mdash; generative AI in particular &mdash; is the input most associated with higher estimates in this model."
        )
    elif more_hours_pct >= 20:
        next_step = (
            f"With {more_hours_pct}% of your workforce wanting more hours, redeploying that latent "
            "capacity toward AI-assisted tasks could compound your gains."
        )
    else:
        next_step = (
            "Your inputs are already fairly strong across the board &mdash; from here, gains likely "
            "come from deepening digitization maturity rather than broader AI adoption."
        )

    plain_terms = (
        f"The model estimates an expected productivity gain of about {pred:.1f}% for your firm, "
        f"{'above' if pred > SAMPLE_AVERAGE else 'below' if pred < SAMPLE_AVERAGE else 'in line with'} "
        f"the {SAMPLE_AVERAGE:.0f}% average across the surveyed firms."
        '<div class="insight-card-note">This estimate reflects firms\' self-reported expectations of '
        "AI-driven productivity gains, rather than measured productivity outcomes.</div>"
    )
    keep_in_mind = (
        f"This is built on firms' self-reported expectations, not measured outcomes, and typically "
        f"lands within &plusmn;{meta['mae']:.1f} percentage points of what the model would predict "
        f"for a firm like yours in the survey."
    )

    cards = [("In plain terms", plain_terms)]
    if peer_line:
        cards.append(("Compared to similar firms", peer_line))
    cards.append(("Biggest driver", driver_line))
    cards.append(("Next step", next_step))
    cards.append(("Keep in mind", keep_in_mind))

    for label, text in cards:
        st.html(f"""
        <div class="insight-card">
          <div class="insight-card-bar"></div>
          <strong>{label}:</strong> {text}
        </div>
        """)

    st.html(
        f"""<div class="footnote">
        Model: {meta['model_name']} &middot; Test R2 = {meta['r2']:.3f} &middot;
        RMSE = {meta['rmse']:.2f} &middot; MAE = {meta['mae']:.2f} &middot;
        trained on {meta['n_train']} firms, evaluated on {meta['n_test']} held-out firms.<br>
        The prediction reflects each firm's <em>self-reported expectation</em> of AI-driven
        productivity gains, not a measured outcome -- treat it as directional, not exact.
        </div>"""
    )
