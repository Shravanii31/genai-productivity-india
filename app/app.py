"""Streamlit wizard app: predicts a firm's expected AI-driven productivity increase."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
SAMPLE_AVERAGE = 11.0  # mean of ai_f2a2 in the modeling sample
SCALE_MAX = 35  # visual gauge scale, %

st.set_page_config(page_title="AI Productivity Estimator", page_icon="📈", layout="centered")

STEPS = ["Intro", "Firm Profile", "AI Usage", "Digitization", "Operations & Outlook", "Policy", "Result"]

# ---------------------------------------------------------------------------
# Model / preprocessing loading
# ---------------------------------------------------------------------------


@st.cache_resource
def load_artifacts():
    with open(MODELS_DIR / "final_model_meta.json") as fh:
        meta = json.load(fh)
    with open(ROOT / "data" / "processed" / "feature_list.json") as fh:
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
  --bg: #0F1324; --panel: #171C36; --panel-2: #1E2444;
  --border: #2A315A; --text: #F5F6FB; --text-dim: #B4BAD9; --text-faint: #7B82AC;
  --accent: #F0A83A; --accent-2: #33C7B0;
}
.stApp {
  background: radial-gradient(circle at 20% 0%, #1B2147 0%, var(--bg) 55%);
  color: var(--text);
  font-family: 'Inter', sans-serif;
}
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; color: var(--text) !important; }
p, span, label, div { color: var(--text); }
.block-container { max-width: 640px; padding-top: 2rem; }

/* progress dots */
.dots { display:flex; gap:8px; justify-content:center; margin-bottom: 1.6rem; }
.dot { width:8px; height:8px; border-radius:50%; background: var(--border); }
.dot.active { background: var(--accent); width:22px; border-radius:6px; }
.dot.done { background: var(--accent-2); }

.step-title { font-size: 1.5rem; font-weight: 600; margin-bottom: 0.15rem; }
.step-sub { color: var(--text-dim); font-size: 0.9rem; margin-bottom: 1.4rem; }

div[data-testid="stButton"] > button {
  background: var(--accent); color: #201202; border: none; border-radius: 10px;
  font-weight: 600; padding: 0.55rem 1.4rem;
}
div[data-testid="stButton"] > button:hover { background: #ffbb55; color:#201202; }
.back-btn button {
  background: transparent !important; color: var(--text-dim) !important;
  border: 1px solid var(--border) !important;
}

div[data-testid="stForm"], .panel {
  background: var(--panel); border: 1px solid var(--border); border-radius: 16px;
  padding: 1.4rem;
}

/* gauge */
.gauge-wrap { display:flex; justify-content:center; margin: 1rem 0; }
.gauge {
  width: 220px; height: 220px; border-radius: 50%;
  display:flex; align-items:center; justify-content:center;
}
.gauge-inner {
  width: 168px; height: 168px; border-radius: 50%; background: var(--panel);
  display:flex; flex-direction:column; align-items:center; justify-content:center;
}
.gauge-num { font-family:'Space Grotesk', sans-serif; font-size: 2.3rem; font-weight:700; color: var(--accent); }
.gauge-lbl { font-size: 0.75rem; color: var(--text-faint); }

.compare-row { margin-bottom: 0.9rem; }
.compare-label { display:flex; justify-content:space-between; font-size:0.85rem; color: var(--text-dim); margin-bottom:4px; }
.compare-track { background: var(--panel-2); border-radius: 8px; height: 10px; overflow:hidden; }
.compare-fill { height: 100%; border-radius: 8px; }

.footnote { color: var(--text-faint); font-size: 0.75rem; margin-top: 1.6rem; line-height:1.5; }
.insight { background: var(--panel-2); border-left: 3px solid var(--accent-2); border-radius: 8px; padding: 0.8rem 1rem; margin: 1rem 0; font-size: 0.92rem; }
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "step" not in st.session_state:
    st.session_state.step = 0
if "answers" not in st.session_state:
    st.session_state.answers = {}


def goto(i):
    st.session_state.step = i


def render_dots():
    content_steps = STEPS[1:]  # skip Intro in the dot bar
    current = st.session_state.step
    dots_html = '<div class="dots">'
    for i in range(1, len(STEPS)):
        cls = "dot"
        if i < current:
            cls += " done"
        elif i == current:
            cls += " active"
        dots_html += f'<div class="{cls}"></div>'
    dots_html += "</div>"
    if current > 0:
        st.markdown(dots_html, unsafe_allow_html=True)


def header(title, subtitle):
    st.markdown(f'<div class="step-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="step-sub">{subtitle}</div>', unsafe_allow_html=True)


def nav_buttons(back_to=None, next_label="Continue", on_next=None, next_disabled=False):
    # Navigation uses on_click callbacks (not "if st.button(...): ...; st.rerun()")
    # so the step change is applied before Streamlit's automatic post-click rerun,
    # rather than forcing a second, nested rerun mid-script.
    c1, c2 = st.columns([1, 1])
    with c1:
        if back_to is not None:
            st.markdown('<div class="back-btn">', unsafe_allow_html=True)
            st.button("Back", key=f"back_{st.session_state.step}", on_click=goto, args=(back_to,))
            st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.button(
            next_label, key=f"next_{st.session_state.step}", disabled=next_disabled,
            on_click=(on_next if on_next else lambda: None),
        )


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


def step_intro():
    st.markdown('<div class="step-title">AI Productivity Estimator</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="step-sub">Estimate your firm\'s expected productivity gain from AI, '
        "based on a survey of 986 Indian firms.</div>",
        unsafe_allow_html=True,
    )
    st.write("Six short steps: firm profile, AI usage, digitization, operations, and policy.")
    st.button("Start", on_click=goto, args=(1,))


def step_firm_profile():
    render_dots()
    header("Firm profile", "A few basics about your workforce")
    a = st.session_state.answers
    a["total_workers"] = st.number_input(
        "Total workers", min_value=1, max_value=100000,
        value=a.get("total_workers", 25), help="Total headcount at your firm",
    )
    a["pct_managers"] = st.slider("% managers", 0, 100, a.get("pct_managers", 10))
    a["pct_professionals"] = st.slider("% professionals", 0, 100, a.get("pct_professionals", 15))
    a["pct_clerical"] = st.slider("% clerical staff", 0, 100, a.get("pct_clerical", 10))
    a["pct_admin"] = st.slider(
        "% workers in admin roles", 0, 100, a.get("pct_admin", 10),
        help="Share of staff in administrative/back-office functions",
    )
    a["has_website"] = st.segmented_control(
        "Does your firm have a website?", ["No", "Yes"],
        default=a.get("has_website", "Yes"), key="has_website_sc",
    )
    nav_buttons(back_to=0, on_next=lambda: goto(2))


def step_ai_usage():
    render_dots()
    header("AI usage", "How is AI used at your firm today?")
    a = st.session_state.answers
    a["pct_computer_use"] = st.slider(
        "% of staff using computers", 0, 100, a.get("pct_computer_use", 40),
    )
    a["ai_tech"] = st.pills(
        "AI technologies in use",
        ["Machine learning", "Chatbots", "AI agents", "Generative AI", "Automation", "Autonomous systems"],
        selection_mode="multi", default=a.get("ai_tech", []),
        help="Select all that apply", key="ai_tech_pills",
    )
    a["ai_tasks"] = st.pills(
        "Tasks AI is used for",
        ["Finding info", "Summarizing", "Drafting", "Coding", "Translation", "Analysis", "Process control", "Customer interaction"],
        selection_mode="multi", default=a.get("ai_tasks", []),
        help="Select all that apply", key="ai_tasks_pills",
    )
    nav_buttons(back_to=1, on_next=lambda: goto(3))


def step_digitization():
    render_dots()
    header("Digitization maturity", "How advanced is each process, on a 0-7 scale?")
    a = st.session_state.answers
    a["finance_digitization"] = st.slider(
        "Finance / accounting digitization", 0, 7, a.get("finance_digitization", 2),
        help="0 = none, 7 = fully digitized",
    )
    a["supplychain_digitization"] = st.slider(
        "Supply chain digitization", 0, 7, a.get("supplychain_digitization", 2),
        help="0 = none, 7 = fully digitized",
    )
    a["customer_digitization"] = st.slider(
        "Customer-facing digitization", 0, 8, a.get("customer_digitization", 2),
        help="0 = none, 8 = fully digitized",
    )
    nav_buttons(back_to=2, on_next=lambda: goto(4))


def step_operations():
    render_dots()
    header("Operations & outlook", "Capacity, seasonality, and reskilling needs")
    a = st.session_state.answers
    a["idle_time_pct"] = st.slider("% of time production sits idle", 0, 100, a.get("idle_time_pct", 10))
    a["more_hours_pct"] = st.slider(
        "% of workers wanting more hours", 0, 100, a.get("more_hours_pct", 10),
    )
    a["can_absorb_shock"] = st.segmented_control(
        "Could you absorb a sudden demand increase?", ["No", "Yes"],
        default=a.get("can_absorb_shock", "Yes"), key="shock_sc",
    )
    c1, c2 = st.columns(2)
    with c1:
        a["peak_sales"] = st.number_input("Peak-season sales (₹L)", min_value=0.0, value=a.get("peak_sales", 10.0))
        a["peak_workers"] = st.number_input("Peak-season workers", min_value=0, value=a.get("peak_workers", 25))
    with c2:
        a["low_sales"] = st.number_input("Low-season sales (₹L)", min_value=0.1, value=a.get("low_sales", 8.0))
        a["low_workers"] = st.number_input("Low-season workers", min_value=1, value=a.get("low_workers", 22))
    a["reskilling_pct"] = st.slider(
        "% of workforce needing reskilling", 0, 100, a.get("reskilling_pct", 15),
    )
    nav_buttons(back_to=3, on_next=lambda: goto(5))


def step_policy():
    render_dots()
    header("Policy support", "Government / institutional AI support your firm uses")
    a = st.session_state.answers
    a["policy_support"] = st.pills(
        "Support programs used or aware of",
        ["Funding", "Workforce training", "Tax incentives", "R&D support", "Advisory services", "Graduate skills programs"],
        selection_mode="multi", default=a.get("policy_support", []),
        help="Select all that apply", key="policy_pills",
    )
    nav_buttons(back_to=4, next_label="See my result", on_next=lambda: goto(6))


def step_result():
    a = st.session_state.answers

    sales_ratio = min((a["peak_sales"] / max(a["low_sales"], 1e-6)), 20)
    worker_ratio = min((a["peak_workers"] / max(a["low_workers"], 1e-6)), 20)

    ai_tech = a.get("ai_tech", [])
    ai_tasks = a.get("ai_tasks", [])
    policy = a.get("policy_support", [])

    features = {
        "reskilling_need_share": a["reskilling_pct"] / 100,
        "workers_wanting_more_hours_share": a["more_hours_pct"] / 100,
        "idle_time_share": a["idle_time_pct"] / 100,
        "admin_worker_share": a["pct_admin"] / 100,
        "share_clerical": a["pct_clerical"] / 100,
        "sales_seasonality_ratio": sales_ratio,
        "policy_support_score": len(policy),
        "ai_b3a3_flag": int("Drafting" in ai_tasks),
        "share_professionals": a["pct_professionals"] / 100,
        "ai_b2a4_flag": int("Generative AI" in ai_tech),
        "share_managers": a["pct_managers"] / 100,
        "worker_seasonality_ratio": worker_ratio,
        "uses_genai": int("Generative AI" in ai_tech),
        "share_computer_use": a["pct_computer_use"] / 100,
        "log_workers": np.log1p(a["total_workers"]),
        "task_breadth": len(ai_tasks),
        "finance_digitization": a["finance_digitization"],
        "ai_g1b_flag": int("Workforce training" in policy),
        "customer_digitization": a["customer_digitization"],
        "ai_b3a2_flag": int("Summarizing" in ai_tasks),
        "can_absorb_demand_shock": int(a["can_absorb_shock"] == "Yes"),
        "supplychain_digitization": a["supplychain_digitization"],
        "ai_breadth_score": len(ai_tech),
        "ai_g1f_flag": int("Graduate skills programs" in policy),
        "has_website": int(a["has_website"] == "Yes"),
    }

    pred, meta = predict(features)

    header("Your estimate", "Expected AI-driven productivity increase")

    frac = min(pred / SCALE_MAX, 1.0)
    angle = frac * 360
    gauge_html = f"""
    <div class="gauge-wrap">
      <div class="gauge" style="background: conic-gradient(var(--accent) {angle}deg, var(--panel-2) {angle}deg);">
        <div class="gauge-inner">
          <div class="gauge-num">{pred:.1f}%</div>
          <div class="gauge-lbl">of {SCALE_MAX}% scale</div>
        </div>
      </div>
    </div>
    """
    st.markdown(gauge_html, unsafe_allow_html=True)

    you_pct = min(pred / SCALE_MAX, 1.0) * 100
    avg_pct = min(SAMPLE_AVERAGE / SCALE_MAX, 1.0) * 100
    compare_html = f"""
    <div class="compare-row">
      <div class="compare-label"><span>Your firm</span><span>{pred:.1f}%</span></div>
      <div class="compare-track"><div class="compare-fill" style="width:{you_pct}%; background: var(--accent);"></div></div>
    </div>
    <div class="compare-row">
      <div class="compare-label"><span>Sample average</span><span>{SAMPLE_AVERAGE:.1f}%</span></div>
      <div class="compare-track"><div class="compare-fill" style="width:{avg_pct}%; background: var(--accent-2);"></div></div>
    </div>
    """
    st.markdown(compare_html, unsafe_allow_html=True)

    if pred > SAMPLE_AVERAGE * 1.15:
        insight = "Your firm's AI usage and digital maturity point to above-average expected gains."
    elif pred < SAMPLE_AVERAGE * 0.85:
        insight = "Your firm's profile suggests below-average expected gains -- broader AI/digitization adoption may help."
    else:
        insight = "Your firm's expected gains are roughly in line with the sample average."
    st.markdown(f'<div class="insight">{insight}</div>', unsafe_allow_html=True)

    st.markdown(
        f"""<div class="footnote">
        Model: {meta['model_name']} &middot; Test R2 = {meta['r2']:.3f} &middot;
        RMSE = {meta['rmse']:.2f} &middot; MAE = {meta['mae']:.2f} &middot;
        trained on {meta['n_train']} firms, evaluated on {meta['n_test']} held-out firms.<br>
        The prediction reflects each firm's <em>self-reported expectation</em> of AI-driven
        productivity gains, not a measured outcome -- treat it as directional, not exact.
        </div>""",
        unsafe_allow_html=True,
    )

    nav_buttons(back_to=5, next_label="Start over", on_next=lambda: (st.session_state.answers.clear(), goto(0)))


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

STEP_FUNCS = [
    step_intro,
    step_firm_profile,
    step_ai_usage,
    step_digitization,
    step_operations,
    step_policy,
    step_result,
]

STEP_FUNCS[st.session_state.step]()
