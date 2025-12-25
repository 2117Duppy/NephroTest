# ckd_predictor_app.py
"""
NEPHROTEST — polished UI update (transparent buttons, paired zoomable projections,
centered KPIs, larger headings, plain-language result text).
Place this file next to ckd_model.pkl and run with: streamlit run ckd_predictor_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components
from datetime import datetime

# ------------- Page config & colors -------------
st.set_page_config(page_title="NEPHROTEST", layout="wide", page_icon="🩺")
COL_1 = "#2C7A7B"
COL_2 = "#319699"
COL_3 = "#1B5E63"
PALE = "#E0F2F1"
PALE2 = "#E3F2FD"
TEXT = "#000000"
RED = "#e15759"
BLUE = "#0f62fe"
GREEN = "#16a085"

# ------------- CSS (escape braces inside f-string with double {{ }}) -------------
CSS = f"""
<style>
/* page background and typography */
[data-testid="stAppViewContainer"] {{ background: {PALE2}; color: {TEXT}; font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial; }}
.project-title {{ text-align:center; margin-top:8px; margin-bottom:6px; }}
.project-name {{ font-size:40px; font-weight:900; color:{COL_3}; letter-spacing:1px; }}
.project-sub {{ font-size:24px; font-weight:700; color:{COL_2}; margin-bottom:12px; }}

/* single-column input card */
.form-card {{ background: white; border-radius:12px; padding:16px 18px; box-shadow:0 10px 30px rgba(15,23,42,0.06); max-width:780px; margin:auto; }}
.field-label {{ color:{TEXT}; font-weight:700; font-size:15px; margin-top:8px; margin-bottom:6px; }}
.input-widget {{ background: transparent; color:{TEXT}; font-size:14px; }}

/* centered transparent main button */
.big-center {{ display:flex; justify-content:center; margin-top:14px; margin-bottom:6px; }}
.stButton>button {{
  background: transparent !important;
  color: {COL_3} !important;
  border: 2px solid {COL_1} !important;
  border-radius:8px !important;
  padding:8px 16px !important;
  font-weight:700 !important;
  font-size:16px !important;
  box-shadow:none !important;
}}

/* download button styling */
.stDownloadButton>button {{
  background: transparent !important;
  color: {COL_3} !important;
  border: 2px solid {COL_1} !important;
  border-radius:8px !important;
  padding:8px 16px !important;
  font-weight:700 !important;
}}

/* results cards */
.card {{ background: rgba(255,255,255,0.0); border-radius:10px; padding:10px; margin-bottom:12px; max-width:1100px; margin-left:auto; margin-right:auto; }}
.banner-ok {{ border-left:6px solid {COL_2}; background: rgba(224,242,241,0.6); padding:12px; border-radius:8px; }}
.banner-bad {{ border-left:6px solid {RED}; background: rgba(255,235,238,0.6); padding:12px; border-radius:8px; }}
.headline {{ text-align:center; font-weight:800; color:{COL_3}; font-size:20px; margin-bottom:8px; }}
.subhead {{ text-align:center; font-weight:700; color:{COL_2}; font-size:16px; margin-bottom:6px; }}

/* KPI row centered and display flex */
.kpi-row {{ display:flex; justify-content:center; gap:18px; align-items:center; margin-bottom:8px; }}
.kpi {{ min-width:150px; text-align:center; padding:10px; border-radius:8px; background: rgba(255,255,255,0.0); }}
.kpi .val {{ font-size:18px; font-weight:800; color:{TEXT}; }}
.kpi .lbl {{ font-size:13px; color:#374151; }}

/* reveal animation */
.reveal {{ opacity:0; transform:translateY(18px); transition: all 650ms cubic-bezier(.2,.8,.2,1); }}
.reveal.visible {{ opacity:1; transform:translateY(0); }}

/* increase heading sizes */
.section-title {{ font-size:18px; font-weight:800; color:{COL_3}; text-align:center; margin-bottom:6px; }}

/* small footer */
.footer {{ text-align:center; color:#374151; font-size:12px; margin-top:8px; }}

/* literal brace escape (example) */
.css-fake {{ padding-top:4px; }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ------------- Load model safely -------------
MODEL_PATH = "ckd_model.pkl"
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    expected_features = list(model.feature_names_in_)
except FileNotFoundError:
    st.error(f"Model file '{MODEL_PATH}' not found. Put ckd_model.pkl here.")
    st.stop()
except Exception as e:
    st.error("Failed to load model.")
    st.exception(e)
    st.stop()

# ------------- Helpers -------------
INPUT_KEYS = [
    "age","bp","sg","al","su","rbc","pc","pcc","ba","bgr","bu","sc","sod","pot","hemo","pcv","wc","rc",
    "htn","dm","cad","appet","pe","ane"
]

def reset_inputs():
    for k in INPUT_KEYS:
        if k in st.session_state:
            del st.session_state[k]

def safe_rerun():
    if hasattr(st, "experimental_rerun"):
        try:
            st.experimental_rerun()
            return
        except Exception:
            pass
    try:
        q = dict(st.query_params)
        q["_ts"] = str(int(datetime.utcnow().timestamp()))
        st.query_params = q
    except Exception:
        pass

def build_features_from_state():
    missing = [k for k in INPUT_KEYS if k not in st.session_state]
    if missing:
        raise ValueError("Missing inputs: " + ", ".join(missing))
    s = st.session_state
    mapping = {
        "age": s["age"],
        "blood_pressure": s["bp"],
        "specific_gravity": s["sg"],
        "albumin": s["al"],
        "sugar": s["su"],
        "red_blood_cells": 1 if s["rbc"] == "normal" else 0,
        "pus_cell": 1 if s["pc"] == "normal" else 0,
        "pus_cell_clumps": 1 if s["pcc"] == "present" else 0,
        "bacteria": 1 if s["ba"] == "present" else 0,
        "blood_glucose_random": s["bgr"],
        "blood_urea": s["bu"],
        "serum_creatinine": s["sc"],
        "sodium": s["sod"],
        "potassium": s["pot"],
        "haemoglobin": s["hemo"],
        "packed_cell_volume": s["pcv"],
        "white_blood_cell_count": s["wc"],
        "red_blood_cell_count": s["rc"],
        "hypertension": 1 if s["htn"] == "yes" else 0,
        "diabetes_mellitus": 1 if s["dm"] == "yes" else 0,
        "coronary_artery_disease": 1 if s["cad"] == "yes" else 0,
        "appetite": 1 if s["appet"] == "good" else 0,
        "peda_edema": 1 if s["pe"] == "yes" else 0,
        "aanemia": 1 if s["ane"] == "yes" else 0,
    }
    missing_feat = [k for k in expected_features if k not in mapping]
    if missing_feat:
        raise ValueError("Feature mapping missing: " + ", ".join(missing_feat))
    row = [mapping[k] for k in expected_features]
    return pd.DataFrame([row], columns=expected_features)

# ------------- Session routing -------------
if "page" not in st.session_state:
    st.session_state.page = "input"
if "last_features" not in st.session_state:
    st.session_state.last_features = None
if "last_pred" not in st.session_state:
    st.session_state.last_pred = None
if "last_proba" not in st.session_state:
    st.session_state.last_proba = None
if "proj_csv" not in st.session_state:
    st.session_state.proj_csv = None

# ------------- INPUT PAGE (list-style) -------------
def input_page():
    st.markdown('<div class="project-title"><div class="project-name">NEPHROTEST</div><div class="project-sub">CKD Smart Detector Centre</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="form-card">', unsafe_allow_html=True)

    with st.form("input_form"):
        # single-column list-like inputs
        st.markdown('<div class="field-label">Age (years)</div>', unsafe_allow_html=True)
        st.number_input("", min_value=0, max_value=120, key="age", value=45)

        st.markdown('<div class="field-label">Blood Pressure (mm Hg)</div>', unsafe_allow_html=True)
        st.number_input("", key="bp", value=120)

        st.markdown('<div class="field-label">Specific Gravity</div>', unsafe_allow_html=True)
        st.selectbox("", [1.005,1.010,1.015,1.020,1.025], key="sg", index=2)

        st.markdown('<div class="field-label">Albumin (0-5)</div>', unsafe_allow_html=True)
        st.slider("", 0, 5, 0, key="al")

        st.markdown('<div class="field-label">Sugar (0-5)</div>', unsafe_allow_html=True)
        st.slider("", 0, 5, 0, key="su")

        st.markdown('<div class="field-label">Red Blood Cells</div>', unsafe_allow_html=True)
        st.selectbox("", ["normal","abnormal"], key="rbc")

        st.markdown('<div class="field-label">Pus Cell</div>', unsafe_allow_html=True)
        st.selectbox("", ["normal","abnormal"], key="pc")

        st.markdown('<div class="field-label">Pus Cell Clumps</div>', unsafe_allow_html=True)
        st.selectbox("", ["present","notpresent"], key="pcc")

        st.markdown('<div class="field-label">Bacteria</div>', unsafe_allow_html=True)
        st.selectbox("", ["present","notpresent"], key="ba")

        st.markdown('<div class="field-label">Blood Glucose Random (mg/dL)</div>', unsafe_allow_html=True)
        st.number_input("", key="bgr", value=100)

        st.markdown('<div class="field-label">Blood Urea (mg/dL)</div>', unsafe_allow_html=True)
        st.number_input("", key="bu", value=20)

        st.markdown('<div class="field-label">Serum Creatinine (mg/dL)</div>', unsafe_allow_html=True)
        st.number_input("", key="sc", value=1.0)

        st.markdown('<div class="field-label">Sodium (mEq/L)</div>', unsafe_allow_html=True)
        st.number_input("", key="sod", value=140)

        st.markdown('<div class="field-label">Potassium (mEq/L)</div>', unsafe_allow_html=True)
        st.number_input("", key="pot", value=4.2)

        st.markdown('<div class="field-label">Hemoglobin (g/dL)</div>', unsafe_allow_html=True)
        st.number_input("", key="hemo", value=14.0)

        st.markdown('<div class="field-label">Packed Cell Volume (%)</div>', unsafe_allow_html=True)
        st.number_input("", key="pcv", value=40)

        st.markdown('<div class="field-label">WBC (cells/cumm)</div>', unsafe_allow_html=True)
        st.number_input("", key="wc", value=7000)

        st.markdown('<div class="field-label">RBC (millions/cmm)</div>', unsafe_allow_html=True)
        st.number_input("", key="rc", value=4.5)

        st.markdown('<div class="field-label">Hypertension</div>', unsafe_allow_html=True)
        st.selectbox("", ["yes","no"], key="htn")

        st.markdown('<div class="field-label">Diabetes Mellitus</div>', unsafe_allow_html=True)
        st.selectbox("", ["yes","no"], key="dm")

        st.markdown('<div class="field-label">Coronary Artery Disease</div>', unsafe_allow_html=True)
        st.selectbox("", ["yes","no"], key="cad")

        st.markdown('<div class="field-label">Appetite</div>', unsafe_allow_html=True)
        st.selectbox("", ["good","poor"], key="appet")

        st.markdown('<div class="field-label">Pedal Edema</div>', unsafe_allow_html=True)
        st.selectbox("", ["yes","no"], key="pe")

        st.markdown('<div class="field-label">Anemia</div>', unsafe_allow_html=True)
        st.selectbox("", ["yes","no"], key="ane")

        # centered transparent analyze button
        st.markdown('<div class="big-center">', unsafe_allow_html=True)
        submitted = st.form_submit_button("ANALYSE")
        st.markdown('</div>', unsafe_allow_html=True)

        if submitted:
            try:
                features_df = build_features_from_state()
            except Exception as e:
                st.error("Input collection error: " + str(e))
                return

            try:
                pred = model.predict(features_df)
            except Exception as e:
                st.error("Prediction error: check model compatibility.")
                st.exception(e)
                return

            proba = None
            if hasattr(model, "predict_proba"):
                try:
                    pr = model.predict_proba(features_df)[0]
                    if len(pr) == 2:
                        proba = {"healthy_prob": float(pr[1])}
                except Exception:
                    proba = None

            st.session_state.last_features = features_df
            st.session_state.last_pred = int(pred[0])
            st.session_state.last_proba = proba
            st.session_state.page = "results"
            safe_rerun()
            return

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="footer">TEAM NEPHROTEST.</div>', unsafe_allow_html=True)

# ------------- RESULTS PAGE (one-click back + reset) -------------
def results_page():
    # one-click new patient
    if st.button("← New patient"):
        st.session_state.page = "input"
        st.session_state.last_features = None
        st.session_state.last_pred = None
        st.session_state.last_proba = None
        st.session_state.proj_csv = None
        reset_inputs()
        safe_rerun()
        return

    features_df = st.session_state.last_features
    pred = st.session_state.last_pred
    proba = st.session_state.last_proba

    if features_df is None or pred is None:
        st.error("No results to display. Return to input.")
        return

    # headline
    st.markdown('<div class="headline">Patient Analysis Result</div>', unsafe_allow_html=True)

    # banner (centered text + confidence)
    if pred == 0:
        conf_text = f"Confidence: {proba['healthy_prob']*100:.1f}% " if proba and "healthy_prob" in proba else ""
        st.markdown(f'<div class="banner-bad"><div style="display:flex;justify-content:space-between;align-items:center">'
                    f'<div><strong style="font-size:16px">🚨 Likely CKD</strong><div class="muted">Follow-up and confirmatory testing recommended.</div></div>'
                    f'<div style="text-align:right"><div style="font-weight:800;color:{COL_3}">{conf_text}</div>'
                    f'<div class="muted">Result ID: <span style="font-family:monospace">{pd.Timestamp.now().strftime("%Y%m%d%H%M%S")}</span></div></div></div></div>', unsafe_allow_html=True)
    else:
        conf_text = f"Confidence: {proba['healthy_prob']*100:.1f}% " if proba and "healthy_prob" in proba else ""
        st.markdown(f'<div class="banner-ok"><div style="display:flex;justify-content:space-between;align-items:center">'
                    f'<div><strong style="font-size:16px">✅ Low CKD Risk</strong><div class="muted">Routine monitoring recommended.</div></div>'
                    f'<div style="text-align:right"><div style="font-weight:800;color:{COL_3}">{conf_text}</div>'
                    f'<div class="muted">Result ID: <span style="font-family:monospace">{pd.Timestamp.now().strftime("%Y%m%d%H%M%S")}</span></div></div></div></div>', unsafe_allow_html=True)

    # Plain-language summary (transparent bg)
    simple_expl = []
    if pred == 0:
        simple_expl.append("The model flags this report as likely showing signs of Chronic Kidney Disease.")
        simple_expl.append("Recommendation: Please visit a clinician for confirmatory tests (e.g., repeat creatinine, eGFR calculation, urine albumin).")
    else:
        simple_expl.append("The model indicates low risk for CKD based on the provided lab values.")
        simple_expl.append("Recommendation: Continue routine monitoring and maintain healthy BP and blood glucose levels.")

    st.markdown('<div class="card reveal">', unsafe_allow_html=True)
    st.markdown('<div style="text-align:center;font-weight:700;color:'+COL_3+';font-size:16px">Summary (plain language)</div>', unsafe_allow_html=True)
    for s in simple_expl:
        st.markdown(f"<div style='text-align:center;color:{TEXT};'>{s}</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # KPI row (centered display flex)
    st.markdown('<div class="card reveal"><div class="kpi-row">', unsafe_allow_html=True)
    k_age = int(features_df.loc[0,"age"])
    k_bp = float(features_df.loc[0,"blood_pressure"])
    k_sc = float(features_df.loc[0,"serum_creatinine"])
    k_hemo = float(features_df.loc[0,"haemoglobin"])
    st.markdown(f'<div class="kpi"><div class="val">{k_age}</div><div class="lbl">Age</div></div>'
                f'<div class="kpi"><div class="val">{k_bp}</div><div class="lbl">Systolic BP</div></div>'
                f'<div class="kpi"><div class="val">{k_sc:.2f} mg/dL</div><div class="lbl">Serum Creatinine</div></div>'
                f'<div class="kpi"><div class="val">{k_hemo:.2f} g/dL</div><div class="lbl">Hemoglobin</div></div>', unsafe_allow_html=True)
    st.markdown('</div></div>', unsafe_allow_html=True)

    # Biomarker bar chart (transparent)
    st.markdown('<div class="card reveal">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Biomarker Overview</div>', unsafe_allow_html=True)
    numeric_features = ["blood_pressure","blood_glucose_random","blood_urea","serum_creatinine","haemoglobin"]
    numeric_values = [float(features_df.loc[0,f]) for f in numeric_features]
    df_bar = pd.DataFrame({"biomarker": numeric_features, "value": numeric_values})
    fig_bar = px.bar(df_bar, x="biomarker", y="value", text="value",
                     color="biomarker", color_discrete_sequence=[COL_1,COL_2,COL_3,RED,BLUE])
    fig_bar.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig_bar.update_layout(height=360, margin=dict(t=8,b=20,l=10,r=10),
                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
    fig_bar.update_yaxes(range=[0, max(numeric_values)*1.5 if numeric_values else 1])
    st.plotly_chart(fig_bar, use_container_width=True, config={'scrollZoom': False})
    st.markdown('</div>', unsafe_allow_html=True)

    # Radar chart (transparent)
    st.markdown('<div class="card reveal">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Radar — Patient vs Healthy</div>', unsafe_allow_html=True)
    categories = numeric_features
    patient_vals = numeric_values
    healthy_vals = [120,100,20,1.0,14.0]
    fig_r = go.Figure()
    fig_r.add_trace(go.Scatterpolar(r=patient_vals + [patient_vals[0]], theta=categories + [categories[0]], fill='toself', name='Patient', line_color=COL_2))
    fig_r.add_trace(go.Scatterpolar(r=healthy_vals + [healthy_vals[0]], theta=categories + [categories[0]], fill='toself', name='Healthy', line_color=COL_1, opacity=0.6))
    fig_r.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True, height=420,
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=10,b=10))
    st.plotly_chart(fig_r, use_container_width=True, config={'scrollZoom': False})
    st.markdown('</div>', unsafe_allow_html=True)

    # Projections: show biomarker graphs paired (2 per row), colored lines (red/blue/green)
    st.markdown('<div class="card reveal">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Future Patterns — Projections</div>', unsafe_allow_html=True)
    months = st.slider("Projection horizon (months)", 1, 24, 6, key="proj_h")
    scenarios = st.multiselect("Scenarios to show", ["Best","Likely","Worst"], default=["Likely","Worst"])

    per_month_changes = {
        "Best": {"blood_pressure": -0.005, "blood_glucose_random": -0.01, "blood_urea": -0.01, "serum_creatinine": -0.005, "haemoglobin": +0.005},
        "Likely": {"blood_pressure": +0.002, "blood_glucose_random": +0.005, "blood_urea": +0.01, "serum_creatinine": +0.01, "haemoglobin": -0.002},
        "Worst": {"blood_pressure": +0.01, "blood_glucose_random": +0.02, "blood_urea": +0.03, "serum_creatinine": +0.04, "haemoglobin": -0.01}
    }

    base = {feat: float(features_df.loc[0, feat]) for feat in numeric_features}

    if scenarios:
        # prepare a per-biomarker projection DataFrame for each scenario
        proj_frames = []
        for scen in scenarios:
            ch = per_month_changes[scen]
            months_idx = list(range(0, months+1))
            data = {"month": months_idx}
            for feat in numeric_features:
                vals = []
                cur = base[feat]
                for m in months_idx:
                    if m == 0:
                        vals.append(cur)
                    else:
                        cur = cur * (1 + ch[feat])
                        vals.append(cur)
                data[feat] = vals
            dfp = pd.DataFrame(data)
            dfp["scenario"] = scen
            proj_frames.append(dfp)
        proj_all = pd.concat(proj_frames, ignore_index=True)

        # display pairs of biomarker plots 2-per-row
        pairs = []
        for i in range(0, len(numeric_features), 2):
            if i + 1 < len(numeric_features):
                pairs.append((numeric_features[i], numeric_features[i + 1]))
            else:
                pairs.append((numeric_features[i], None))

        for a, b in pairs:
            c1, c2 = st.columns(2, gap="small")
            with c1:
                df_plot = proj_all.melt(id_vars=["month","scenario"], value_vars=[a], var_name="biomarker", value_name="value")
                fig_a = px.line(df_plot, x="month", y="value", color="scenario", markers=True,
                                color_discrete_sequence=[RED, BLUE, GREEN])
                fig_a.update_layout(title=a.replace("_"," ").title(), height=320,
                                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                    title_font_color=TEXT)
                fig_a.update_traces(mode="lines+markers")
                st.plotly_chart(fig_a, use_container_width=True, config={'scrollZoom': False})
            with c2:
                # b might be out of range if odd count; handle gracefully
                if b in proj_all.columns or any(b == col for col in proj_all.columns):
                    df_plot2 = proj_all.melt(id_vars=["month","scenario"], value_vars=[b], var_name="biomarker", value_name="value")
                    fig_b = px.line(df_plot2, x="month", y="value", color="scenario", markers=True,
                                    color_discrete_sequence=[RED, BLUE, GREEN])
                    fig_b.update_layout(title=b.replace("_"," ").title(), height=320,
                                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                        title_font_color=TEXT)
                    fig_b.update_traces(mode="lines+markers")
                    st.plotly_chart(fig_b, use_container_width=True, config={'scrollZoom': False})
                else:
                    st.markdown("", unsafe_allow_html=True)

        # download CSV (transparent)
        st.download_button("Download projection CSV", data=proj_all.to_csv(index=False),
                           file_name="nephrotest_projection.csv", mime="text/csv")
        st.session_state.proj_csv = proj_all.to_csv(index=False)
    else:
        st.info("Pick scenarios to display projections.")
    st.markdown('</div>', unsafe_allow_html=True)


    # JS for fade-in reveal
    components.html("""
    <script>
    const cb = (entries, observer) => { entries.forEach(e => { if (e.isIntersecting) e.target.classList.add('visible'); }); };
    const obs = new IntersectionObserver(cb, {threshold:0.15});
    document.querySelectorAll('.reveal').forEach(el => obs.observe(el));
    </script>
    """, height=0)

# ------------- Router -------------
if st.session_state.page == "input":
    input_page()
else:
    results_page()
