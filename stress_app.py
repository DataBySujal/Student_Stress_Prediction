from pathlib import Path

import pandas as pd
import streamlit as st  # pyright: ignore[reportMissingImports]
from xgboost import XGBClassifier  # pyright: ignore[reportMissingImports]


MODEL_PATH = Path(__file__).resolve().parent / "student_stress_xgb.json"
FEATURES = [
    "Student_Type",
    "Sleep_Hours",
    "Study_Hours",
    "Social_Media_Hours",
    "Attendance",
    "Exam_Pressure",
    "Family_Support",
    "Month",
]
STUDENT_TYPE_CODES = {
    "School student": 0,
    "College student": 1,
    "Working student": 2,
}


st.set_page_config(
    page_title="StressSense | Student Stress Predictor",
    page_icon="🧠",
    layout="wide",
)


@st.cache_resource
def load_model() -> XGBClassifier:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH.name}")
    model = XGBClassifier()
    model.load_model(MODEL_PATH)
    return model


def add_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@700;800&display=swap');

        :root {
            --ink: #172d29;
            --muted: #6f817c;
            --green: #146b55;
            --green-dark: #0e4f40;
            --line: #dde9e5;
            --cream: #f7faf8;
        }

        html, body, [class*="css"] {
            font-family: "DM Sans", sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at 94% 3%, rgba(196, 229, 215, .75), transparent 24rem),
                linear-gradient(135deg, #fbfdfc 0%, #f1f6f3 100%);
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        .block-container {
            max-width: 1240px;
            padding-top: 1.4rem;
            padding-bottom: 2rem;
        }

        .topbar {
            height: 58px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 22px;
            border-bottom: 1px solid rgba(194, 210, 204, .65);
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 11px;
            color: var(--ink);
            font-family: "Manrope", sans-serif;
            font-size: 20px;
            font-weight: 800;
        }

        .brand-mark {
            width: 38px;
            height: 38px;
            display: grid;
            place-items: center;
            border-radius: 12px;
            background: var(--green);
            color: white;
            font-size: 19px;
            box-shadow: 0 8px 20px rgba(20, 107, 85, .17);
        }

        .status {
            display: inline-flex;
            align-items: center;
            gap: 7px;
            color: #63766f;
            font-size: 12px;
            font-weight: 700;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #4eaa83;
            box-shadow: 0 0 0 4px rgba(78, 170, 131, .12);
        }

        .hero {
            display: grid;
            grid-template-columns: 1fr auto;
            align-items: end;
            gap: 24px;
            margin: 4px 0 22px;
        }

        .eyebrow {
            color: var(--green);
            font-size: 11px;
            font-weight: 800;
            letter-spacing: 1.5px;
            text-transform: uppercase;
        }

        .hero-title {
            margin: 7px 0 7px;
            color: var(--ink);
            font-family: "Manrope", sans-serif;
            font-size: clamp(34px, 4vw, 48px);
            line-height: 1.05;
            letter-spacing: -1.8px;
        }

        .hero-copy {
            max-width: 680px;
            margin: 0;
            color: var(--muted);
            font-size: 15px;
            line-height: 1.55;
        }

        .privacy-chip {
            display: flex;
            align-items: center;
            gap: 9px;
            margin-bottom: 4px;
            padding: 9px 12px;
            border: 1px solid var(--line);
            border-radius: 999px;
            background: rgba(255, 255, 255, .72);
            color: #657973;
            font-size: 12px;
            font-weight: 700;
        }

        [data-testid="stForm"] {
            border: 1px solid var(--line);
            border-radius: 18px;
            background: rgba(255, 255, 255, .94);
            box-shadow: 0 14px 34px rgba(43, 86, 72, .07);
            padding: 24px;
        }

        .section-heading {
            display: flex;
            align-items: center;
            gap: 8px;
            margin: 1px 0 12px;
            color: #73877f;
            font-size: 10px;
            font-weight: 800;
            letter-spacing: 1.25px;
            text-transform: uppercase;
        }

        .section-heading:after {
            content: "";
            height: 1px;
            flex: 1;
            background: #edf2ef;
        }

        [data-testid="stWidgetLabel"] p,
        [data-testid="stWidgetLabel"] span,
        [data-testid="stWidgetLabel"] label,
        [data-testid="stNumberInput"] label,
        [data-testid="stSelectbox"] label,
        [data-testid="stSlider"] label {
            color: #2c5047 !important;
            font-weight: 700 !important;
            opacity: 1 !important;
        }

        [data-testid="stSlider"] [data-testid="stThumbValue"] {
            color: var(--green) !important;
            font-weight: 800;
        }

        [data-testid="stFormSubmitButton"] button {
            width: 100%;
            min-height: 48px;
            border: 0;
            border-radius: 11px;
            background: linear-gradient(135deg, #196e59, #124d40);
            color: white;
            font-weight: 800;
            box-shadow: 0 10px 24px rgba(20, 107, 85, .16);
        }

        [data-testid="stFormSubmitButton"] button:hover {
            border: 0;
            background: var(--green-dark);
            color: white;
        }

        .result-panel {
            min-height: 470px;
            border-radius: 18px;
            padding: 26px;
            color: white;
            background:
                radial-gradient(circle at 100% 0%, rgba(163, 211, 191, .25), transparent 16rem),
                linear-gradient(145deg, #174f43 0%, #0f382f 100%);
            box-shadow: 0 18px 38px rgba(25, 79, 65, .15);
        }

        .result-panel.empty-panel {
            display: grid;
            place-items: center;
            text-align: center;
            background:
                linear-gradient(rgba(255,255,255,.02), rgba(255,255,255,.02)),
                linear-gradient(145deg, #1d5749 0%, #12382f 100%);
        }

        .result-kicker {
            color: #afd8c8;
            font-size: 10px;
            font-weight: 800;
            letter-spacing: 1.35px;
            text-transform: uppercase;
        }

        .gauge-wrap {
            display: grid;
            place-items: center;
            margin: 18px 0 12px;
        }

        .gauge {
            position: relative;
            width: 164px;
            height: 164px;
            display: grid;
            place-items: center;
            border-radius: 50%;
        }

        .gauge:before {
            content: "";
            position: absolute;
            inset: 13px;
            border-radius: 50%;
            background: #16483d;
            box-shadow: inset 0 0 20px rgba(0,0,0,.09);
        }

        .gauge-content {
            position: relative;
            z-index: 1;
            text-align: center;
        }

        .gauge-value {
            display: block;
            font-family: "Manrope", sans-serif;
            font-size: 34px;
            line-height: 1;
        }

        .gauge-label {
            display: block;
            margin-top: 5px;
            color: #badacf;
            font-size: 10px;
            font-weight: 700;
            text-transform: uppercase;
        }

        .result-name {
            margin: 8px 0 5px;
            text-align: center;
            font-family: "Manrope", sans-serif;
            font-size: 24px;
            letter-spacing: -.5px;
        }

        .result-advice {
            max-width: 400px;
            min-height: 48px;
            margin: 0 auto;
            color: #d2e8df;
            text-align: center;
            font-size: 12.5px;
            line-height: 1.5;
        }

        .result-metrics {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 8px;
            margin-top: 20px;
            padding-top: 16px;
            border-top: 1px solid rgba(255,255,255,.12);
        }

        .result-metric {
            text-align: center;
        }

        .result-metric span {
            display: block;
            color: #a9cbbf;
            font-size: 9px;
            font-weight: 700;
            text-transform: uppercase;
        }

        .result-metric strong {
            display: block;
            margin-top: 4px;
            color: white;
            font-size: 14px;
        }

        .empty-icon {
            width: 78px;
            height: 78px;
            display: grid;
            place-items: center;
            margin: 0 auto 16px;
            border: 1px solid rgba(255,255,255,.16);
            border-radius: 50%;
            background: rgba(255,255,255,.05);
            font-size: 34px;
        }

        .empty-title {
            font-family: "Manrope", sans-serif;
            font-size: 20px;
            margin-bottom: 7px;
        }

        .empty-copy {
            max-width: 280px;
            color: #bdd7cd;
            font-size: 13px;
            line-height: 1.55;
        }

        .info-row {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 12px;
            margin-top: 12px;
        }

        .info-card {
            padding: 13px 15px;
            border: 1px solid var(--line);
            border-radius: 12px;
            background: rgba(255,255,255,.68);
        }

        .info-card span {
            color: #82958f;
            font-size: 10px;
            font-weight: 700;
            text-transform: uppercase;
        }

        .info-card strong {
            display: block;
            margin-top: 3px;
            color: #294d44;
            font-size: 14px;
        }

        [data-testid="stCaptionContainer"] {
            color: #85958f;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def risk_message(probability: float) -> tuple[str, str, str]:
    if probability >= 0.70:
        return (
            "Higher predicted stress",
            "Several lifestyle signals align with the higher-stress pattern. Consider rest, workload adjustments, and support.",
            "#d98b79",
        )
    if probability >= 0.40:
        return (
            "Moderate predicted stress",
            "The model sees a mixed pattern. Small changes to rest, workload, and support may help.",
            "#e1b768",
        )
    return (
        "Lower predicted stress",
        "The model did not identify a strong high-stress pattern. Keep maintaining your supportive routines.",
        "#70bda1",
    )


def main() -> None:
    add_styles()

    st.markdown(
        """
        <div class="topbar">
          <div class="brand"><div class="brand-mark">🧠</div>StressSense</div>
          <div class="status"><span class="status-dot"></span>Model ready</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="hero">
          <div>
            <div class="eyebrow">Personal wellbeing assessment</div>
            <div class="hero-title">Understand your study-stress pattern.</div>
            <p class="hero-copy">Enter your recent lifestyle and study details for an instant, data-informed stress prediction.</p>
          </div>
          <div class="privacy-chip">🔒 Private &amp; on-device</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    form_column, result_column = st.columns([1.55, 1], gap="large")

    with form_column:
        with st.form("stress_prediction_form"):
            st.markdown('<div class="section-heading">Student profile</div>', unsafe_allow_html=True)
            profile_col, month_col = st.columns(2)
            with profile_col:
                student_type = st.selectbox("Student type", list(STUDENT_TYPE_CODES))
            with month_col:
                month = st.selectbox(
                    "Current month",
                    list(range(1, 13)),
                    index=0,
                    format_func=lambda value: pd.Timestamp(2026, value, 1).strftime("%B"),
                )

            st.markdown('<div class="section-heading">Daily routine</div>', unsafe_allow_html=True)
            routine_left, routine_right = st.columns(2)
            with routine_left:
                sleep_hours = st.number_input(
                    "Sleep hours / day", min_value=0.0, max_value=24.0,
                    value=7.0, step=0.25,
                )
                social_media_hours = st.number_input(
                    "Social media / day", min_value=0.0, max_value=24.0,
                    value=3.0, step=0.25,
                )
            with routine_right:
                study_hours = st.number_input(
                    "Study hours / day", min_value=0.0, max_value=24.0,
                    value=4.0, step=0.25,
                )
                attendance = st.number_input(
                    "Attendance (%)", min_value=0.0, max_value=100.0,
                    value=80.0, step=1.0,
                )

            st.markdown('<div class="section-heading">Support &amp; pressure</div>', unsafe_allow_html=True)
            scale_left, scale_right = st.columns(2)
            with scale_left:
                exam_pressure = st.slider(
                    "Exam pressure", 1, 10, 5,
                    help="1 = very low pressure, 10 = very high pressure",
                )
            with scale_right:
                family_support = st.slider(
                    "Family support", 1, 10, 6,
                    help="1 = very low support, 10 = very strong support",
                )

            submitted = st.form_submit_button("Analyze stress pattern")

        st.markdown(
            f"""
            <div class="info-row">
              <div class="info-card"><span>Sleep balance</span><strong>{sleep_hours:.1f} hrs</strong></div>
              <div class="info-card"><span>Study load</span><strong>{study_hours:.1f} hrs</strong></div>
              <div class="info-card"><span>Attendance</span><strong>{attendance:.0f}%</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with result_column:
        if submitted:
            try:
                values = [[
                    STUDENT_TYPE_CODES[student_type],
                    sleep_hours,
                    study_hours,
                    social_media_hours,
                    attendance,
                    exam_pressure,
                    family_support,
                    month,
                ]]
                inputs = pd.DataFrame(values, columns=FEATURES)
                probability = float(load_model().predict_proba(inputs)[0][1])
                label, advice, gauge_color = risk_message(probability)
                confidence = max(probability, 1 - probability)
                gauge_degrees = probability * 360

                st.markdown(
                    f"""
                    <div class="result-panel">
                      <div class="result-kicker">Your assessment</div>
                      <div class="gauge-wrap">
                        <div class="gauge" style="background: conic-gradient({gauge_color} 0deg {gauge_degrees:.0f}deg, rgba(255,255,255,.10) {gauge_degrees:.0f}deg 360deg);">
                          <div class="gauge-content">
                            <span class="gauge-value">{probability:.0%}</span>
                            <span class="gauge-label">Risk signal</span>
                          </div>
                        </div>
                      </div>
                      <div class="result-name">{label}</div>
                      <div class="result-advice">{advice}</div>
                      <div class="result-metrics">
                        <div class="result-metric"><span>Confidence</span><strong>{confidence:.0%}</strong></div>
                        <div class="result-metric"><span>Profile</span><strong>{student_type.split()[0]}</strong></div>
                        <div class="result-metric"><span>Month</span><strong>{pd.Timestamp(2026, month, 1).strftime('%b')}</strong></div>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            except Exception as error:
                st.error(f"The model could not generate a prediction: {error}")
        else:
            st.markdown(
                """
                <div class="result-panel empty-panel">
                  <div>
                    <div class="empty-icon">✦</div>
                    <div class="empty-title">Ready for your assessment</div>
                    <div class="empty-copy">Complete the form, then select <strong>Analyze stress pattern</strong> to view your prediction.</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.caption(
        "Educational use only. This result is a model estimate, not a medical or mental-health diagnosis. "
        "For persistent distress or immediate safety concerns, contact a qualified professional or local support service."
    )


if __name__ == "__main__":
    main()
