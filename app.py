
from pathlib import Path
import os
import sys

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv


# =========================================================
# 1. PATHS AND CONFIGURATION
# =========================================================

BASE = Path(__file__).resolve().parent

load_dotenv(BASE / ".env")

DATA_PATH = BASE / "cleaned_leads.csv"
AI_DIR = BASE / "Member3_AI_Recommendation"

PIPELINE_PATH = BASE / "lead_model.joblib"
MODEL_PATH = BASE / "best_lead_model.pkl"
PREPROCESSOR_PATH = BASE / "preprocessor.pkl"

st.set_page_config(
    page_title="LeadPulse | Sales Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# 2. DASHBOARD STYLING
# =========================================================

st.markdown("""
<style>

:root {
    --ink: #f8fafc;
    --muted: #cbd5e1;
    --line: #475569;
}

.block-container {
    padding-top: 1.25rem;
    padding-bottom: 2rem;
    max-width: 1450px;
}

[data-testid="stSidebar"] {
    border-right: 1px solid #475569;
}

[data-testid="stMetric"] {
    background: #1e293b;
    border: 1px solid #475569;
    border-radius: 14px;
    padding: 14px 16px;
}

[data-testid="stMetricLabel"],
[data-testid="stMetricValue"] {
    color: #f8fafc !important;
}

[data-testid="stForm"] {
    background-color: #1e293b !important;
    border: 1px solid #475569 !important;
    border-radius: 15px !important;
    padding: 25px !important;
}

[data-testid="stForm"] [data-testid="stWidgetLabel"],
[data-testid="stForm"] [data-testid="stWidgetLabel"] p,
[data-testid="stForm"] label,
[data-testid="stForm"] p {
    color: #f8fafc !important;
    font-weight: 600 !important;
}

[data-testid="stForm"] input {
    background-color: #334155 !important;
    color: #ffffff !important;
    border-color: #64748b !important;
}

[data-testid="stForm"] [data-baseweb="select"] > div {
    background-color: #334155 !important;
    border-color: #64748b !important;
}

[data-testid="stForm"] [data-baseweb="select"] span {
    color: #ffffff !important;
}

[data-testid="stForm"] button {
    background-color: #16a34a !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: bold !important;
}

h1, h2, h3 {
    color: #f8fafc;
}

.small-muted {
    color: #cbd5e1;
    font-size: 0.85rem;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. LOAD DATASET
# =========================================================

@st.cache_data
def load_data():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Place cleaned_leads.csv in the same folder as app.py."
        )

    data = pd.read_csv(DATA_PATH)

    if "Converted" not in data.columns:
        raise ValueError(
            "Dataset must contain the Converted target column."
        )

    data["Converted"] = pd.to_numeric(
        data["Converted"],
        errors="coerce"
    )

    data = data.dropna(subset=["Converted"]).copy()
    data["Converted"] = data["Converted"].astype(int)

    return data


# =========================================================
# 4. LOAD MACHINE LEARNING MODEL
# =========================================================

def get_predictor():

    # Option 1: Complete pipeline
    if PIPELINE_PATH.exists():

        predictor = joblib.load(PIPELINE_PATH)

        if not hasattr(predictor, "classes_"):
            raise ValueError(
                "The complete pipeline does not expose classes_. "
                "Please verify the model artifact."
            )

        return predictor, "Complete pipeline"

    # Option 2: Separate model and preprocessor
    if MODEL_PATH.exists() and PREPROCESSOR_PATH.exists():

        model = joblib.load(MODEL_PATH)
        prep = joblib.load(PREPROCESSOR_PATH)

        class Predictor:

            def __init__(self, model, prep):
                self.model = model
                self.prep = prep
                self.classes_ = model.classes_

            def predict_proba(self, frame):
                transformed = self.prep.transform(frame)
                return self.model.predict_proba(transformed)

            def predict(self, frame):
                transformed = self.prep.transform(frame)
                return self.model.predict(transformed)

        return Predictor(model, prep), "Separate model + preprocessor"

    return None, "Model files not found"


# =========================================================
# 5. CONVERSION PROBABILITY
# =========================================================

def get_conversion_probability(predictor, frame):

    classes = list(predictor.classes_)

    if 1 not in classes:
        raise ValueError(
            "The model does not contain class 1 (Converted)."
        )

    positive_index = classes.index(1)

    probabilities = predictor.predict_proba(frame)

    return float(probabilities[0][positive_index])


# =========================================================
# 6. LEAD PRIORITY CLASSIFICATION
# =========================================================

def bucket(probability):

    if probability >= 0.70:
        return "High"

    elif probability >= 0.40:
        return "Medium"

    else:
        return "Low"


# =========================================================
# 7. AI MODULE IMPORT
# =========================================================

def ai_functions():

    if str(AI_DIR) not in sys.path:
        sys.path.insert(0, str(AI_DIR))

    try:

        from main_ai import run_lead_ai, ask_about_lead

        return run_lead_ai, ask_about_lead, None

    except Exception as e:

        return None, None, str(e)


# =========================================================
# 8. COMMON HEADER
# =========================================================

def header(title, subtitle):

    st.title(title)
    st.caption(subtitle)


# =========================================================
# 9. INITIALIZE DATA
# =========================================================

try:

    df = load_data()

except Exception as e:

    st.error(str(e))
    st.stop()


features = df.drop(columns=["Converted"])

total = len(df)

converted = int(df["Converted"].sum())

conversion_rate = converted / total if total else 0


# =========================================================
# 10. SIDEBAR NAVIGATION
# =========================================================

nav = [
    "Overview",
    "Lead Explorer",
    "Predict a Lead",
    "AI Sales Assistant",
    "Model Insights"
]

with st.sidebar:

    st.markdown("## 📈 LeadPulse")

    st.caption("AI SALES INTELLIGENCE")

    st.divider()

    page = st.radio(
        "WORKSPACE",
        nav,
        label_visibility="visible"
    )

    st.divider()

    st.caption(f"Dataset · {total:,} leads")

    st.caption("Target · Converted = 1")

    st.caption("Priority bands · High ≥70% · Medium ≥40%")

    st.divider()

    st.caption(
        "AI-Based Lead Conversion Prediction"
    )


# =========================================================
# 11. OVERVIEW DASHBOARD
# =========================================================

if page == "Overview":

    header(
        "Business overview",
        "A focused view of historical lead performance and conversion outcomes."
    )

    a, b, c, d = st.columns(4)

    a.metric("Total leads", f"{total:,}")

    b.metric("Converted leads", f"{converted:,}")

    c.metric("Not converted", f"{total-converted:,}")

    d.metric("Historical conversion rate", f"{conversion_rate:.1%}")

    st.markdown(" ")

    left, right = st.columns([1, 1.35], gap="large")

    with left:

        st.subheader("Conversion outcome")

        outcome = pd.DataFrame({
            "Outcome": ["Converted", "Not converted"],
            "Leads": [converted, total-converted]
        })

        fig = px.pie(
            outcome,
            names="Outcome",
            values="Leads",
            hole=.68,
            color="Outcome",
            color_discrete_map={
                "Converted": "#16a36a",
                "Not converted": "#94a3b8"
            }
        )

        fig.update_layout(
            margin=dict(t=15, b=10, l=5, r=5),
            legend_title_text=""
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:

        st.subheader("Conversion by lead source")

        if "Lead Source" in df:

            src = df.groupby(
                "Lead Source",
                dropna=False
            )["Converted"].agg(
                Leads="size",
                Converted="sum",
                Rate="mean"
            ).reset_index()

            src["Conversion rate"] = src["Rate"] * 100

            src = src.sort_values(
                "Leads",
                ascending=False
            ).head(10)

            fig = px.bar(
                src,
                x="Lead Source",
                y="Conversion rate",
                hover_data=["Leads", "Converted"],
                color="Conversion rate",
                color_continuous_scale="Blues"
            )

            fig.update_layout(
                coloraxis_showscale=False,
                margin=dict(t=10, b=5),
                yaxis_title="Conversion rate (%)"
            )

            fig.update_xaxes(tickangle=-25)

            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Lead activity")

    c1, c2 = st.columns(2)

    if "Total Time Spent on Website" in df:

        with c1:

            fig = px.histogram(
                df,
                x="Total Time Spent on Website",
                color=df["Converted"].map({
                    0: "Not converted",
                    1: "Converted"
                }),
                nbins=30,
                barmode="overlay",
                opacity=.75,
                labels={"color": "Outcome"},
                title="Website time by outcome"
            )

            fig.update_layout(
                legend_title_text="",
                margin=dict(t=35, b=5)
            )

            st.plotly_chart(fig, use_container_width=True)

    if "TotalVisits" in df:

        with c2:

            fig = px.box(
                df,
                x="Converted",
                y="TotalVisits",
                color="Converted",
                title="Website visits by outcome",
                labels={"Converted": "Converted (0/1)"}
            )

            fig.update_layout(
                showlegend=False,
                margin=dict(t=35, b=5)
            )

            st.plotly_chart(fig, use_container_width=True)

    st.info(
        "The supplied dataset has no date/time field, so a genuine "
        "month-wise conversion trend cannot be calculated. "
        "The overview therefore focuses on source performance "
        "and available lead activity."
    )


# =========================================================
# 12. LEAD EXPLORER
# =========================================================

elif page == "Lead Explorer":

    header(
        "Lead explorer",
        "Filter historical records, inspect patterns and export the selected leads."
    )

    f1, f2, f3 = st.columns(3)

    view = df.copy()

    if "Lead Source" in view:

        options = sorted(
            view["Lead Source"].fillna("Unknown").astype(str).unique()
        )

        source = f1.multiselect(
            "Lead source",
            options,
            default=[]
        )

        if source:

            view = view[
                view["Lead Source"].fillna("Unknown").astype(str).isin(source)
            ]

    if "Lead Origin" in view:

        origins = sorted(
            view["Lead Origin"].fillna("Unknown").astype(str).unique()
        )

        origin = f2.multiselect(
            "Lead origin",
            origins,
            default=[]
        )

        if origin:

            view = view[
                view["Lead Origin"].fillna("Unknown").astype(str).isin(origin)
            ]

    outcome = f3.selectbox(
        "Actual outcome",
        ["All", "Converted", "Not converted"]
    )

    if outcome == "Converted":

        view = view[view["Converted"] == 1]

    elif outcome == "Not converted":

        view = view[view["Converted"] == 0]

    k1, k2, k3 = st.columns(3)

    k1.metric("Matching leads", f"{len(view):,}")

    k2.metric(
        "Converted",
        f"{int(view['Converted'].sum()):,}"
    )

    k3.metric(
        "Conversion rate",
        f"{view['Converted'].mean():.1%}" if len(view) else "—"
    )

    st.subheader("Conversion by selected segment")

    if len(view) and "Lead Source" in view:

        grp = view.groupby(
            "Lead Source"
        )["Converted"].agg(
            ["count", "mean"]
        ).reset_index()

        grp["Conversion rate (%)"] = grp["mean"] * 100

        fig = px.bar(
            grp,
            x="Lead Source",
            y="Conversion rate (%)",
            hover_data=["count"]
        )

        fig.update_layout(margin=dict(t=10, b=5))

        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Lead records")

    cols = [
        c for c in [
            "Lead Source",
            "Lead Origin",
            "TotalVisits",
            "Total Time Spent on Website",
            "Page Views Per Visit",
            "Last Activity",
            "Converted"
        ]
        if c in view.columns
    ]

    st.dataframe(
        view[cols],
        use_container_width=True,
        hide_index=True,
        height=360
    )

    st.download_button(
        "⬇ Download filtered records",
        view.to_csv(index=False).encode(),
        "leadpulse_filtered_leads.csv",
        "text/csv"
    )


# =========================================================
# 13. PREDICT A LEAD
# =========================================================

elif page == "Predict a Lead":

    header(
        "Predict a lead",
        "Enter the available lead attributes to estimate conversion likelihood."
    )

    predictor, source = get_predictor()

    if predictor is None:

        st.warning(
            "The trained model artifact is not available."
        )

        st.markdown("""
**To enable live predictions, place one of these supported model setups beside `app.py`:**

- `lead_model.joblib` — complete fitted pipeline; or
- `best_lead_model.pkl` and `preprocessor.pkl` — separate classifier and fitted preprocessor.
""")

    else:

        st.success(f"Loaded: {source}")

        with st.form("lead_form"):

            vals = {}

            left, right = st.columns(2)

            for i, col in enumerate(features.columns):

                container = left if i % 2 == 0 else right

                with container:

                    s = features[col]

                    if pd.api.types.is_numeric_dtype(s):

                        med = float(s.median()) if s.notna().any() else 0

                        lo = float(s.min()) if s.notna().any() else 0

                        hi = float(s.max()) if s.notna().any() else 1

                        if hi <= lo:
                            hi = lo + 1

                        vals[col] = st.number_input(
                            col,
                            min_value=lo,
                            max_value=hi,
                            value=min(max(med, lo), hi)
                        )

                    else:

                        opts = sorted(
                            s.dropna().astype(str).unique()
                        )

                        vals[col] = st.selectbox(
                            col,
                            opts if opts else ["Unknown"]
                        )

            submitted = st.form_submit_button(
                "Analyze lead",
                type="primary",
                use_container_width=True
            )

        if submitted:

            row = pd.DataFrame(
                [vals],
                columns=features.columns
            )

            try:

                prob = get_conversion_probability(
                    predictor,
                    row
                )

                cat = bucket(prob)

                st.session_state["active_lead"] = vals.copy()

                st.session_state["active_probability"] = prob

                st.session_state["active_category"] = cat

                # Clear old AI recommendation for the previous lead
                st.session_state.pop("ai_result", None)

                st.session_state.pop(
                    "ai_result_probability",
                    None
                )

            except Exception as e:

                st.error(
                    f"Prediction could not be generated: {e}"
                )

        if "active_probability" in st.session_state:

            prob = st.session_state["active_probability"]

            cat = st.session_state["active_category"]

            st.divider()

            x, y = st.columns([1, 2])

            x.metric(
                "Conversion probability",
                f"{prob:.1%}"
            )

            x.metric(
                "Lead priority",
                cat
            )

            y.progress(
                prob,
                text="Estimated probability of Converted = 1"
            )

            st.caption(
                "This is a model estimate, not a guarantee of conversion."
            )

            st.subheader("AI explanation & sales action")

            run_ai, ask_ai, err = ai_functions()

            if run_ai is None:

                st.warning(
                    f"AI module could not be loaded: {err}"
                )

            else:

                if st.button(
                    "Generate recommendation",
                    type="primary"
                ):

                    try:

                        with st.spinner(
                            "Preparing lead-specific recommendation..."
                        ):

                            result = run_ai(
                                st.session_state["active_lead"],
                                prob
                            )

                        if not isinstance(result, dict):
                            raise ValueError(
                                "AI module returned an unexpected response."
                            )

                        st.session_state["ai_result"] = result

                        st.session_state[
                            "ai_result_probability"
                        ] = prob

                    except Exception as e:

                        st.error(
                            "AI request failed. Check OPENROUTER_API_KEY "
                            f"and API availability. Details: {e}"
                        )

                # Display saved result even after a Streamlit rerun
                saved_result = st.session_state.get("ai_result")

                saved_probability = st.session_state.get(
                    "ai_result_probability"
                )

                if (
                    saved_result is not None
                    and saved_probability == prob
                ):

                    with st.expander(
                        "Why this prediction?",
                        expanded=True
                    ):

                        st.write(
                            saved_result.get(
                                "explanation",
                                "No explanation returned."
                            )
                        )

                    with st.expander(
                        "Recommended sales action",
                        expanded=True
                    ):

                        st.write(
                            saved_result.get(
                                "recommendation",
                                "No recommendation returned."
                            )
                        )

                    with st.expander("Follow-up strategy"):

                        st.write(
                            saved_result.get(
                                "follow_up",
                                "No follow-up strategy returned."
                            )
                        )


            # =========================================================
            # DOWNLOADABLE PREDICTION REPORT
            # =========================================================

            st.divider()
            st.subheader("Download Prediction Report")

            lead_details = st.session_state.get("active_lead", {})
            ai_result = st.session_state.get("ai_result", {})

            report_lines = [
                "LEADPULSE - LEAD CONVERSION PREDICTION REPORT",
                "=" * 55,
                "",
                f"Conversion Probability: {prob:.2%}",
                f"Lead Priority: {cat}",
                "",
                "LEAD ATTRIBUTES",
                "-" * 30
            ]

            for feature, value in lead_details.items():
                report_lines.append(f"{feature}: {value}")

            report_lines.extend([
                "",
                "AI EXPLANATION",
                "-" * 30,
                str(ai_result.get(
                    "explanation",
                    "AI explanation was not generated."
                )),
                "",
                "RECOMMENDED SALES ACTION",
                "-" * 30,
                str(ai_result.get(
                    "recommendation",
                    "AI recommendation was not generated."
                )),
                "",
                "FOLLOW-UP STRATEGY",
                "-" * 30,
                str(ai_result.get(
                    "follow_up",
                    "AI follow-up strategy was not generated."
                )),
                "",
                "DISCLAIMER",
                "This conversion probability is a model estimate, "
                "not a guarantee of conversion."
            ])

            report_text = "\n".join(report_lines)

            st.download_button(
                label="Download Lead Prediction Report",
                data=report_text,
                file_name="LeadPulse_Prediction_Report.txt",
                mime="text/plain",
                type="primary",
                use_container_width=True
            )

# =========================================================
# 14. AI SALES ASSISTANT
# =========================================================

elif page == "AI Sales Assistant":

    header(
        "AI sales assistant",
        "Ask practical questions about a lead and its model-estimated conversion likelihood."
    )

    predictor, source = get_predictor()

    if predictor is None:

        st.warning(
            "Add the trained model artifact to enable lead-specific questions."
        )

    else:

        idx = st.number_input(
            "Choose lead row",
            min_value=1,
            max_value=len(df),
            value=1,
            step=1
        ) - 1

        row_number = int(idx)

        lead = features.iloc[row_number].to_dict()

        prob = get_conversion_probability(
            predictor,
            features.iloc[[row_number]]
        )

        cat = bucket(prob)

        a, b = st.columns(2)

        a.metric(
            "Estimated probability",
            f"{prob:.1%}"
        )

        b.metric(
            "Priority",
            cat
        )

        with st.expander("Lead attributes"):

            st.json(lead)

        question = st.text_area(
            "Ask your question",
            placeholder=(
                "What follow-up approach would you recommend for this lead?"
            )
        )

        run_ai, ask_ai, err = ai_functions()

        if st.button(
            "Ask assistant",
            type="primary",
            disabled=not question.strip()
        ):

            if ask_ai is None:

                st.error(
                    f"AI module unavailable: {err}"
                )

            else:

                try:

                    with st.spinner("Generating answer..."):

                        answer = ask_ai(
                            question,
                            lead,
                            prob,
                            cat
                        )

                    st.session_state["assistant_answer"] = answer

                    st.session_state["assistant_lead_row"] = row_number

                except Exception as e:

                    st.error(
                        f"AI request failed: {e}"
                    )

        # Show saved answer after reruns
        if (
            st.session_state.get("assistant_answer") is not None
            and st.session_state.get("assistant_lead_row") == row_number
        ):

            st.markdown("#### Assistant response")

            st.write(
                st.session_state["assistant_answer"]
            )


# =========================================================
# 15. MODEL INSIGHTS
# =========================================================

elif page == "Model Insights":

    header(
        "Model insights",
        "Review the verified model evaluation produced by Member 2."
    )

    result_path = BASE / "Model_Comparison_Verified.csv"

    if result_path.exists():

        metrics = pd.read_csv(result_path)

        st.dataframe(
            metrics,
            use_container_width=True,
            hide_index=True
        )

        numeric = [
            c for c in [
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score",
                "ROC-AUC"
            ]
            if c in metrics.columns
        ]

        if numeric and "Model" in metrics.columns:

            melted = metrics.melt(
                id_vars="Model",
                value_vars=numeric,
                var_name="Metric",
                value_name="Score"
            )

            fig = px.bar(
                melted,
                x="Model",
                y="Score",
                color="Metric",
                barmode="group",
                range_y=[0, 1]
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    else:

        st.info(
            "Model comparison file not found. "
            "Add Member 2's verified Model_Comparison_Verified.csv "
            "to this folder."
        )

    # =========================================================
    # CONFUSION MATRIX
    # =========================================================

    st.divider()
    st.subheader("Confusion Matrix - XGBoost Classifier")

    x_test_path = BASE / "X_test_processed.pkl"
    y_test_path = BASE / "y_test.pkl"

    if x_test_path.exists() and y_test_path.exists():
        try:
            from sklearn.metrics import confusion_matrix

            X_test = joblib.load(x_test_path)
            y_test = np.asarray(joblib.load(y_test_path)).ravel()

            model = joblib.load(MODEL_PATH)
            y_pred = model.predict(X_test)

            cm = confusion_matrix(
                y_test,
                y_pred,
                labels=[0, 1]
            )

            fig_cm = go.Figure(
                data=go.Heatmap(
                    z=cm,
                    x=["Not Converted", "Converted"],
                    y=["Not Converted", "Converted"],
                    text=cm,
                    texttemplate="%{text}",
                    colorscale="Blues",
                    showscale=True
                )
            )

            fig_cm.update_layout(
                title="Actual vs Predicted Lead Outcomes",
                xaxis_title="Predicted Outcome",
                yaxis_title="Actual Outcome",
                height=450
            )

            st.plotly_chart(
                fig_cm,
                use_container_width=True
            )

            st.caption(
                "Evaluation uses the original saved test features "
                "and actual test labels."
            )

        except Exception as e:
            st.error(f"Could not generate confusion matrix: {e}")

    else:
        st.warning(
            "X_test_processed.pkl or y_test.pkl is missing."
        )
    
    # =========================================================
    # FEATURE IMPORTANCE ANALYSIS
    # =========================================================

    st.divider()
    st.subheader("Feature Importance Analysis - XGBoost")

    feature_names_path = BASE / "processed_feature_names.csv"

    if feature_names_path.exists():
        try:
            model = joblib.load(MODEL_PATH)

            feature_df = pd.read_csv(feature_names_path)

            importance_df = pd.DataFrame({
                "Feature": feature_df["Feature Name"],
                "Importance": model.feature_importances_
            })

            importance_df = importance_df.sort_values(
                by="Importance",
                ascending=False
            )

            top_features = importance_df.head(10).sort_values(
                by="Importance",
                ascending=True
            )

            fig_importance = px.bar(
                top_features,
                x="Importance",
                y="Feature",
                orientation="h",
                title="Top 10 Features Influencing Model Predictions",
                labels={
                    "Importance": "Relative Importance",
                    "Feature": "Lead Attributes"
                }
            )

            fig_importance.update_layout(
                height=550,
                yaxis_title="Lead Attributes",
                xaxis_title="Relative Importance"
            )

            st.plotly_chart(
                fig_importance,
                use_container_width=True
            )

            with st.expander("View all 50 feature importance scores"):
                st.dataframe(
                    importance_df,
                    use_container_width=True,
                    hide_index=True
                )

            st.caption(
                "Feature importance represents the model's relative "
                "use of each feature. It does not establish causation."
            )

        except Exception as e:
            st.error(f"Could not generate feature importance: {e}")

    else:
        st.warning("processed_feature_names.csv is missing.")

    st.markdown("#### Priority thresholds")

    st.write(
        "High: ≥70%  ·  Medium: 40%–69.9%  ·  Low: <40%"
    )

    st.caption(
        "These are configurable business rules, not universal definitions. "
        "Review them against sales capacity and validation results."
    )


# =========================================================
# 16. FOOTER
# =========================================================

st.divider()

st.caption(
    "LeadPulse · AI-Based Lead Conversion Prediction "
    "and Intelligent Sales Recommendation System"
)
