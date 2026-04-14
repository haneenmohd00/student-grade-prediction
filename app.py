import os
import logging

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import (
    DATA_PATH,
    MODEL_PATH,
    PREPROCESSOR_PATH,
    PROJECT_TITLE,
    TARGET_COLUMN,
)
from hf_api import generate_ai_response
from predict import ensure_model, predict_from_dict
from train import main as train_main
from utils import load_dataset, load_metrics, setup_logging


setup_logging()
logger = logging.getLogger("app")


def load_local_css(path: str) -> None:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


@st.cache_resource(show_spinner=True)
def get_model_and_metrics():
    model, metrics = ensure_model()
    return model, metrics


def sidebar_navigation() -> str:
    st.sidebar.title("Navigation")
    return st.sidebar.radio(
        "Go to",
        [
            "Dashboard",
            "Ticket Classifier",
            "AI Response Generator",
            "Model Analytics",
            "Admin Panel",
        ],
    )


def render_kpi(label: str, value: str, sub: str = "") -> None:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_dashboard(model, metrics):
    st.subheader("Overview")
    try:
        df = load_dataset(DATA_PATH)
        dataset_size = len(df)
    except Exception:
        df = None
        dataset_size = 0

    best_model_name = metrics.get("best_model", "N/A") if metrics else "N/A"
    best_accuracy = 0.0
    if metrics and "best_model" in metrics:
        best_accuracy = metrics["models"][best_model_name]["accuracy"]

    col1, col2, col3 = st.columns(3)
    with col1:
        render_kpi("Dataset Size", f"{dataset_size}", "Rows in current dataset")
    with col2:
        render_kpi("Best Model", best_model_name, "Selected by validation accuracy")
    with col3:
        render_kpi("Accuracy", f"{best_accuracy:.3f}", "Validation accuracy")

    st.markdown("---")
    st.subheader("Class Distribution")

    if df is not None and TARGET_COLUMN in df.columns:
        class_counts = df[TARGET_COLUMN].value_counts(dropna=False)
        fig = px.bar(
            class_counts,
            x=class_counts.index.astype(str),
            y=class_counts.values,
            labels={"x": "Class", "y": "Count"},
            title="Class Distribution",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Target column not found in dataset. Upload a dataset with a valid target.")


def page_ticket_classifier(model, metrics):
    st.subheader("Ticket Classifier")
    st.caption(
        "Enter student and study details to predict the performance class and confidence."
    )

    with st.form("ticket_classifier_form"):
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Age", min_value=5, max_value=100, value=18)
            study_time = st.number_input(
                "Study Time Weekly (hours)", min_value=0.0, value=10.0
            )
            absences = st.number_input("Absences", min_value=0, value=0)
            gpa = st.number_input("GPA", min_value=0.0, max_value=4.0, value=3.0)
        with c2:
            gender = st.selectbox("Gender", ["", "Male", "Female", "Other"])
            ethnicity = st.text_input("Ethnicity", "")
            parental_education = st.text_input("Parental Education", "")
            parental_support = st.selectbox(
                "Parental Support", ["", "Low", "Medium", "High"]
            )

        c3, c4 = st.columns(2)
        with c3:
            tutoring = st.selectbox("Tutoring", ["", "Yes", "No"])
            extracurricular = st.selectbox("Extracurricular", ["", "Yes", "No"])
        with c4:
            sports = st.selectbox("Sports", ["", "Yes", "No"])
            music = st.selectbox("Music", ["", "Yes", "No"])
            volunteering = st.selectbox("Volunteering", ["", "Yes", "No"])

        notes = st.text_area("Additional Notes (optional text input)", "")

        submitted = st.form_submit_button("Predict Grade Class")

    if submitted:
        with st.spinner("Predicting grade class..."):
            sample = {
                "Age": age,
                "StudyTimeWeekly": study_time,
                "Absences": absences,
                "GPA": gpa,
                "Gender": gender or None,
                "Ethnicity": ethnicity or None,
                "ParentalEducation": parental_education or None,
                "ParentalSupport": parental_support or None,
                "Tutoring": tutoring or None,
                "Extracurricular": extracurricular or None,
                "Sports": sports or None,
                "Music": music or None,
                "Volunteering": volunteering or None,
                "StudentID": None,
            }

            try:
                result = predict_from_dict(model, sample)
                prediction = result.get("prediction", "N/A")
                confidence = result.get("confidence")

                st.success(f"Predicted Grade Class: **{prediction}**")
                if confidence is not None:
                    st.metric("Confidence", f"{confidence:.3f}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")


def page_ai_response():
    st.subheader("AI Response Generator")
    st.caption("Use the HuggingFace Router-backed LLM to generate a helpful response.")

    ticket_text = st.text_area(
        "Enter your question or ticket text",
        placeholder="Describe the student's situation, challenges, or support needs...",
        height=200,
    )

    if st.button("Generate AI Response"):
        if not ticket_text.strip():
            st.warning("Please enter some text for the AI to respond to.")
            return

        with st.spinner("Generating AI response..."):
            response = generate_ai_response(ticket_text.strip())
            st.markdown("**AI Response:**")
            st.write(response)


def page_model_analytics(model, metrics):
    st.subheader("Model Analytics")

    if not metrics:
        st.info("No metrics available yet. Train a model from the Admin Panel.")
        return

    best_model_name = metrics.get("best_model")
    models_info = metrics.get("models", {})

    if best_model_name not in models_info:
        st.info("Best model metrics not found.")
        return

    best_info = models_info[best_model_name]

    st.markdown(f"**Best Model:** `{best_model_name}`")
    st.markdown(f"**Accuracy:** `{best_info.get('accuracy', 0.0):.3f}`")

    st.markdown("---")
    st.subheader("Accuracy Comparison")

    model_names = list(models_info.keys())
    accuracies = [models_info[m]["accuracy"] for m in model_names]
    fig_acc = px.bar(
        x=model_names,
        y=accuracies,
        labels={"x": "Model", "y": "Accuracy"},
        title="Model Accuracy Comparison",
    )
    st.plotly_chart(fig_acc, use_container_width=True)

    st.markdown("---")
    st.subheader("Confusion Matrix")

    cm = np.array(best_info.get("confusion_matrix"))
    if cm.size > 0:
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            color_continuous_scale="Blues",
            labels=dict(x="Predicted", y="Actual", color="Count"),
            title="Confusion Matrix",
        )
        st.plotly_chart(fig_cm, use_container_width=True)
    else:
        st.info("Confusion matrix not available.")

    st.markdown("---")
    st.subheader("Feature Importance (if available)")

    clf = model.named_steps.get("clf")
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
        indices = np.argsort(importances)[::-1][:20]
        fig_imp = go.Figure(
            data=go.Bar(
                x=[str(i) for i in indices],
                y=importances[indices],
            )
        )
        fig_imp.update_layout(
            title="Top Feature Importances (indices from transformed space)",
            xaxis_title="Feature Index",
            yaxis_title="Importance",
        )
        st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.info("Feature importances are not available for the selected model.")


def page_admin_panel():
    st.subheader("Admin Panel")
    st.caption("Upload a new dataset and retrain the model.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is not None:
        try:
            df_new = pd.read_csv(uploaded)
            os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
            df_new.to_csv(DATA_PATH, index=False)
            st.success("New dataset uploaded successfully.")
        except Exception as e:
            st.error(f"Failed to save uploaded dataset: {e}")

    if st.button("Retrain Model"):
        with st.spinner("Retraining model..."):
            try:
                train_main()
                # Clear cache so new model is loaded
                st.cache_resource.clear()
                st.success("Model retrained successfully.")
            except Exception as e:
                st.error(f"Retraining failed: {e}")


def main():
    st.set_page_config(
        page_title=PROJECT_TITLE,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    load_local_css("assets/style.css")

    st.title(PROJECT_TITLE)
    st.caption(
        "End-to-end student performance prediction with classical ML models and an AI tutor (LLM)."
    )

    page = sidebar_navigation()

    if page in {"Dashboard", "Ticket Classifier", "Model Analytics"}:
        with st.spinner("Loading model and metrics..."):
            try:
                model, metrics = get_model_and_metrics()
            except Exception as e:
                st.error(f"Failed to load or train model: {e}")
                model, metrics = None, None
    else:
        model, metrics = None, None

    if page == "Dashboard":
        if model is not None:
            page_dashboard(model, metrics or {})
        else:
            st.info("Model is not available yet. Go to Admin Panel to train.")
    elif page == "Ticket Classifier":
        if model is not None:
            page_ticket_classifier(model, metrics or {})
        else:
            st.info("Model is not available yet. Go to Admin Panel to train.")
    elif page == "AI Response Generator":
        page_ai_response()
    elif page == "Model Analytics":
        if model is not None:
            page_model_analytics(model, metrics or {})
        else:
            st.info("Model is not available yet. Go to Admin Panel to train.")
    elif page == "Admin Panel":
        page_admin_panel()


if __name__ == "__main__":
    main()

