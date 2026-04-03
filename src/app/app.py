import streamlit as st
from src.pipeline.demo_pipeline_v1 import DemoPipelineV1

st.set_page_config(page_title="Evidence-Backed Clinical Insight Assistant", layout="wide")
st.title("Evidence-Backed Clinical Insight Assistant")
st.caption("Member C demo v1: clustering + integration shell")

pipeline = DemoPipelineV1()

default_text = (
    "Patient presents with chest pain and shortness of breath. Past history includes "
    "coronary artery disease and tachycardia. Nitroglycerin provided partial relief."
)

user_text = st.text_area("Paste an EMR note", value=default_text, height=220)

if st.button("Analyze"):
    result = pipeline.analyze_emr(user_text)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Summary")
        st.write(result["summary"] or "No summary available.")

        st.subheader("Extracted Entities")
        st.write("**Diseases**")
        st.write(result["entities"].get("diseases", []))
        st.write("**Symptoms**")
        st.write(result["entities"].get("symptoms", []))
        st.write("**Medications**")
        st.write(result["entities"].get("medications", []))

    with col2:
        st.subheader("Cluster Insight")
        st.write(f"**Cluster ID:** {result['cluster']['cluster_id']}")
        st.write(f"**Description:** {result['cluster']['description']}")
        st.write(f"**Top Terms:** {result['cluster']['top_terms']}")
        st.write(f"**Top Entities:** {result['cluster']['top_entities']}")

        st.subheader("Specialty Prediction")
        specialty = result["specialty"]
        if specialty["predicted_label"] is None:
            st.info("Pending Member 1 integration.")
        else:
            st.write(f"**Predicted Label:** {specialty['predicted_label']}")
            st.write(f"**Confidence:** {specialty['confidence']}")

    st.subheader("Evidence Note")
    st.write(result["evidence_note"])
