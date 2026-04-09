import sys
from pathlib import Path

import streamlit as st

# Ensure the repository root is importable when Streamlit runs this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.demo_pipeline import DemoPipelineV1


# Configure the page before rendering any widgets.
st.set_page_config(page_title="Evidence-Backed Clinical Insight Assistant", layout="wide")
st.title("Evidence-Backed Clinical Insight Assistant")
st.caption("Integrated demo: summary + NER + clustering + classification")

@st.cache_resource
def load_pipeline():
    # Cache the assembled pipeline so the app does not reload models on every interaction.
    return DemoPipelineV1(base_dir=Path(__file__).resolve().parent)

# Initialize the shared inference pipeline once per Streamlit session.
pipeline = load_pipeline()

# Seed the interface with a realistic sample note for quick testing.
default_text = (
    "Patient presents with chest pain and shortness of breath. Past history includes "
    "coronary artery disease and tachycardia. Nitroglycerin provided partial relief."
)

user_text = st.text_area("Paste an EMR note", value=default_text, height=240)

if st.button("Analyze"):
    # Execute the end-to-end inference flow for the current note.
    result = pipeline.analyze_emr(user_text)

    # Split the UI into evidence extraction on the left and model outputs on the right.
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Summary")
        st.write(result["summary"] or "No summary available.")

        st.subheader("Extracted Entities")
        entities = result["entities"]
        st.markdown(f"**Diseases:** {', '.join(entities.get('diseases', [])) or 'None'}")
        st.markdown(f"**Symptoms:** {', '.join(entities.get('symptoms', [])) or 'None'}")
        st.markdown(f"**Medications:** {', '.join(entities.get('medications', [])) or 'None'}")

    with col2:
        st.subheader("Cluster Insight")
        cluster = result["cluster"]
        st.write(f"**Cluster ID:** {cluster['cluster_id']}")
        st.write(f"**Description:** {cluster['description']}")
        st.write(f"**Top Terms:** {cluster['top_terms'] or 'N/A'}")
        st.write(f"**Top Entities:** {cluster['top_entities'] or 'N/A'}")

        st.subheader("Specialty Prediction")
        specialty = result["specialty"]
        if specialty["predicted_label"] is None:
            st.info("Classifier output not available.")
        else:
            st.write(f"**Predicted Label:** {specialty['predicted_label']}")
            if specialty["confidence"] is not None:
                st.write(f"**Confidence:** {specialty['confidence']:.4f}")
            else:
                st.write("**Confidence:** N/A")
            st.write(f"**Feature Mode:** {specialty.get('feature_mode', 'unknown')}")

    # Surface the final cross-module explanation below the detailed outputs.
    st.subheader("Evidence Note")
    st.write(result["evidence_note"])

    # Keep the raw structured payload available for debugging and demos.
    with st.expander("Raw JSON Output"):
        st.json(result)
