import json
import pickle
import importlib.util
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd

from src.config import CLUSTERING_MODELS_DIR, NER_SUMMARIZATION_DIR, PROCESSED_DATA_DIR


def _load_member_b_module(module_path: Path):
    """Dynamically load Member B's text_processing.py if available."""
    if not module_path.exists():
        return None

    spec = importlib.util.spec_from_file_location("member_b_text_processing", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class DemoPipelineV1:
    """
    Member C integration pipeline v1.

    Current behavior:
    - Uses Member B's extract_entities() and generate_summary() when available.
    - Uses Member C's clustering model/vectorizer.
    - Reserves a stable interface for Member A's classification model.
    """

    def __init__(self, artifacts_dir: Path = None):
        self.artifacts_dir = Path(artifacts_dir) if artifacts_dir else CLUSTERING_MODELS_DIR

        with open(self.artifacts_dir / "cluster_model.pkl", "rb") as f:
            self.cluster_model = pickle.load(f)

        with open(self.artifacts_dir / "cluster_vectorizer.pkl", "rb") as f:
            self.cluster_vectorizer = pickle.load(f)

        self.cluster_interpretation = pd.read_csv(PROCESSED_DATA_DIR / "cluster_interpretation.csv")

        mock_path = NER_SUMMARIZATION_DIR / "member_B_mock_results.json"
        if mock_path.exists():
            with open(mock_path, "r", encoding="utf-8") as f:
                self.mock_results = json.load(f)
        else:
            self.mock_results = []

        self.member_b = _load_member_b_module(NER_SUMMARIZATION_DIR / "text_processing.py")

        # Placeholder for Member A integration
        self.classifier = None
        self.classifier_vectorizer = None

    @staticmethod
    def preprocess_input(text: str) -> str:
        return " ".join(str(text).strip().split())

    @staticmethod
    def _entities_to_text(entities: Dict[str, List[str]]) -> str:
        return " ".join(
            entities.get("diseases", [])
            + entities.get("symptoms", [])
            + entities.get("medications", [])
        ).strip()

    def run_ner_and_summary(self, text: str) -> Dict[str, Any]:
        """
        Prefer Member B real functions.
        Fallback to the first mock result template if the module is unavailable.
        """
        if self.member_b is not None:
            entities = self.member_b.extract_entities(text)
            summary = self.member_b.generate_summary(text)
            return {"summary": summary, "entities": entities}

        if self.mock_results:
            sample = self.mock_results[0]
            return {
                "summary": sample.get("evidence_summary", ""),
                "entities": sample.get("extracted_entities", {
                    "diseases": [],
                    "symptoms": [],
                    "medications": []
                })
            }

        return {
            "summary": "",
            "entities": {"diseases": [], "symptoms": [], "medications": []}
        }

    def predict_cluster(self, entity_text_all: str, summary: str) -> Dict[str, Any]:
        hybrid_text = f"{entity_text_all} {summary}".strip()
        if not hybrid_text:
            hybrid_text = summary.strip()

        X = self.cluster_vectorizer.transform([hybrid_text])
        cluster_id = int(self.cluster_model.predict(X)[0])

        row = self.cluster_interpretation[
            self.cluster_interpretation["cluster_id"] == cluster_id
        ].iloc[0]

        return {
            "cluster_id": cluster_id,
            "top_terms": row["top_terms"],
            "top_entities": row["top_entities"],
            "description": row["short_description"],
        }

    def predict_specialty(self, text: str, entities: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Reserved Member A interface.
        Later, Member A can attach classifier + vectorizer here.
        """
        if self.classifier is None or self.classifier_vectorizer is None:
            return {
                "predicted_label": None,
                "confidence": None,
                "status": "pending member1 integration"
            }

        # Placeholder branch kept for future integration.
        return {
            "predicted_label": None,
            "confidence": None,
            "status": "interface reserved"
        }

    @staticmethod
    def build_evidence_note(entities: Dict[str, List[str]], cluster_result: Dict[str, Any]) -> str:
        support_items = (
            entities.get("diseases", [])[:2]
            + entities.get("symptoms", [])[:3]
            + entities.get("medications", [])[:2]
        )
        if not support_items:
            return (
                f"This cluster assignment is mainly supported by the summary pattern "
                f"and matches the cohort described as: {cluster_result['description']}."
            )

        support_str = ", ".join(support_items)
        return (
            f"This cluster assignment is mainly supported by extracted evidence such as "
            f"{support_str}. It is most aligned with the cohort described as: "
            f"{cluster_result['description']}."
        )

    def analyze_emr(self, text: str) -> Dict[str, Any]:
        clean_text = self.preprocess_input(text)
        base_result = self.run_ner_and_summary(clean_text)
        entity_text_all = self._entities_to_text(base_result["entities"])
        cluster_result = self.predict_cluster(entity_text_all, base_result["summary"])
        specialty_result = self.predict_specialty(clean_text, base_result["entities"])
        evidence_note = self.build_evidence_note(base_result["entities"], cluster_result)

        return {
            "summary": base_result["summary"],
            "entities": base_result["entities"],
            "cluster": cluster_result,
            "specialty": specialty_result,
            "evidence_note": evidence_note,
        }


if __name__ == "__main__":
    demo = DemoPipelineV1()
    sample_text = (
        "Patient presents with chest pain, shortness of breath, and history of "
        "coronary artery disease. Current medications include nitroglycerin."
    )
    result = demo.analyze_emr(sample_text)
    print(json.dumps(result, indent=2))
