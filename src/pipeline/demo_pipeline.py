import ast
import importlib.util
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import (
    CLASSIFICATION_MODELS_DIR,
    CLASSIFICATION_VECTORIZER_PATH,
    CLUSTERING_MODELS_DIR,
    CLUSTER_VECTORIZER_PATH,
    INTERIM_DATA_DIR,
    NER_SUMMARIZATION_DIR,
    PIPELINE_DIR,
    PROCESSED_DATA_DIR,
)

BASE_DIR = Path(__file__).resolve().parent


def _load_member_b_module(module_path: Path):
    """Dynamically load Member B's text_processing.py if available."""
    if not module_path.exists():
        return None
    spec = importlib.util.spec_from_file_location("member_b_text_processing", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _clean_text(text: Any) -> str:
    if pd.isna(text):
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


class DemoPipelineV1:
    """
    Unified inference pipeline for Member C integration.

    Current behavior:
    - Loads clustering assets and cluster interpretation.
    - Loads Member B's NER + summarization module when available.
    - Loads Member A's classifier and infers which feature mode it expects.
    - Returns a unified JSON-style response for the Streamlit demo.
    """

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = Path(base_dir) if base_dir else BASE_DIR
        self.cluster_model_path = CLUSTERING_MODELS_DIR / "cluster_model.pkl"
        self.cluster_vectorizer_path = CLUSTER_VECTORIZER_PATH
        self.cluster_interpretation_path = PROCESSED_DATA_DIR / "cluster_interpretation.csv"
        self.classifier_path = CLASSIFICATION_MODELS_DIR / "best_classifier.pkl"
        self.classification_vectorizer_path = CLASSIFICATION_VECTORIZER_PATH
        self.classification_results_path = PROCESSED_DATA_DIR / "classification_results.csv"
        self.test_data_path = INTERIM_DATA_DIR / "test.csv"
        self.ner_results_path = PROCESSED_DATA_DIR / "ner results.csv"
        self.mock_results_path = NER_SUMMARIZATION_DIR / "member_B_mock_results.json"
        self.member_b_module_path = NER_SUMMARIZATION_DIR / "text_processing.py"

        self.cluster_model = self._load_joblib(self.cluster_model_path)
        self.cluster_vectorizer = self._load_joblib(self.cluster_vectorizer_path)
        self.cluster_interpretation = self._safe_read_csv(self.cluster_interpretation_path)

        self.classifier = self._load_joblib(self.classifier_path)
        self.classification_vectorizer = self._load_joblib(self.classification_vectorizer_path)
        self.classification_results = self._safe_read_csv(self.classification_results_path)

        self.member_b = _load_member_b_module(self.member_b_module_path)
        self.mock_results = self._load_mock_results()

        self.classification_feature_mode = self._infer_classification_feature_mode()

    def _load_joblib(self, path: Path):
        if path.exists():
            return joblib.load(path)
        return None

    def _safe_read_csv(self, path: Path) -> pd.DataFrame:
        if path.exists():
            return pd.read_csv(path)
        return pd.DataFrame()

    def _load_mock_results(self) -> List[Dict[str, Any]]:
        path = self.mock_results_path
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    @staticmethod
    def preprocess_input(text: str) -> str:
        return _clean_text(text)

    @staticmethod
    def _entities_to_text_jsonlike(entities: Dict[str, List[str]]) -> str:
        """
        Mimic Member A's training format as closely as possible.
        train_classifier.py built entity_text from string columns such as
        diseases + symptoms + medications, where the NER CSV stores list-like strings.
        """
        diseases = json.dumps(entities.get("diseases", []), ensure_ascii=False)
        symptoms = json.dumps(entities.get("symptoms", []), ensure_ascii=False)
        medications = json.dumps(entities.get("medications", []), ensure_ascii=False)
        return _clean_text(f"{diseases} {symptoms} {medications}")

    def _infer_classification_feature_mode(self) -> str:
        """
        Infer whether Member A's saved classifier expects transcription,
        entity_text, or hybrid_text by matching saved predictions against
        uploaded classification_results.csv.
        """
        if self.classifier is None:
            return "hybrid_text"

        test_path = self.test_data_path
        ner_path = self.ner_results_path

        if not (test_path.exists() and ner_path.exists() and not self.classification_results.empty):
            return "hybrid_text"

        test_df = pd.read_csv(test_path)
        ner_df = pd.read_csv(ner_path)

        ner_df["record_id"] = ner_df["record_id"].astype(str).str.strip()
        for col in ["diseases", "symptoms", "medications"]:
            if col not in ner_df.columns:
                ner_df[col] = ""
            ner_df[col] = ner_df[col].fillna("").astype(str)

        ner_df["entity_text"] = (
            ner_df["diseases"] + " " + ner_df["symptoms"] + " " + ner_df["medications"]
        ).apply(_clean_text)
        ner_feature_df = ner_df[["record_id", "entity_text"]].drop_duplicates("record_id")

        test_df["record_id"] = test_df["record_id"].astype(str).str.strip()
        test_df["transcription"] = test_df["transcription"].apply(_clean_text)
        merged = test_df.merge(ner_feature_df, on="record_id", how="left")
        merged["entity_text"] = merged["entity_text"].fillna("").apply(_clean_text)
        merged["hybrid_text"] = (merged["transcription"] + " " + merged["entity_text"]).apply(_clean_text)

        expected = self.classification_results["predicted_label"].astype(str).tolist()
        scores = {}
        for mode in ["transcription", "entity_text", "hybrid_text"]:
            try:
                pred = self.classifier.predict(merged[mode]).tolist()
                scores[mode] = sum(str(a) == str(b) for a, b in zip(pred, expected)) / len(expected)
            except Exception:
                scores[mode] = -1.0

        return max(scores, key=scores.get)

    def run_ner_and_summary(self, text: str) -> Dict[str, Any]:
        if self.member_b is not None:
            entities = self.member_b.extract_entities(text)
            summary = self.member_b.generate_summary(text)
            return {"summary": summary, "entities": entities}

        if self.mock_results:
            sample = self.mock_results[0]
            return {
                "summary": sample.get("evidence_summary", ""),
                "entities": sample.get(
                    "extracted_entities",
                    {"diseases": [], "symptoms": [], "medications": []},
                ),
            }

        return {
            "summary": "",
            "entities": {"diseases": [], "symptoms": [], "medications": []},
        }

    def predict_cluster(self, entity_text_all: str, summary: str, clean_text: str) -> Dict[str, Any]:
        hybrid_text = _clean_text(f"{entity_text_all} {summary}")
        if not hybrid_text:
            hybrid_text = clean_text

        if self.cluster_model is None or self.cluster_vectorizer is None:
            return {
                "cluster_id": None,
                "top_terms": "",
                "top_entities": "",
                "description": "Cluster model not available.",
            }

        X = self.cluster_vectorizer.transform([hybrid_text])
        cluster_id = int(self.cluster_model.predict(X)[0])

        if not self.cluster_interpretation.empty and "cluster_id" in self.cluster_interpretation.columns:
            row = self.cluster_interpretation[
                self.cluster_interpretation["cluster_id"] == cluster_id
            ]
            if not row.empty:
                row = row.iloc[0]
                return {
                    "cluster_id": cluster_id,
                    "top_terms": row.get("top_terms", ""),
                    "top_entities": row.get("top_entities", ""),
                    "description": row.get("short_description", ""),
                }

        return {
            "cluster_id": cluster_id,
            "top_terms": "",
            "top_entities": "",
            "description": "Cluster assigned successfully.",
        }

    def predict_specialty(self, clean_text: str, entities: Dict[str, List[str]]) -> Dict[str, Any]:
        if self.classifier is None:
            return {
                "predicted_label": None,
                "confidence": None,
                "status": "classifier not available",
                "feature_mode": self.classification_feature_mode,
            }

        entity_text = self._entities_to_text_jsonlike(entities)
        feature_map = {
            "transcription": clean_text,
            "entity_text": entity_text,
            "hybrid_text": _clean_text(f"{clean_text} {entity_text}"),
        }
        input_text = feature_map.get(self.classification_feature_mode, feature_map["hybrid_text"])

        predicted_label = self.classifier.predict([input_text])[0]
        confidence = None

        try:
            if hasattr(self.classifier, "predict_proba"):
                probas = self.classifier.predict_proba([input_text])[0]
                confidence = float(max(probas))
            elif self.classification_vectorizer is not None:
                X_vec = self.classification_vectorizer.transform([input_text])
                if hasattr(self.classifier, "predict_proba"):
                    probas = self.classifier.predict_proba(X_vec)[0]
                    confidence = float(max(probas))
            elif isinstance(self.classifier, Pipeline) and "clf" in self.classifier.named_steps:
                clf = self.classifier.named_steps["clf"]
                tfidf = self.classifier.named_steps.get("tfidf")
                if hasattr(clf, "predict_proba") and tfidf is not None:
                    X_vec = tfidf.transform([input_text])
                    probas = clf.predict_proba(X_vec)[0]
                    confidence = float(max(probas))
        except Exception:
            confidence = None

        return {
            "predicted_label": str(predicted_label),
            "confidence": confidence,
            "status": "ready",
            "feature_mode": self.classification_feature_mode,
        }

    @staticmethod
    def build_evidence_note(
        entities: Dict[str, List[str]],
        cluster_result: Dict[str, Any],
        specialty_result: Dict[str, Any],
    ) -> str:
        support_items = (
            entities.get("diseases", [])[:2]
            + entities.get("symptoms", [])[:3]
            + entities.get("medications", [])[:2]
        )
        specialty_text = specialty_result.get("predicted_label") or "the current specialty output"
        cluster_desc = cluster_result.get("description") or "the assigned cohort"

        if not support_items:
            return (
                f"The specialty prediction is {specialty_text}. "
                f"The case also aligns with {cluster_desc}, based mainly on summary-level evidence."
            )

        support_str = ", ".join(support_items)
        return (
            f"The specialty prediction is {specialty_text}. "
            f"This result is supported by extracted evidence such as {support_str}, "
            f"and the case is also aligned with {cluster_desc}."
        )

    def analyze_emr(self, text: str) -> Dict[str, Any]:
        clean_text = self.preprocess_input(text)
        base_result = self.run_ner_and_summary(clean_text)
        entity_text_all = self._entities_to_text_jsonlike(base_result["entities"])
        cluster_result = self.predict_cluster(entity_text_all, base_result["summary"], clean_text)
        specialty_result = self.predict_specialty(clean_text, base_result["entities"])
        evidence_note = self.build_evidence_note(
            base_result["entities"], cluster_result, specialty_result
        )

        return {
            "summary": base_result["summary"],
            "entities": base_result["entities"],
            "cluster": cluster_result,
            "specialty": specialty_result,
            "evidence_note": evidence_note,
        }


if __name__ == "__main__":
    demo = DemoPipelineV1(base_dir=PIPELINE_DIR)
    sample_text = (
        "Patient presents with chest pain, shortness of breath, and history of "
        "coronary artery disease. Current medications include nitroglycerin."
    )
    result = demo.analyze_emr(sample_text)
    print(json.dumps(result, indent=2, ensure_ascii=False))
