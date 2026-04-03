import ast
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

from src.config import CLUSTERING_MODELS_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR


CLEANED_PATH = INTERIM_DATA_DIR / "cleaned_dataset.csv"
NER_PATH = PROCESSED_DATA_DIR / "ner results.csv"
SUMMARY_PATH = PROCESSED_DATA_DIR / "summary results.csv"
CLUSTER_RESULTS_PATH = PROCESSED_DATA_DIR / "cluster_results.csv"
CLUSTER_INTERPRETATION_PATH = PROCESSED_DATA_DIR / "cluster_interpretation.csv"
SILHOUETTE_SCORES_PATH = PROCESSED_DATA_DIR / "silhouette_scores.csv"
CLUSTER_MODEL_PATH = CLUSTERING_MODELS_DIR / "cluster_model.pkl"
CLUSTER_VECTORIZER_PATH = CLUSTERING_MODELS_DIR / "cluster_vectorizer.pkl"
INTEGRATION_BASE_PATH = CLUSTERING_MODELS_DIR / "integration_base.pkl"


def parse_list_cell(value) -> List[str]:
    """Convert a CSV cell like '["Pain", "Fever"]' to a Python list."""
    if pd.isna(value):
        return []
    text = str(value).strip()
    if text in {"", "[]", "nan", "None"}:
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except Exception:
        pass
    return [text]


def build_integration_base() -> pd.DataFrame:
    cleaned = pd.read_csv(CLEANED_PATH)
    ner = pd.read_csv(NER_PATH)
    summary = pd.read_csv(SUMMARY_PATH)

    for col in ["diseases", "symptoms", "medications"]:
        ner[col] = ner[col].apply(parse_list_cell)

    ner["entity_text_all"] = ner.apply(
        lambda row: " ".join(row["diseases"] + row["symptoms"] + row["medications"]),
        axis=1,
    )

    base = cleaned.merge(
        ner[[
            "record_id",
            "diseases",
            "symptoms",
            "medications",
            "disease_count",
            "symptom_count",
            "medication_count",
            "entity_text_all",
        ]],
        on="record_id",
        how="left",
    ).merge(
        summary[["record_id", "summary"]],
        on="record_id",
        how="left",
    )

    base["summary"] = base["summary"].fillna("")
    base["entity_text_all"] = base["entity_text_all"].fillna("")
    base["hybrid_text"] = (base["entity_text_all"] + " " + base["summary"]).str.strip()
    # fallback so records with empty entities still have signal
    empty_mask = base["hybrid_text"].eq("")
    base.loc[empty_mask, "hybrid_text"] = base.loc[empty_mask, "transcription"].fillna("")

    return base


def run_kmeans_search(texts: pd.Series, k_values: List[int]) -> Tuple[TfidfVectorizer, pd.DataFrame, MiniBatchKMeans]:
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
    )
    X = vectorizer.fit_transform(texts)

    score_rows = []
    best_score = -1.0
    best_model = None

    for k in k_values:
        model = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=512, n_init=10)
        labels = model.fit_predict(X)
        score = silhouette_score(X, labels, sample_size=min(2000, X.shape[0]), random_state=42)
        score_rows.append({"k": k, "silhouette_score": float(score)})
        if score > best_score:
            best_score = score
            best_model = model

    scores_df = pd.DataFrame(score_rows)
    return vectorizer, scores_df, best_model


def get_top_terms_per_cluster(model: MiniBatchKMeans, vectorizer: TfidfVectorizer, top_n: int = 12):
    terms = np.array(vectorizer.get_feature_names_out())
    results = []
    for cluster_id, center in enumerate(model.cluster_centers_):
        top_idx = center.argsort()[-top_n:][::-1]
        top_terms = terms[top_idx].tolist()
        results.append({
            "cluster_id": cluster_id,
            "top_terms": "; ".join(top_terms),
        })
    return pd.DataFrame(results)


def get_top_entities_for_cluster(df: pd.DataFrame, cluster_id: int, top_n: int = 8) -> str:
    subset = df[df["cluster_id"] == cluster_id]
    all_entities = []
    for col in ["diseases", "symptoms", "medications"]:
        vals = subset[col].tolist()
        for item in vals:
            if isinstance(item, list):
                all_entities.extend(item)
    if not all_entities:
        return ""
    vc = pd.Series(all_entities).value_counts().head(top_n)
    return "; ".join(vc.index.tolist())


def describe_cluster(top_terms: str, top_entities: str) -> str:
    text = f"{top_terms} {top_entities}".lower()
    if any(w in text for w in ["stomach", "colon", "abdomen", "bowel", "gastro", "liver"]):
        return "Likely gastrointestinal-related cases"
    if any(w in text for w in ["knee", "hip", "fracture", "spine", "shoulder", "orthopedic"]):
        return "Likely orthopedic or musculoskeletal cases"
    if any(w in text for w in ["heart", "atrial", "ventricular", "pulmonary", "cardiac", "hypertension"]):
        return "Likely cardiovascular or pulmonary cases"
    if any(w in text for w in ["surgery", "incision", "operative", "procedure", "anesthesia"]):
        return "Likely surgery or perioperative cases"
    if any(w in text for w in ["brain", "seizure", "headache", "neurology", "nerve"]):
        return "Likely neurology-related cases"
    return "General mixed clinical cohort"


def main():
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    CLUSTERING_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = build_integration_base()
    vectorizer, scores_df, model = run_kmeans_search(df["hybrid_text"], [4, 5, 6])
    df["cluster_id"] = model.predict(vectorizer.transform(df["hybrid_text"]))

    cluster_results = df[["record_id", "cluster_id"]].copy()
    cluster_results.to_csv(CLUSTER_RESULTS_PATH, index=False)

    top_terms_df = get_top_terms_per_cluster(model, vectorizer, top_n=12)
    interp_rows = []
    for cid in sorted(df["cluster_id"].unique()):
        top_terms = top_terms_df[top_terms_df["cluster_id"] == cid]["top_terms"].iloc[0]
        top_entities = get_top_entities_for_cluster(df, cid, top_n=8)
        interp_rows.append({
            "cluster_id": cid,
            "top_terms": top_terms,
            "top_entities": top_entities,
            "short_description": describe_cluster(top_terms, top_entities),
        })
    interp_df = pd.DataFrame(interp_rows)
    interp_df.to_csv(CLUSTER_INTERPRETATION_PATH, index=False)

    scores_df.to_csv(SILHOUETTE_SCORES_PATH, index=False)
    joblib.dump(model, CLUSTER_MODEL_PATH)
    joblib.dump(vectorizer, CLUSTER_VECTORIZER_PATH)
    joblib.dump(df, INTEGRATION_BASE_PATH)

    print("Saved:")
    print("- cluster_results.csv")
    print("- cluster_interpretation.csv")
    print("- silhouette_scores.csv")
    print("- cluster_model.pkl")
    print("- cluster_vectorizer.pkl")
    print("- integration_base.pkl")


if __name__ == "__main__":
    main()
