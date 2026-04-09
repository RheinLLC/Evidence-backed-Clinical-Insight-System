import os
import re
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.pipeline import Pipeline

from src.config import (
    CLASSIFICATION_MODELS_DIR,
    CLASSIFICATION_VECTORIZER_PATH,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
)


# Clean raw text consistently across transcription and NER-derived features.
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Load train/validation/test splits plus the exported NER results.
def load_data(project_root):
    train_path = INTERIM_DATA_DIR / "train.csv"
    val_path = INTERIM_DATA_DIR / "val.csv"
    test_path = INTERIM_DATA_DIR / "test.csv"

    ner_path_1 = PROCESSED_DATA_DIR / "ner results.csv"
    ner_path_2 = PROCESSED_DATA_DIR / "ner_results.csv"

    print("Checking input files...", flush=True)
    print(f"train_path = {train_path}", flush=True)
    print(f"val_path   = {val_path}", flush=True)
    print(f"test_path  = {test_path}", flush=True)

    if not train_path.exists():
        raise FileNotFoundError(f"train.csv not found: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"val.csv not found: {val_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"test.csv not found: {test_path}")

    if ner_path_1.exists():
        ner_path = ner_path_1
    elif ner_path_2.exists():
        ner_path = ner_path_2
    else:
        raise FileNotFoundError(
            f"NER file not found. Checked:\n{ner_path_1}\n{ner_path_2}"
        )

    print(f"ner_path   = {ner_path}", flush=True)

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    ner_df = pd.read_csv(ner_path)

    print("Loaded CSV files successfully.", flush=True)
    print(f"train_df columns = {list(train_df.columns)}", flush=True)
    print(f"val_df columns   = {list(val_df.columns)}", flush=True)
    print(f"test_df columns  = {list(test_df.columns)}", flush=True)
    print(f"ner_df columns   = {list(ner_df.columns)}", flush=True)

    return train_df, val_df, test_df, ner_df


# Convert entity columns into a single text feature keyed by record_id.
def prepare_ner_features(ner_df):
    ner_df = ner_df.copy()

    if "record_id" not in ner_df.columns:
        raise ValueError("NER file must contain 'record_id' column.")

    for col in ["diseases", "symptoms", "medications"]:
        if col not in ner_df.columns:
            print(f"Warning: '{col}' not found in ner_df, filling with empty string.", flush=True)
            ner_df[col] = ""

    ner_df["record_id"] = ner_df["record_id"].astype(str).str.strip()
    ner_df["diseases"] = ner_df["diseases"].fillna("").astype(str)
    ner_df["symptoms"] = ner_df["symptoms"].fillna("").astype(str)
    ner_df["medications"] = ner_df["medications"].fillna("").astype(str)

    ner_df["entity_text"] = (
        ner_df["diseases"] + " " +
        ner_df["symptoms"] + " " +
        ner_df["medications"]
    ).apply(clean_text)

    ner_feature_df = ner_df[["record_id", "entity_text"]].drop_duplicates(subset=["record_id"]).reset_index(drop=True)

    print("NER feature preparation done.", flush=True)
    print(f"ner_feature_df size = {len(ner_feature_df)}", flush=True)

    return ner_feature_df


# Join structured NER features back onto each dataset split.
def merge_features(base_df, ner_feature_df, df_name="dataset"):
    df = base_df.copy()

    required_cols = ["record_id", "transcription", "medical_specialty"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"{df_name} missing required column: {col}")

    df["record_id"] = df["record_id"].astype(str).str.strip()
    df["transcription"] = df["transcription"].apply(clean_text)
    df["medical_specialty"] = df["medical_specialty"].astype(str).str.strip()

    before_merge = len(df)
    df = df.merge(ner_feature_df, on="record_id", how="left")
    after_merge = len(df)

    df["entity_text"] = df["entity_text"].fillna("").apply(clean_text)
    df["hybrid_text"] = (df["transcription"] + " " + df["entity_text"]).apply(clean_text)

    print(f"{df_name}: merge done | before={before_merge}, after={after_merge}", flush=True)
    print(
        f"{df_name}: empty entity_text rows = {(df['entity_text'].str.len() == 0).sum()}",
        flush=True
    )

    return df


# Define the candidate classifiers compared under the same feature pipeline.
def build_models():
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ),
        "linear_svc": LinearSVC(
            class_weight="balanced",
            random_state=42
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=80,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        )
    }
    return models


# Compute the summary metrics used for model selection.
def evaluate_predictions(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "accuracy": acc,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1
    }


# Train each candidate model for one feature type and keep the best performer.
def train_and_select_best(train_df, val_df, feature_col):
    models = build_models()
    results = []
    fitted_objects = {}

    print(f"\n===== feature: {feature_col} =====", flush=True)

    for model_name, clf in models.items():
        print(f"start model: {model_name}", flush=True)

        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=8000, ngram_range=(1, 2), min_df=2)),
            ("clf", clf)
        ])

        pipeline.fit(train_df[feature_col], train_df["medical_specialty"])
        print(f"fit done: {model_name}", flush=True)

        val_pred = pipeline.predict(val_df[feature_col])
        print(f"predict done: {model_name}", flush=True)

        metrics = evaluate_predictions(val_df["medical_specialty"], val_pred)
        metrics["feature_type"] = feature_col
        metrics["model_name"] = model_name
        results.append(metrics)

        fitted_objects[(feature_col, model_name)] = pipeline

        print(
            f"done model: {model_name}, macro_f1={metrics['macro_f1']:.4f}",
            flush=True
        )

    results_df = pd.DataFrame(results).sort_values(
        by="macro_f1", ascending=False
    ).reset_index(drop=True)

    best_row = results_df.iloc[0]
    best_key = (best_row["feature_type"], best_row["model_name"])
    best_pipeline = fitted_objects[best_key]

    print(f"best for {feature_col}: {best_row['model_name']}", flush=True)

    return results_df, best_pipeline, best_row


# Persist a readable confusion matrix figure for the final test predictions.
def save_confusion_matrix(y_true, y_pred, labels, output_path):
    fig, ax = plt.subplots(figsize=(14, 12))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(y_true, y_pred, labels=labels),
        display_labels=labels
    )
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
    plt.title("Confusion Matrix on Test Set")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    # Initialize paths and ensure output directories exist before training.
    print("STEP 1: enter main()", flush=True)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

    processed_dir = PROCESSED_DATA_DIR
    model_dir = CLASSIFICATION_MODELS_DIR
    vectorizer_path = CLASSIFICATION_VECTORIZER_PATH

    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(vectorizer_path.parent, exist_ok=True)

    print("STEP 2: paths ready", flush=True)
    print(f"project_root = {project_root}", flush=True)
    print(f"processed_dir = {processed_dir}", flush=True)
    print(f"model_dir = {model_dir}", flush=True)

    # Load all required tabular inputs and prepare derived NER text features.
    print("STEP 3: start loading data", flush=True)
    train_df, val_df, test_df, ner_df = load_data(project_root)
    print("STEP 4: data loaded", flush=True)
    print(f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}, ner={len(ner_df)}", flush=True)

    print("STEP 5: prepare ner features", flush=True)
    ner_feature_df = prepare_ner_features(ner_df)
    print("STEP 6: ner features ready", flush=True)

    # Merge base text with NER-derived features for every dataset split.
    print("STEP 7: merge train", flush=True)
    train_df = merge_features(train_df, ner_feature_df, "train_df")
    print("STEP 8: merge val", flush=True)
    val_df = merge_features(val_df, ner_feature_df, "val_df")
    print("STEP 9: merge test", flush=True)
    test_df = merge_features(test_df, ner_feature_df, "test_df")
    print("STEP 10: merge done", flush=True)

    # Benchmark separate feature representations and collect their validation metrics.
    all_result_tables = []
    candidate_best = []

    print("STEP 11: start training transcription", flush=True)
    result_df, best_pipeline, best_row = train_and_select_best(train_df, val_df, "transcription")
    all_result_tables.append(result_df)
    candidate_best.append((best_row, best_pipeline))

    print("STEP 12: start training entity_text", flush=True)
    result_df, best_pipeline, best_row = train_and_select_best(train_df, val_df, "entity_text")
    all_result_tables.append(result_df)
    candidate_best.append((best_row, best_pipeline))

    print("STEP 13: start training hybrid_text", flush=True)
    result_df, best_pipeline, best_row = train_and_select_best(train_df, val_df, "hybrid_text")
    all_result_tables.append(result_df)
    candidate_best.append((best_row, best_pipeline))

    # Compare the best candidate from each feature representation.
    print("STEP 14: combine results", flush=True)
    all_results_df = pd.concat(all_result_tables, ignore_index=True)
    all_results_df = all_results_df.sort_values(by="macro_f1", ascending=False).reset_index(drop=True)

    best_overall_row = all_results_df.iloc[0]
    best_feature = best_overall_row["feature_type"]
    best_model_name = best_overall_row["model_name"]

    print(f"STEP 15: best setting found -> {best_feature} + {best_model_name}", flush=True)

    best_pipeline = None
    for row, pipeline in candidate_best:
        if row["feature_type"] == best_feature and row["model_name"] == best_model_name:
            best_pipeline = pipeline
            break

    if best_pipeline is None:
        raise ValueError("best_pipeline is None. Failed to match best model.")

    # Run a final evaluation on the held-out test set with the chosen pipeline.
    print("STEP 16: start final test prediction", flush=True)
    test_pred = best_pipeline.predict(test_df[best_feature])

    test_metrics = evaluate_predictions(test_df["medical_specialty"], test_pred)
    labels = sorted(test_df["medical_specialty"].unique())

    # Save prediction outputs, comparison tables, and a markdown summary report.
    print("STEP 17: save classification_results.csv", flush=True)
    results_df = pd.DataFrame({
        "record_id": test_df["record_id"],
        "true_label": test_df["medical_specialty"],
        "predicted_label": test_pred
    })
    results_df.to_csv(processed_dir / "classification_results.csv", index=False)

    print("STEP 18: save comparison csv", flush=True)
    all_results_df.to_csv(processed_dir / "classification_model_comparison.csv", index=False)

    print("STEP 19: save metrics md", flush=True)
    report_text = []
    report_text.append("# Classification Metrics")
    report_text.append("")
    report_text.append("## Best Validation Setting")
    report_text.append(f"- Feature type: {best_feature}")
    report_text.append(f"- Model: {best_model_name}")
    report_text.append("")
    report_text.append("## Test Set Performance")
    report_text.append(f"- Accuracy: {test_metrics['accuracy']:.4f}")
    report_text.append(f"- Macro Precision: {test_metrics['macro_precision']:.4f}")
    report_text.append(f"- Macro Recall: {test_metrics['macro_recall']:.4f}")
    report_text.append(f"- Macro F1: {test_metrics['macro_f1']:.4f}")
    report_text.append("")
    report_text.append("## Detailed Classification Report")
    report_text.append("")
    report_text.append("```")
    report_text.append(classification_report(
        test_df["medical_specialty"],
        test_pred,
        zero_division=0
    ))
    report_text.append("```")

    with open(processed_dir / "classification_metrics.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report_text))

    # Export the confusion matrix and serialized artifacts needed by the app pipeline.
    print("STEP 20: save confusion matrix", flush=True)
    save_confusion_matrix(
        test_df["medical_specialty"],
        test_pred,
        labels,
        processed_dir / "confusion_matrix.png"
    )

    print("STEP 21: save model", flush=True)
    joblib.dump(best_pipeline, model_dir / "best_classifier.pkl")

    print("STEP 22: save vectorizer", flush=True)
    tfidf_vectorizer = best_pipeline.named_steps["tfidf"]
    joblib.dump(tfidf_vectorizer, vectorizer_path)

    print("STEP 23: finished", flush=True)
    print("Outputs saved:", flush=True)
    print("- data/processed/classification_results.csv", flush=True)
    print("- data/processed/classification_model_comparison.csv", flush=True)
    print("- data/processed/classification_metrics.md", flush=True)
    print("- data/processed/confusion_matrix.png", flush=True)
    print("- models/classification/best_classifier.pkl", flush=True)
    print("- models/vectorizers/classification_vectorizer.pkl", flush=True)


if __name__ == "__main__":
    main()
