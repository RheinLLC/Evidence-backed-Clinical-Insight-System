import re

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from src.config import INTERIM_DATA_DIR, RAW_DATA_DIR


# Normalize free-text fields before downstream modeling.
def clean_text(text: str) -> str:
    """
    Clean transcription text by normalizing whitespace.
    """
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Support record filtering by note length.
def count_words(text: str) -> int:
    """
    Count words in a text string.
    """
    if not text:
        return 0
    return len(text.split())


def main():
    # Define the input source, output folder, and filtering thresholds for preprocessing.
    # ========= 1. Config =========
    input_file = RAW_DATA_DIR / "mtsamples.csv"
    output_dir = INTERIM_DATA_DIR
    top_n_specialties = 12
    min_word_threshold = 20
    random_state = 42

    output_dir.mkdir(parents=True, exist_ok=True)

    # Stop early when the expected raw CSV is missing.
    # ========= 2. Check input file =========
    if not input_file.exists():
        raise FileNotFoundError(
            f"Cannot find input file: {input_file}\n"
            "Please make sure 'mtsamples.csv' is under data/raw."
        )

    # Load the raw dataset and validate the columns needed by later stages.
    # ========= 3. Load data =========
    df = pd.read_csv(input_file)

    # Keep only columns needed for this stage
    required_cols = ["transcription", "medical_specialty", "keywords"]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not found in dataset.\n"
                f"Available columns: {list(df.columns)}"
            )

    df = df[required_cols].copy()
    original_count = len(df)

    # Standardize whitespace and categorical text fields before filtering.
    # ========= 4. Clean key fields =========
    df["transcription"] = df["transcription"].apply(clean_text)
    df["medical_specialty"] = df["medical_specialty"].apply(
        lambda x: x if pd.isna(x) else str(x).strip()
    )
    df["keywords"] = df["keywords"].fillna("").astype(str).str.strip()

    # Normalize invalid specialty strings
    df["medical_specialty"] = df["medical_specialty"].replace(
        ["nan", "None", "", " "], pd.NA
    )

    # Remove rows that are unusable because critical inputs are blank.
    # ========= 5. Drop missing critical values =========
    before_missing_drop = len(df)
    df = df.dropna(subset=["transcription", "medical_specialty"])
    df = df[df["transcription"].str.strip() != ""]
    df = df[df["medical_specialty"].astype(str).str.strip() != ""]
    after_missing_drop = len(df)
    dropped_missing = before_missing_drop - after_missing_drop

    # Drop exact duplicate note-specialty pairs to reduce training noise.
    # ========= 6. Remove duplicates =========
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["transcription", "medical_specialty"]).reset_index(drop=True)
    after_dedup = len(df)
    dropped_duplicates = before_dedup - after_dedup

    # Remove very short notes because they carry too little clinical context.
    # ========= 7. Remove abnormal records =========
    # Abnormal here means the transcription is too short to be useful
    df["word_count"] = df["transcription"].apply(count_words)

    before_abnormal_filter = len(df)
    df = df[df["word_count"] >= min_word_threshold].reset_index(drop=True)
    after_abnormal_filter = len(df)
    dropped_abnormal = before_abnormal_filter - after_abnormal_filter

    # Keep the most common specialties to create a balanced multi-class task.
    # ========= 8. Inspect specialty distribution =========
    specialty_counts_before_topn = df["medical_specialty"].value_counts()

    # Keep top N specialties only
    top_specialties = specialty_counts_before_topn.head(top_n_specialties).index.tolist()
    df = df[df["medical_specialty"].isin(top_specialties)].reset_index(drop=True)

    # Recalculate counts after filtering
    specialty_counts = df["medical_specialty"].value_counts()

    # Rebuild the final table schema and attach stable record identifiers.
    # ========= 9. Standardize final columns =========
    df = df[["transcription", "medical_specialty", "keywords"]].copy()
    df.insert(0, "record_id", [f"REC_{i:05d}" for i in range(1, len(df) + 1)])

    # Persist the cleaned master dataset for reuse by later modules.
    # ========= 10. Save cleaned dataset =========
    cleaned_path = output_dir / "cleaned_dataset.csv"
    df.to_csv(cleaned_path, index=False)

    # Create stratified train/validation/test splits for reproducible experiments.
    # ========= 11. Train / val / test split =========
    # 70 / 15 / 15 stratified split
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df["medical_specialty"],
        random_state=random_state
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["medical_specialty"],
        random_state=random_state
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"
    test_path = output_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    # Visualize the retained label distribution for reporting and sanity checks.
    # ========= 12. Plot label distribution =========
    plt.figure(figsize=(14, 6))
    specialty_counts.plot(kind="bar")
    plt.title("Label Distribution of Selected Medical Specialties")
    plt.xlabel("Medical Specialty")
    plt.ylabel("Number of Records")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    plot_path = output_dir / "label_distribution.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()

    # Write a human-readable note summarizing the preprocessing decisions.
    # ========= 13. Write data note =========
    note_lines = []
    note_lines.append("Data Preparation Note")
    note_lines.append("====================")
    note_lines.append(f"Original sample count: {original_count}")
    note_lines.append(f"Dropped due to missing critical fields: {dropped_missing}")
    note_lines.append(f"Dropped duplicate records: {dropped_duplicates}")
    note_lines.append(f"Dropped abnormal short records (< {min_word_threshold} words): {dropped_abnormal}")
    note_lines.append(f"Final sample count after cleaning and top-{top_n_specialties} filtering: {len(df)}")
    note_lines.append("")
    note_lines.append(f"Selected specialties (top {top_n_specialties}):")

    for label, count in specialty_counts.items():
        note_lines.append(f"- {label}: {count}")

    note_lines.append("")
    note_lines.append("Split ratio:")
    note_lines.append("- Train: 70%")
    note_lines.append("- Validation: 15%")
    note_lines.append("- Test: 15%")
    note_lines.append("")
    note_lines.append(f"Train size: {len(train_df)}")
    note_lines.append(f"Validation size: {len(val_df)}")
    note_lines.append(f"Test size: {len(test_df)}")

    note_path = output_dir / "data_note.txt"
    with open(note_path, "w", encoding="utf-8") as f:
        f.write("\n".join(note_lines))

    # Print output locations and dataset sizes for quick command-line verification.
    # ========= 14. Print summary =========
    print("Done.")
    print(f"Input file: {input_file}")
    print(f"Output folder: {output_dir}")
    print(f"Cleaned dataset saved to: {cleaned_path}")
    print(f"Train file saved to: {train_path}")
    print(f"Validation file saved to: {val_path}")
    print(f"Test file saved to: {test_path}")
    print(f"Label distribution plot saved to: {plot_path}")
    print(f"Data note saved to: {note_path}")
    print("")
    print(f"Train size: {len(train_df)}")
    print(f"Validation size: {len(val_df)}")
    print(f"Test size: {len(test_df)}")
    print("")
    print("Selected specialties:")
    print(specialty_counts)


if __name__ == "__main__":
    main()
