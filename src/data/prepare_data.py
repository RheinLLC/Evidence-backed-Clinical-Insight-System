import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def clean_text(text: str) -> str:
    """
    Clean transcription text by normalizing whitespace.
    """
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def count_words(text: str) -> int:
    """
    Count words in a text string.
    """
    if not text:
        return 0
    return len(text.split())


def main():
    # ========= 1. Config =========
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, "mtsamples.csv")
    output_dir = os.path.join(script_dir, "member1_outputs")
    top_n_specialties = 12
    min_word_threshold = 20
    random_state = 42

    os.makedirs(output_dir, exist_ok=True)

    # ========= 2. Check input file =========
    if not os.path.exists(input_file):
        raise FileNotFoundError(
            f"Cannot find input file: {input_file}\n"
            f"Please make sure 'mtsamples.csv' is in the same folder as prepare_data.py."
        )

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

    # ========= 5. Drop missing critical values =========
    before_missing_drop = len(df)
    df = df.dropna(subset=["transcription", "medical_specialty"])
    df = df[df["transcription"].str.strip() != ""]
    df = df[df["medical_specialty"].astype(str).str.strip() != ""]
    after_missing_drop = len(df)
    dropped_missing = before_missing_drop - after_missing_drop

    # ========= 6. Remove duplicates =========
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["transcription", "medical_specialty"]).reset_index(drop=True)
    after_dedup = len(df)
    dropped_duplicates = before_dedup - after_dedup

    # ========= 7. Remove abnormal records =========
    # Abnormal here means the transcription is too short to be useful
    df["word_count"] = df["transcription"].apply(count_words)

    before_abnormal_filter = len(df)
    df = df[df["word_count"] >= min_word_threshold].reset_index(drop=True)
    after_abnormal_filter = len(df)
    dropped_abnormal = before_abnormal_filter - after_abnormal_filter

    # ========= 8. Inspect specialty distribution =========
    specialty_counts_before_topn = df["medical_specialty"].value_counts()

    # Keep top N specialties only
    top_specialties = specialty_counts_before_topn.head(top_n_specialties).index.tolist()
    df = df[df["medical_specialty"].isin(top_specialties)].reset_index(drop=True)

    # Recalculate counts after filtering
    specialty_counts = df["medical_specialty"].value_counts()

    # ========= 9. Standardize final columns =========
    df = df[["transcription", "medical_specialty", "keywords"]].copy()
    df.insert(0, "record_id", [f"REC_{i:05d}" for i in range(1, len(df) + 1)])

    # ========= 10. Save cleaned dataset =========
    cleaned_path = os.path.join(output_dir, "cleaned_dataset.csv")
    df.to_csv(cleaned_path, index=False)

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

    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "val.csv")
    test_path = os.path.join(output_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    # ========= 12. Plot label distribution =========
    plt.figure(figsize=(14, 6))
    specialty_counts.plot(kind="bar")
    plt.title("Label Distribution of Selected Medical Specialties")
    plt.xlabel("Medical Specialty")
    plt.ylabel("Number of Records")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "label_distribution.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

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

    note_path = os.path.join(output_dir, "data_note.txt")
    with open(note_path, "w", encoding="utf-8") as f:
        f.write("\n".join(note_lines))

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