# Evidence-backed-Clinical-Insight-System

Evidence-Backed Clinical Insight Assistant built with Streamlit. The app combines:

- extractive summarization
- medical entity extraction
- cluster-based cohort insight
- specialty classification
- evidence-backed explanation text

## Project Structure

- `main.py`: Streamlit Community Cloud entrypoint
- `src/app/app.py`: Streamlit UI
- `src/pipeline/demo_pipeline.py`: unified inference pipeline
- `src/classification/train_classifier.py`: classification training script
- `src/clustering/clustering_train.py`: clustering training script
- `src/data/prepare_data.py`: data cleaning and split generation
- `src/ner&summarization/text_processing.py`: NER and summarization logic
- `data/interim/`: cleaned dataset and train/val/test splits
- `data/processed/`: generated evaluation outputs and interpretation files
- `models/classification/`: classifier artifact
- `models/clustering/`: clustering model artifacts
- `models/vectorizers/`: vectorizer artifacts
- `demo/demo_inputs/`: demo sample inputs
- `demo/demo_outputs/`: demo sample outputs

## Local Run

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the app:

```bash
streamlit run src/app/app.py
```

## Streamlit Community Cloud Deployment

Use these settings:

- Repository: this repo
- Branch: your deploy branch
- Main file path: `src/app/app.py`

The app is configured so Streamlit can launch directly from `src/app/app.py`.

## Required Runtime Assets

The deployed app expects these files to already exist in the repository:

- `data/interim/test.csv`
- `data/processed/ner results.csv`
- `data/processed/classification_results.csv`
- `data/processed/cluster_interpretation.csv`
- `models/classification/best_classifier.pkl`
- `models/clustering/cluster_model.pkl`
- `models/vectorizers/classification_vectorizer.pkl`
- `models/vectorizers/cluster_vectorizer.pkl`

Optional:

- `src/ner&summarization/member_B_mock_results.json`

If some model or data assets are missing, parts of the app will degrade gracefully, but related predictions or insights may be unavailable.

## Notes

- `spacy` and `scispacy` are not required for deployment. If they are unavailable, the NER module falls back to a rule-based extractor.
- Large model or data files may affect Streamlit Community Cloud startup time and repository size limits.
