from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
CLASSIFICATION_MODELS_DIR = MODELS_DIR / "classification"
CLUSTERING_MODELS_DIR = MODELS_DIR / "clustering"
VECTORIZERS_DIR = MODELS_DIR / "vectorizers"
CLASSIFICATION_VECTORIZER_PATH = VECTORIZERS_DIR / "classification_vectorizer.pkl"
CLUSTER_VECTORIZER_PATH = VECTORIZERS_DIR / "cluster_vectorizer.pkl"

SRC_DIR = PROJECT_ROOT / "src"
NER_SUMMARIZATION_DIR = SRC_DIR / "ner&summarization"
PIPELINE_DIR = SRC_DIR / "pipeline"
