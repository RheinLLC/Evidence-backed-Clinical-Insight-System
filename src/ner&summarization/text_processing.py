"""
text_processing.py — Member B: NER + Extractive Summarization + Explainability Layer
======================================================================================
Intelligent Clinical EMR Chatbot System — Final Project

This module provides three core functions:
  1. extract_entities(text)      — Medical Named Entity Recognition
  2. generate_summary(text)      — 3-sentence Extractive Summarization (TF-IDF)
  3. format_evidence_layer(...)  — Explainability / Evidence Layer output

Dependencies:
  pip install pandas scikit-learn
  pip install scispacy
  pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz

When scispaCy model is not installed, the script falls back to a rule-based
medical entity extractor so the pipeline still runs end-to-end.
"""

import re
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import INTERIM_DATA_DIR, NER_SUMMARIZATION_DIR

# ---------------------------------------------------------------------------
# Attempt to load scispaCy; fall back to rule-based NER if unavailable
# ---------------------------------------------------------------------------
USE_SCISPACY = False
try:
    import spacy
    nlp = spacy.load("en_core_sci_sm")
    USE_SCISPACY = True
    print("[INFO] scispaCy model 'en_core_sci_sm' loaded successfully.")
except Exception:
    warnings.warn(
        "[WARN] scispaCy model not found — using rule-based medical NER fallback. "
        "Install with: pip install scispacy && pip install "
        "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/"
        "en_core_sci_sm-0.5.4.tar.gz"
    )

# ============================================================================
# 1. MEDICAL NAMED ENTITY RECOGNITION
# ============================================================================

# --- Curated medical lexicons for rule-based fallback & category filtering ---
DISEASE_KEYWORDS = [
    "disease", "disorder", "syndrome", "cancer", "carcinoma", "tumor", "tumour",
    "infection", "failure", "insufficiency", "stenosis", "aneurysm", "fibrosis",
    "cirrhosis", "hepatitis", "diabetes", "hypertension", "hypotension",
    "arrhythmia", "fibrillation", "thrombosis", "embolism", "pneumonia",
    "bronchitis", "asthma", "copd", "arthritis", "osteoporosis", "fracture",
    "stroke", "infarction", "ischemia", "anemia", "leukemia", "lymphoma",
    "melanoma", "epilepsy", "dementia", "alzheimer", "parkinson", "hernia",
    "appendicitis", "cholecystitis", "pancreatitis", "colitis", "gastritis",
    "coronary artery disease", "congestive heart failure", "atrial fibrillation",
    "deep vein thrombosis", "pulmonary embolism", "chronic kidney disease",
    "urinary tract infection", "gastroesophageal reflux", "obstructive sleep apnea",
    "mitral valve prolapse", "aortic stenosis",
]

SYMPTOM_KEYWORDS = [
    "pain", "ache", "fever", "cough", "nausea", "vomiting", "diarrhea",
    "constipation", "fatigue", "weakness", "dizziness", "headache", "dyspnea",
    "shortness of breath", "chest pain", "abdominal pain", "back pain",
    "swelling", "edema", "rash", "itching", "numbness", "tingling",
    "bleeding", "hematuria", "dysuria", "palpitations", "tachycardia",
    "bradycardia", "syncope", "seizure", "tremor", "insomnia", "anxiety",
    "depression", "confusion", "malaise", "anorexia", "weight loss",
    "weight gain", "blurred vision", "tinnitus", "dysphagia",
]

MEDICATION_KEYWORDS = [
    "aspirin", "ibuprofen", "acetaminophen", "tylenol", "motrin", "advil",
    "metformin", "insulin", "lisinopril", "amlodipine", "atenolol",
    "metoprolol", "losartan", "hydrochlorothiazide", "furosemide",
    "omeprazole", "pantoprazole", "atorvastatin", "simvastatin", "warfarin",
    "heparin", "enoxaparin", "clopidogrel", "prednisone", "dexamethasone",
    "amoxicillin", "azithromycin", "ciprofloxacin", "levofloxacin",
    "vancomycin", "morphine", "fentanyl", "hydrocodone", "oxycodone",
    "gabapentin", "pregabalin", "diazepam", "lorazepam", "alprazolam",
    "sertraline", "fluoxetine", "escitalopram", "duloxetine", "bupropion",
    "albuterol", "ipratropium", "fluticasone", "montelukast", "levothyroxine",
    "nitroglycerin", "digoxin", "diltiazem", "verapamil", "propofol",
    "lidocaine", "epinephrine", "norepinephrine", "dopamine", "dobutamine",
    "dilantin", "klonopin", "elavil", "thorazine", "neurontin", "phenergan",
    "methadone", "colchicine", "allopurinol", "tamsulosin", "finasteride",
]

# Regex patterns to catch common drug-name suffixes
MEDICATION_SUFFIX_PATTERN = re.compile(
    r"\b[A-Za-z]+(olol|pril|sartan|statin|mycin|cillin|prazole|dipine|"
    r"oxacin|barb|azepam|etine|pramine|amine|caine|nazole|afil|gliptin"
    r"|flozin|lukast|terol|parin|mab|nib|tinib)\b",
    re.IGNORECASE,
)


def _rule_based_extract(text: str) -> dict:
    """
    Rule-based medical entity extraction using curated keyword lists.
    Returns a dict with keys: diseases, symptoms, medications.
    """
    text_lower = text.lower()
    found = {"diseases": set(), "symptoms": set(), "medications": set()}

    for kw in DISEASE_KEYWORDS:
        if kw in text_lower:
            found["diseases"].add(kw.title())

    for kw in SYMPTOM_KEYWORDS:
        if kw in text_lower:
            found["symptoms"].add(kw.title())

    for kw in MEDICATION_KEYWORDS:
        if kw in text_lower:
            found["medications"].add(kw.title())

    # Catch additional medications by suffix pattern
    for match in MEDICATION_SUFFIX_PATTERN.finditer(text):
        found["medications"].add(match.group().title())

    return {k: sorted(v) for k, v in found.items()}


def _scispacy_extract(text: str) -> dict:
    """
    Use scispaCy en_core_sci_sm to extract entities, then categorize them
    into diseases, symptoms, and medications using keyword heuristics.
    """
    doc = nlp(text)
    raw_entities = list({ent.text.strip() for ent in doc.ents if len(ent.text.strip()) > 2})

    found = {"diseases": set(), "symptoms": set(), "medications": set()}
    text_lower = text.lower()

    for ent in raw_entities:
        ent_lower = ent.lower()
        # Try to categorize each extracted entity
        if any(kw in ent_lower for kw in DISEASE_KEYWORDS[:30]):
            found["diseases"].add(ent)
        elif any(kw in ent_lower for kw in SYMPTOM_KEYWORDS[:25]):
            found["symptoms"].add(ent)
        elif any(kw in ent_lower for kw in MEDICATION_KEYWORDS):
            found["medications"].add(ent)
        elif MEDICATION_SUFFIX_PATTERN.search(ent):
            found["medications"].add(ent)

    # Supplement with rule-based to improve recall
    rule_results = _rule_based_extract(text)
    for cat in found:
        found[cat] = found[cat].union(set(rule_results[cat]))

    return {k: sorted(v) for k, v in found.items()}


def extract_entities(text: str) -> dict:
    """
    Extract medical entities from a clinical note and categorize them into:
      - diseases
      - symptoms
      - medications

    Parameters
    ----------
    text : str
        A single clinical note / transcription string.

    Returns
    -------
    dict : {"diseases": [...], "symptoms": [...], "medications": [...]}
    """
    if not text or not isinstance(text, str) or text.strip() == "":
        return {"diseases": [], "symptoms": [], "medications": []}

    if USE_SCISPACY:
        return _scispacy_extract(text)
    else:
        return _rule_based_extract(text)


# ============================================================================
# 2. EXTRACTIVE SUMMARIZATION (TF-IDF based)
# ============================================================================

def _split_sentences(text: str) -> list:
    """
    Split clinical text into sentences using regex heuristics.
    Handles common abbreviations and section headers in EMR notes.
    """
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Split on period/question/exclamation followed by space + uppercase, or on newline
    raw = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    # Further split on comma-separated section headers (e.g., "HISTORY: , The patient...")
    sentences = []
    for seg in raw:
        sub = re.split(r',\s*(?=[A-Z][a-z])', seg)
        sentences.extend(sub)
    # Clean and filter
    sentences = [s.strip() for s in sentences if len(s.strip().split()) >= 5]
    return sentences


def generate_summary(text: str, num_sentences: int = 3) -> str:
    """
    Extractive summarization: select the top-N most important sentences
    from a clinical note using TF-IDF sentence scoring.

    Algorithm:
      1. Split text into sentences.
      2. Compute TF-IDF matrix over sentences.
      3. Score each sentence as the mean of its TF-IDF values.
      4. Return the top-N sentences in their original order.

    Parameters
    ----------
    text : str
        A single clinical note string.
    num_sentences : int
        Number of sentences to extract (default=3).

    Returns
    -------
    str : The extractive summary (top sentences joined by space).
    """
    if not text or not isinstance(text, str) or text.strip() == "":
        return ""

    sentences = _split_sentences(text)

    if len(sentences) <= num_sentences:
        return " ".join(sentences)

    # Build TF-IDF matrix over sentences
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    try:
        tfidf_matrix = vectorizer.fit_transform(sentences)
    except ValueError:
        # All sentences might be stop words only
        return " ".join(sentences[:num_sentences])

    # Score each sentence: mean TF-IDF value
    scores = np.asarray(tfidf_matrix.mean(axis=1)).flatten()

    # Get indices of top-scoring sentences
    top_indices = np.argsort(scores)[-num_sentences:]
    # Preserve original document order
    top_indices = sorted(top_indices)

    summary_sentences = [sentences[i] for i in top_indices]
    return " ".join(summary_sentences)


# ============================================================================
# 3. EXPLAINABILITY / EVIDENCE LAYER
# ============================================================================

def format_evidence_layer(
    text: str,
    predicted_specialty: str = "Placeholder",
    record_id: str = "N/A",
) -> dict:
    """
    Combine NER and summarization results into a structured Evidence Layer.
    This output enables clinicians to understand WHY a specialty was predicted.

    Parameters
    ----------
    text : str
        A single clinical note string.
    predicted_specialty : str
        The specialty predicted by the classification model (from Member A).
    record_id : str
        An optional record identifier.

    Returns
    -------
    dict : Structured evidence layer with summary, entities, and metadata.
    """
    summary = generate_summary(text)
    entities = extract_entities(text)

    evidence = {
        "record_id": record_id,
        "predicted_specialty": predicted_specialty,
        "evidence_summary": summary,
        "extracted_entities": entities,
        "entity_counts": {
            "diseases": len(entities.get("diseases", [])),
            "symptoms": len(entities.get("symptoms", [])),
            "medications": len(entities.get("medications", [])),
        },
        "note_length_words": len(text.split()) if text else 0,
    }
    return evidence


# ============================================================================
# 4. MOCK DATA GENERATION (Deliverable 2)
# ============================================================================

if __name__ == "__main__":
    # --- Configuration ---
    TEST_CSV = INTERIM_DATA_DIR / "test.csv"
    OUTPUT_FILE = NER_SUMMARIZATION_DIR / "member_B_mock_results.json"
    NUM_SAMPLES = 5

    # --- Load test data ---
    if not TEST_CSV.exists():
        raise FileNotFoundError(
            f"Cannot find '{TEST_CSV}'. "
            "Please make sure data/interim/test.csv exists."
        )

    df = pd.read_csv(TEST_CSV)
    print(f"[INFO] Loaded test.csv with {len(df)} records.")
    print(f"[INFO] Processing first {NUM_SAMPLES} records...\n")

    # --- Run evidence layer on first N records ---
    results = []
    for idx, row in df.head(NUM_SAMPLES).iterrows():
        text = str(row.get("transcription", ""))
        specialty = str(row.get("medical_specialty", "Unknown"))
        rec_id = str(row.get("record_id", f"ROW_{idx}"))

        evidence = format_evidence_layer(
            text=text,
            predicted_specialty=specialty,
            record_id=rec_id,
        )
        results.append(evidence)

        print(f"  [{rec_id}] Specialty: {specialty}")
        print(f"    Summary length: {len(evidence['evidence_summary'].split())} words")
        print(f"    Entities — Diseases: {evidence['entity_counts']['diseases']}, "
              f"Symptoms: {evidence['entity_counts']['symptoms']}, "
              f"Medications: {evidence['entity_counts']['medications']}")
        print()

    # --- Save to JSON ---
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"[DONE] Results saved to: {OUTPUT_FILE}")
