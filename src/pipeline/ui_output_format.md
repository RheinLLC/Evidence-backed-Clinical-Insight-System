# UI Output Format

## Input
- Raw EMR note pasted by the user.

## Current Output Sections
1. **Summary**
   - 3-sentence extractive summary from Member B module.
2. **Extracted Entities**
   - diseases
   - symptoms
   - medications
3. **Cluster Insight**
   - cluster_id
   - short description
   - top_terms
   - top_entities
4. **Specialty Prediction**
   - reserved for Member 1 classification integration
5. **Evidence Note**
   - concise natural-language rationale combining extracted entities and cluster interpretation

## Expected JSON Schema

```json
{
  "summary": "string",
  "entities": {
    "diseases": ["..."],
    "symptoms": ["..."],
    "medications": ["..."]
  },
  "cluster": {
    "cluster_id": 0,
    "top_terms": "term1; term2; term3",
    "top_entities": "entity1; entity2; entity3",
    "description": "Likely cardiovascular-related cases"
  },
  "specialty": {
    "predicted_label": null,
    "confidence": null,
    "status": "pending member1 integration"
  },
  "evidence_note": "This cluster assignment is mainly supported by ..."
}
```

## Clustering Choice
- Primary algorithm: **MiniBatchKMeans**
- Final chosen `k`: **4**
- Feature input: **hybrid_text = entity_text_all + summary**
- Support metric: silhouette score saved in `silhouette_scores.csv`

## Pending Integration
- Member 1 should connect:
  - `best_classifier.pkl`
  - `vectorizer.pkl`
  - `predict_specialty()` implementation in `demo_pipeline_v1.py`
- Member 2 may further refine:
  - evidence note wording
  - summary/entity presentation order