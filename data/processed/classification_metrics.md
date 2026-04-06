# Classification Metrics

## Best Validation Setting
- Feature type: hybrid_text
- Model: logistic_regression

## Test Set Performance
- Accuracy: 0.4031
- Macro Precision: 0.3945
- Macro Recall: 0.4618
- Macro F1: 0.4145

## Detailed Classification Report

```
                               precision    recall  f1-score   support

   Cardiovascular / Pulmonary       0.36      0.44      0.39        55
   Consult - History and Phy.       0.43      0.34      0.38        77
            Discharge Summary       0.57      0.81      0.67        16
             Gastroenterology       0.43      0.45      0.44        33
             General Medicine       0.33      0.38      0.36        39
                    Neurology       0.27      0.24      0.25        33
      Obstetrics / Gynecology       0.38      0.65      0.48        23
                   Orthopedic       0.38      0.54      0.44        52
                    Radiology       0.29      0.34      0.31        41
SOAP / Chart / Progress Notes       0.23      0.38      0.29        24
                      Surgery       0.61      0.31      0.41       162
                      Urology       0.47      0.65      0.55        23

                     accuracy                           0.40       578
                    macro avg       0.39      0.46      0.41       578
                 weighted avg       0.44      0.40      0.40       578

```