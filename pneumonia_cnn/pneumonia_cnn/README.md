# Pneumonia Detection from Chest X-rays using CNN

This project builds a deep learning classifier to detect pneumonia in chest radiographs.  
The focus is not only predictive accuracy but also understanding model behavior under class imbalance.

---

## Model
We use **ResNet18** with ImageNet pretraining.  
The final fully connected layer is adapted for binary classification (NORMAL vs PNEUMONIA).

---

## Dataset
Chest X-Ray Images (Pneumonia), using the official train / validation / test splits.

---

## Training Methodology

### Baseline
Standard cross-entropy loss.

### Improvement
We observed that the baseline model heavily favored pneumonia due to imbalance.  
To correct this, we introduced **class-weighted cross entropy**, penalizing mistakes on normal cases more strongly.

Model selection is based strictly on validation accuracy.

---

## Test Results

| Model | Accuracy | Precision | Recall | AUC |
|------|------|------|------|------|
| Baseline | 0.62 | 0.63 | 0.98 | 0.57 |
| Weighted | **0.82** | **0.78** | **0.99** | **0.94** |

---

## Key Insight
Without rebalancing, the model predicts pneumonia for most patients.  
After weighting, specificity improves dramatically while maintaining very high sensitivity.  
The AUC jump demonstrates far better class separation.

---

## Repository Structure

