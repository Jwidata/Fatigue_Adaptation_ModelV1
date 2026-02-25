# Model Training Report

Dataset windows: 36483
Feature columns: 15
Participants: 28

## Train/Test Split

Train participants (22): ID02, ID03, ID04, ID05, ID06, ID07, ID08, ID11, ID12, ID14, ID15, ID16, ID17, ID18, ID19, ID20, ID21, ID23, ID24, ID25, ID27, ID28
Test participants (6): ID01, ID09, ID10, ID13, ID22, ID26

## Cross-Validation Results (Train Only)

| Feature Set | Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- | --- |
| pupil_only | logreg | 0.5570 ± 0.0219 | 0.6300 ± 0.0265 | 0.5752 ± 0.1015 | 0.5943 ± 0.0574 | 0.5843 ± 0.0211 |
| pupil_only | rf | 0.5379 ± 0.0130 | 0.5869 ± 0.0077 | 0.6677 ± 0.0171 | 0.6247 ± 0.0116 | 0.5161 ± 0.0200 |
| pupil_blink | logreg | 0.5638 ± 0.0235 | 0.6430 ± 0.0335 | 0.5570 ± 0.0737 | 0.5930 ± 0.0398 | 0.5772 ± 0.0175 |
| pupil_blink | rf | 0.5330 ± 0.0283 | 0.5832 ± 0.0218 | 0.6627 ± 0.0321 | 0.6204 ± 0.0258 | 0.5092 ± 0.0378 |
| all | logreg | 0.5610 ± 0.0291 | 0.6413 ± 0.0407 | 0.5556 ± 0.0828 | 0.5900 ± 0.0483 | 0.5797 ± 0.0271 |
| all | rf | 0.5434 ± 0.0184 | 0.5956 ± 0.0192 | 0.6581 ± 0.0965 | 0.6208 ± 0.0415 | 0.5251 ± 0.0299 |

## Selected Model

Best combo: pupil_only + logreg

## Dropped Features

- aoi_switch_rate_per_min
- aoi_entropy
- aoi_top_frac

## Holdout Test Metrics

| Metric | Value |
| --- | --- |
| accuracy | 0.5737 |
| precision | 0.7254 |
| recall | 0.4269 |
| f1 | 0.5375 |
| roc_auc | 0.6197 |

Confusion matrix image: `confusion_matrix.png`
