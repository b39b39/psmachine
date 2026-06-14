# 앙상블 모델 결과 요약 — SSU Problem Solving 2026

생성일: 2026-06-02  |  random_state=42


## (a) 모델별 CV 성능 비교

| Model | CV AUC | ±std | Gap | CV Acc | Bal.Acc | F1-macro |
|-------|--------|------|-----|--------|---------|----------|
| Tuned ExtraTrees ★ | 0.6668 | 0.0105 | 0.3318 | 0.7385 | 0.6530 | 0.6613 |
| Base ExtraTrees | 0.6635 | 0.0152 | 0.3365 | 0.7379 | 0.6516 | 0.6596 |
| Stacking | 0.6602 | 0.0145 | nan | 0.7420 | 0.6597 | 0.6691 |
| SoftVoting | 0.6599 | 0.0176 | 0.3395 | 0.7401 | 0.6593 | 0.6685 |
| Base LightGBM | 0.6573 | 0.0151 | 0.3427 | 0.7050 | 0.6393 | 0.6454 |
| Base RandomForest | 0.6572 | 0.0113 | 0.3428 | 0.7405 | 0.6564 | 0.6653 |
| Base XGBoost | 0.6546 | 0.0143 | 0.3454 | 0.6990 | 0.6336 | 0.6391 |
| Base HistGB | 0.6524 | 0.0190 | 0.3476 | 0.7251 | 0.6447 | 0.6519 |

★ = Champion 모델


## (b) 최적 하이퍼파라미터

- 모델: **ExtraTrees** | Tier: **ALL34** | Feature 수: 32
- `n_estimators` = `600`
- `min_samples_leaf` = `1`
- `max_features` = `sqrt`
- `max_depth` = `15`
- `class_weight` = `None`


## (c) Feature Importance 상위 10개

| 순위 | Feature | Importance |
|------|---------|------------|
| 1 | feat_01 | 0.1570 |
| 2 | feat_05 ★ | 0.0856 |
| 3 | feat_31 ★ | 0.0713 |
| 4 | feat_07 ★ | 0.0435 |
| 5 | feat_02 ★ | 0.0282 |
| 6 | feat_23 | 0.0249 |
| 7 | feat_10 | 0.0249 |
| 8 | feat_19 | 0.0247 |
| 9 | feat_03 | 0.0246 |
| 10 | feat_18 | 0.0246 |

★ = CORE6 feature


## (d) OOF Train 평가 (answer2.csv)

- 임계값: 0.42 (Balanced Accuracy 기준 최적)
- OOF AUC: 0.6665
- OOF Accuracy: 0.7403
- OOF Balanced-Acc: 0.6612
- OOF F1-macro: 0.6707


## (e) 다음 단계 관찰 메모

- 앙상블(RF/ET/HistGB)이 단일 DT 대비 CV AUC 유의미하게 향상.
- Soft Voting과 Stacking의 AUC를 비교해 최적 방식 결정.
- CORE6/CORE7 vs ALL34 차이가 작으면 소규모 tier 선호 (과적합 위험 감소).
- 다음: feature engineering (interaction, binning) 또는 optuna 튜닝.