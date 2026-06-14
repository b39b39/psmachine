# 작업 지시: Decision Tree 모델링 (SSU Problem Solving 2026)

## 0. 컨텍스트 (이미 끝난 분석 — 다시 하지 말 것)

`train.csv`는 10,000행 × (`id` + 38개 feature + `target`) 구조의 이진분류 데이터다.
사전 EDA와 3개 LLM 교차검증, permutation 검정, paired CV 검증까지 모두 완료했다.
확정된 사실은 다음과 같다. **아래 결론은 검증을 거쳤으니 재분석 없이 신뢰하고 사용할 것.**

- 결측치 0개, 완전 중복 행 0개.
- `target` 분포: 0이 65.68%, 1이 34.32% (불균형). **base rate 정확도 = 0.657** → accuracy 단독 평가 금지.
- 수치형 24개(`feat_01`~`feat_24`), 범주형 14개(`feat_25`~`feat_38`).
- **이상치 제거·clipping 금지**: 통계적 "이상치"는 비대칭 분포의 정상 꼬리이거나 소수클래스(target=1) 신호와 겹친다. 특히 `feat_01`은 가장 강한 feature이므로 절대 손대지 말 것.
- ID성/암기유발 feature 4개는 무조건 제거 (근거: 무제한 트리에서 train acc=1.0 / CV acc는 base rate 미만, 단일 AUC≈0.50):
  - `id`, `feat_34`(고유 ID_xxxxx 10000개), `feat_08`(0~9999 완전순열=행인덱스), `feat_14`(고카디널리티 균등난수).

## 1. 확정된 Feature Tier (이걸 코드 상단 상수로 박아둘 것)

```python
DROP_ID_LIKE = ['id', 'feat_34', 'feat_08', 'feat_14']   # 항상 제거

# 여러 기준(MI, permutation p값, paired CV)에서 반복 확인된 핵심 신호
CORE6 = ['feat_01', 'feat_02', 'feat_05', 'feat_07', 'feat_31', 'feat_36']

# CORE6 + feat_03. feat_03은 약한 '비선형' extended candidate:
#  - permutation p≈0.023 (단일검정 통과, Bonferroni 38개 보정은 미통과 → 핵심 아님)
#  - 트리계열 paired CV에서 +0.007~0.011 일관된 효과 (ExtraTrees paired p=0.0067)
#  - 선형모델에선 기여 0 → 비선형 신호. 트리 모델엔 넣을 가치 있음.
CORE7 = CORE6 + ['feat_03']

# 나머지(feat_10/11/12/17/19/21 등)는 permutation 검정에서 noise와 구분 안 됨.
```

참고: `feat_31`, `feat_36`은 범주형(object)이다 → 인코딩 필요. CORE6/CORE7의 나머지는 수치형.

## 2. 데이터 로드

로컬에 `train.csv`가 있으면 그대로 사용. 없으면 아래로 받는다.
```python
import kagglehub
path = kagglehub.competition_download('ssu-problem-solving-2026')
print("Path to competition files:", path)
# path 안의 train.csv (및 test.csv가 있으면 그것도) 사용
```
- `test.csv`(또는 대회 제출용 파일)가 함께 있으면 경로를 탐지해 함께 로드하고, 제출 포맷(sample_submission 등)이 있으면 그 컬럼 구조를 맞춰 예측 결과를 저장하는 코드까지 작성할 것. 없으면 train만으로 진행.

## 3. 이번 작업의 목표: Decision Tree

가장 먼저 **단일 Decision Tree** (`sklearn.tree.DecisionTreeClassifier`)로 baseline을 세운다.
세 가지 feature tier(`DROP_ID_LIKE 제거 후 전체 34개` / `CORE6` / `CORE7`)를 **동일한 CV로 비교**한다.

### 요구사항
1. **전처리**
   - 범주형은 `OrdinalEncoder`(트리에는 충분, OHE 불필요)로 인코딩. `handle_unknown='use_encoded_value', unknown_value=-1`.
   - 스케일링 불필요(트리). 이상치 처리 절대 금지.
2. **평가 프로토콜**
   - `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` 고정.
   - **주지표 = ROC-AUC**, 보조 = balanced accuracy, F1(macro), 그리고 참고용 accuracy.
   - 세 tier별로 각 지표의 평균±표준편차를 표로 출력.
3. **과적합 점검**
   - 각 모델에서 train 지표 vs CV 지표 갭을 함께 출력(트리는 과적합이 쉽다 — 갭 모니터링 필수).
4. **하이퍼파라미터 튜닝**
   - `max_depth`, `min_samples_leaf`, `min_samples_split`, `ccp_alpha`(비용복잡도 가지치기), `class_weight`(None vs 'balanced')를 `GridSearchCV` 또는 `RandomizedSearchCV`로 탐색. scoring='roc_auc'.
   - 불균형 데이터이므로 `class_weight='balanced'`를 반드시 후보에 포함.
   - 튜닝은 가장 성능 좋은 tier에 대해 수행하고, 최적 파라미터와 그때의 CV 점수를 보고.
5. **해석/산출물**
   - 최적 트리의 `feature_importances_`를 막대그래프로 저장. CORE6/CORE7의 feature가 상위에 오는지 확인.
   - 트리 구조를 `plot_tree` 또는 `export_text`로 저장(depth 제한해 가독성 확보).
   - 결과 요약을 콘솔에 표로, 그리고 `results_decision_tree.md`로 저장.
   - 재현성을 위해 모든 `random_state=42` 고정.

### 코드 구조 (권장)
```
data_load.py         # 로컬/kagglehub 자동 탐지 로드, tier 상수 정의
dt_baseline.py       # 3개 tier × DecisionTree, CV 비교표 출력
dt_tune.py           # 최적 tier 하이퍼파라미터 튜닝 + 해석 산출물
```
또는 하나의 노트북/스크립트로 묶어도 좋다. 모듈화하되 한 번에 실행 가능하게.

## 4. 기대치 / 주의 (이미 확인된 것)
- HistGB/RF 기준 ID 제거 후 CV AUC는 약 0.67 수준. 단일 Decision Tree는 보통 앙상블보다 낮게 나오는 게 정상이니, **튜닝(특히 max_depth/ccp_alpha)으로 과적합을 잡는 데 집중**할 것. depth 무제한 트리는 train AUC≈1.0, CV는 급락한다.
- tier 비교에서 CORE6 vs CORE7 vs 전체34의 AUC 차이는 작고 오차범위가 겹칠 수 있다. 단정하지 말고 평균±std와 함께 제시.
- 이 작업은 이후 여러 모델(RF, ExtraTrees, HistGB/XGBoost, LogisticRegression 등)로 확장할 baseline이다. 그래서 **tier 비교 로직과 평가 함수는 모델만 갈아끼우면 재사용 가능하도록** 함수화해 둘 것. (예: `evaluate(model, feature_cols) -> dict of metrics`)

## 5. 먼저 할 일
1. 데이터 경로 탐지 후 로드, shape·dtype·tier 컬럼 존재 여부 sanity check 출력.
2. Decision Tree baseline 3-tier 비교 실행.
3. 최적 tier 튜닝 + 산출물 저장.
끝나면 `results_decision_tree.md`에 (a) tier별 성능표, (b) 최적 하이퍼파라미터, (c) feature importance 상위, (d) 다음 모델로 넘어갈 때의 관찰 메모를 정리해 줘.
