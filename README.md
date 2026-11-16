# ah07-cancer-subtype

AI 헬스케어 7기 해커톤 · **유전자 변이 정보 기반 암 아형(SUBCLASS) 분류**

- 과제: 환자별 유전자 변이(유전자 4,384개, `WT` 또는 변이 표기)로 26개 암 아형 예측
- 평가: Macro F1 (Public = Test 100%), 하루 제출 4회
- 일정: 11/11(수) 모든 제출 완료 → 11/12 리허설 → 11/13 발표
- 팀 기준 점수: **CV Macro F1 0.3209** (`notebooks/J_leader/00_baseline.ipynb`, XGBoost + 기본 피처)

## 최종 결과

| 항목 | 내용 |
|---|---|
| 데이콘 최종 제출 | 24번, **LB Macro F1 0.43631** |
| 최종 코드 | [`submit/final_solution.ipynb`](submit/final_solution.ipynb) (파일 하나로 학습부터 예측까지) |
| 모델 | 가중치 XGBoost(E) + 로지스틱 회귀(E + 위치 구간) + LightGBM g3(E + 핫스팟×3 + 위치 구간 + 정확한 위치), 같은 비중 평균 |
| 후처리 | 쌍 2단계 판정(`USE_PAIR_TWO_STAGE = True`): KIPAN/KIRC, GBMLGG/LGG 쌍은 확률을 합쳐 다른 암과 비교하고, 쌍 안의 선택은 train 쌍 환자로만 학습한 작은 LR(유전자 10개 + 변이 수)로 정함 |
| 실행 환경 | Google Colab CPU, Python 3.13.18, [`requirements.txt`](requirements.txt) 버전 |
| 재현 | 같은 라이브러리 버전의 리눅스 환경(코랩·별도 리눅스)에서 재실행 시 test 2,546명 예측 전부 일치 |
| 실행 시간 | 코랩 무료 CPU 약 40분 |

> 처음 왔다면 **[docs/coding_rules.md](docs/coding_rules.md)(팀 코딩 규칙)** 부터 읽어 주세요. 누수 하나면 수상 제외예요.

## 데이터 누수 방지

이 프로젝트의 모든 학습·검증·예측은 아래 원칙을 따름. 세부 규칙은 [docs/coding_rules.md](docs/coding_rules.md) 1장에 있음.

**1. test 데이터의 사용 범위**
- test는 입력 형식 확인과 최종 예측에만 사용함.
- test의 통계(분포·평균·결측 개수 등), 행 개수, index, test를 탐색해서 얻은 정보는 전처리·피처·모델 선택에 사용하지 않음.
- train과 test를 합쳐서 전처리하지 않음. pseudo-labeling(test에 가짜 라벨을 붙여 재학습)을 사용하지 않음.

**2. 학습하는 변환은 train(학습 fold)에서만 fit**
- 빈도 기준으로 고르는 피처(위치 구간, 정확한 변이 위치)는 학습 fold 안에서 "5명 이상" 기준으로 고르고, 검증 fold와 test에는 고른 목록을 적용만 함.
- 모델 학습, 가중치(암 종류별 환자 수 기반) 계산도 학습 fold 또는 전체 train에서만 수행함.
- 한 환자 안에서만 계산하는 변환(변이 여부 0/1, 환자별 변이 개수, 변이 표기에서 유형·위치 추출)은 fit이 없는 행 단위 계산임.
- 빈칸(결측)은 통계로 채우지 않고 행 단위로 "변이 없음(WT)"과 같게 처리함. train에는 빈칸이 없어 이 처리 전후 train 피처는 동일함.

**3. 정직한 검증**
- 변이가 완전히 같은 환자는 같은 fold에 묶는 StratifiedGroupKFold 5-fold(`src/validation.get_folds`)를 모든 실험에 공통으로 사용함.
- 후보 점검용으로 고정 홀드아웃 20%(1,241명)를 따로 둠(`scripts/fixed_holdout_manifest.py`).

**4. 데이터 오류를 이용하지 않음**
- 변이가 완전히 같은데 라벨이 다른 환자(KIPAN–KIRC, GBMLGG–LGG)를 이용해 짝의 라벨로 예측하는 규칙은 CV 점수를 올리지만, 데이터 구성 오류를 이용하는 방법이라 사용하지 않음. 이 구조는 데이터 한계로 분석·보고함.

**5. 외부 데이터와 출처**
- 외부 데이터를 학습이나 피처 값으로 사용하지 않음.
- 논문에 공개된 지식만 참고함: 반복 변이 자리(핫스팟) 목록은 Chang MT, et al. Nat Biotechnol. 2016;34(2):155-163을 참고함.
- 사전학습 모델은 사용하지 않음.

**6. git 기록에 대한 설명**
- 초기 커밋 일부에는 test의 입력 형식 확인 출력(변이 표기 형식, 빈칸 개수)과 재현 확인용 예측 일치율 출력이 남아 있음.
- 이 출력은 전처리·피처·모델 선택에 사용하지 않았고, 관련 셀은 이후 삭제함(PR #44).

## 폴더 구조

| 경로 | 내용 | 수정하는 사람 | 깃허브 |
|---|---|---|---|
| `docs/coding_rules.md` | 팀 코딩 규칙 (누수 방지, 공통 함수, 실험 순서) | ① 장성은 | 올림 |
| `data/` | 대회 csv 3개 (각자 넣기) | - | **안 올림** |
| `src/config.py` | seed, fold 수, 경로 | ④ 박유정 | 올림 |
| `src/data_loader.py` | 데이터 불러오기 | ④ 박유정 | 올림 |
| `src/preprocessing.py` | 기본 전처리 (WT=0, 변이=1) | ② 홍시윤 | 올림 |
| `src/features.py` | **검증된** 공통 피처 | ② 홍시윤 | 올림 |
| `src/validation.py` | 공통 교차검증 (fold, run_cv) | ③ 안주형 | 올림 |
| `src/evaluation.py` | Macro F1, 클래스별 F1, 실험 기록 | ③ 안주형 | 올림 |
| `src/prediction.py` | 제출 파일 만들기 | ④ 박유정 | 올림 |
| `notebooks/J_leader/` | 00_baseline (팀 기준 점수) | ① 장성은 | 올림 |
| `notebooks/H_data/` | 01_eda, 02_features | ② 홍시윤 | 올림 |
| `notebooks/A_model/` | 03_models, 05_ensemble | ③ 안주형 | 올림 |
| `notebooks/P_light/` | 04_lightweight | ④ 박유정 | 올림 |
| `experiments/log_*.csv` | 실험 기록 (사람별 1개) | 각자 자기 파일 | 올림 |
| `experiments/preds/` | 앙상블 후보의 예측 확률 (`save_predictions()`) | 각자 자기 파일 | 올림 |
| `reports/figures/` | 발표용 그래프 | 전원 | 올림 |
| `submit/final_solution.ipynb` | 최종 제출 파일 틀 (각자 자기 칸만 채움) | ④ 박유정 | 올림 |
| `docs/team_plan.md` | 팀 운영 확정본 (역할·일정·규칙) | ① 장성은 | 올림 |
| `docs/ngs_research.md` | NGS·유전체 조사 | ② 홍시윤 | 올림 |
| `docs/service_plan.md` | 서비스 기획 | ① 장성은 | 올림 |
| `models/` | 학습된 모델 파일 | - | **안 올림** |
| `submissions/` | 제출 csv | - | **안 올림** |

실험 기록 파일: `log_J_leader.csv` · `log_H_data.csv` · `log_A_model.csv` · `log_P_light.csv`

## 시작하기

### 1. 데이터 넣기
데이콘에서 받은 `train.csv`, `test.csv`, `sample_submission.csv`를 `data/` 폴더에 넣어요.

### 2-A. Colab
1. 깃허브 레포 페이지 왼쪽 위 브랜치가 **dev**인지 확인하고, **Code → Download ZIP**으로 받아 압축을 풀어요.
2. 폴더 이름을 `ah07-cancer-subtype`으로 맞추고, 구글 드라이브 `내 드라이브`에 통째로 올려요. 그 안 `data/`에 csv를 넣어요.
3. 드라이브에서 자기 폴더의 노트북(예: `notebooks/A_model/03_models.ipynb`)을 Colab으로 열고 위에서부터 실행해요. 첫 셀이 드라이브를 자동으로 연결해요.
4. 작업이 끝나면 **파일 → GitHub에 사본 저장**을 눌러요. 브랜치는 **깃허브 웹에서 미리 만든 이번 작업 브랜치**(예: `feat/model_A_1109`, 아래 "브랜치와 PR"의 Colab 순서 참고), 경로는 `notebooks/자기폴더/파일이름`으로 올려요. 처음 한 번은 깃허브 권한 허용이 필요하고, "비공개 저장소 포함"에 체크해야 레포가 보여요.
5. **기록표와 예측 파일은 Colab에서 바로 못 올려요.** "GitHub에 사본 저장"은 노트북만 올리기 때문이에요. 깃허브 레포 페이지에서 이번 작업 브랜치를 고르고 → `experiments/` 폴더 → **Add file → Upload files**로 드라이브에 생긴 `log_자기이름.csv`(그리고 `preds/` 안 자기 파일)를 끌어다 올려요.
6. 깃허브 레포 페이지에서 **Pull request**를 만들어요 (아래 "브랜치와 PR" 참고).
7. 다른 사람의 `src/` 변경이 dev에 합쳐졌다면, dev 브랜치 ZIP을 다시 받아 `src/` 폴더만 바꿔 넣어요.

### 2-B. VS Code / Jupyter (로컬)
```bash
git clone https://github.com/jsejj87/ah07-cancer-subtype.git
```
```bash
pip install -r requirements.txt
```
`data/`에 csv를 넣고, 커밋 템플릿을 한 번 등록해요.
```bash
git config --local commit.template .github/commit_template.txt
```
그다음 작업마다 새 브랜치를 만들어 작업해요 (아래 "브랜치와 PR").

## 실험하는 법

모든 노트북 첫 부분(환경 설정, 데이터 불러오기)은 이미 들어 있어요. 아래처럼 실험해요.

```python
from xgboost import XGBClassifier
PARAMS = dict(n_estimators=200, max_depth=6, random_state=SEED)

pipe = Pipeline([
    ("features", basic_transformer()),      # 전처리는 Pipeline 안에 (fold 안에서만 fit)
    ("model", XGBClassifier(**PARAMS)),
])
cv = run_cv(pipe, X, y, folds=folds)        # 모두 같은 fold, 같은 지표
per_class_f1(y, cv["oof"], le.classes_)     # 어떤 클래스가 약한지
# cv["oof_proba"]: fold 밖 예측 확률 (앙상블용, 행=train 순서, 열=le.classes_ 순서)
log_result(owner="A_model", notebook="03_models.ipynb", experiment="xgb_depth8",
           features="basic", model="XGBoost", params=PARAMS, cv=cv)   # 기록
```

**앙상블 후보가 된 모델은 예측도 저장해요** (`notebooks/A_model/05_ensemble.ipynb`에서 섞어요)
```python
final = pipe.fit(X, y)                                   # 전체 train으로 학습
test_proba = final.predict_proba(test.drop(columns=["ID"]))
save_predictions(owner="A_model", name="xgb_depth8", oof_proba=cv["oof_proba"],
                 test_proba=test_proba, classes=le.classes_, cv=cv)
```

**시간 아끼는 법**
- 코드가 돌아가는지만 볼 때: `run_cv(pipe, X, y, folds=folds[:1])` (1개 fold). **기록은 5-fold로만** 해요.
- 5-fold 베이스라인은 6코어 PC에서 약 10분이고, Colab 무료 CPU(2코어)는 더 느려요. 무거운 실험은 하루 일정에 미리 넣어 두세요.

**재현성 주의** (최종 코드 검증에서 점수가 재현돼야 해요)
- 모든 모델에 `random_state=SEED` (CatBoost는 `random_seed=SEED`)를 넣어요.
- LightGBM은 `deterministic=True, force_row_wise=True`를 같이 넣어요 (멀티스레드에서 결과가 미세하게 달라질 수 있어요).
- 최종 제출에 쓸 모델은 GPU 말고 CPU로 학습해요.

## 브랜치와 PR

**작업 하나마다 새 브랜치를 만들고, 합치면 지워요.** 충돌은 "사람별 폴더" 규칙으로 막아요.

```
dev ─(새 브랜치)→ feat/data_H_1109 ─PR(팀원 1명 승인)→ dev ─(최종, 팀장)→ main
```

| 브랜치 | 용도 | 누가 |
|---|---|---|
| `main` | 최종 제출본만 | 팀장이 dev에서 합침 (11/11) |
| `dev` | 팀이 같이 쓰는 최신본 | PR로만 합침 (직접 push 금지) |
| `feat/역할_성이니셜_월일` | 그날의 작업 | 본인. 합친 뒤 삭제 |

**브랜치 이름:** `feat/역할_성이니셜_월일` (월일은 4자리, 예: 11월 9일 → `1109`)

| 팀원 | 예시 |
|---|---|
| ① 장성은 | `feat/leader_J_1109` |
| ② 홍시윤 | `feat/data_H_1109` |
| ③ 안주형 | `feat/model_A_1109` |
| ④ 박유정 | `feat/light_P_1109` |

- 같은 날 두 번째 브랜치부터는 뒤에 번호: `feat/model_A_1109_2`
- 팀 공통 문서 수정처럼 특정인의 작업이 아니면: `feat/docs_1109`

**PR 규칙:** dev로 PR을 올리면 **본인 말고 팀원 아무나 1명**이 PR 양식의 누수 체크리스트를 보고 Approve → Merge → **Delete branch** 버튼으로 브랜치 삭제. 누수가 애매하면 팀장에게 넘겨요.

**VS Code: 작업 하나마다 이 순서**
```bash
git checkout dev
```
```bash
git pull origin dev
```
```bash
git checkout -b feat/model_A_1109
```
작업 → 자기 폴더 파일만 add → commit → push
```bash
git push origin feat/model_A_1109
```
그다음 깃허브에서 **Compare & pull request** → base가 **dev**인지 확인 → PR 생성.

**Colab: 작업 하나마다 이 순서** (Colab에서는 새 브랜치를 만들 수 없어서, 깃허브 웹에서 먼저 만들어요)
1. 깃허브 레포 페이지 왼쪽 위 브랜치 버튼(`dev`)을 눌러요.
2. 칸에 새 이름(`feat/model_A_1109`)을 쓰고 → **Create branch: feat/model_A_1109 from dev**를 눌러요.
3. Colab에서 작업한 뒤 **파일 → GitHub에 사본 저장** → 브랜치 목록에서 방금 만든 `feat/model_A_1109`를 골라 저장해요.
4. 깃허브에서 **Compare & pull request** → base가 **dev**인지 확인 → PR 생성.

## 협업 규칙

1. **자기 폴더만 수정해요.** `.ipynb`는 같은 파일을 둘이 고치면 합칠 수 없어요. 새 실험 노트북은 자기 폴더 안에 자유롭게 만들어요.
2. **`src/`는 담당자만 수정해요.** 바꾸면 디스코드에 알려 주세요. 다른 사람은 dev에 합쳐진 뒤 `git pull origin dev`로 받아요.
3. **피처는 검증된 것만 `src/features.py`로.** 노트북에서 `run_cv()` 점수가 오르는 게 확인된 것만 옮겨요.
4. **리더보드 제출은 팀장을 통해서만** (하루 4회). 제출하고 싶은 후보는 기록표의 CV 점수와 함께 알려 주세요.
5. **원본 데이터, 모델 파일, 제출 csv, 가상환경, API 키는 올리지 않아요.** `.gitignore`가 대부분 막아 줘요.
6. **이전 대회나 공개 코드는 복사하지 않아요.** 외부 데이터는 쓰지 않아요.
