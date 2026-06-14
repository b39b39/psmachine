import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("현재 작업 폴더:", os.getcwd())
print("현재 폴더에 있는 파일들:", os.listdir())
current_dir = os.path.dirname(os.path.abspath(__file__))

# 그 위치와 파일 이름을 합쳐서 절대 경로를 만듭니다.
file_path = os.path.join(current_dir, 'train.csv')

train_df = pd.read_csv(file_path)

# 그래프 스타일 설정
sns.set_style("whitegrid")
plt.figure(figsize=(12, 6))

# feat_31과 feat_36은 범주형(Categorical) 변수이므로
# 산점도(Scatter)보다는 Countplot(빈도수 막대그래프)이 분포를 보기에 훨씬 적합합니다.
# x축을 feat_31로, 카테고리를 feat_36으로 분리하여 target별 분포 확인

sns.catplot(
    data=train_df,
    x='feat_31',       # X축: pos / neg
    hue='target',      # 색상: 0 / 1
    col='feat_36',     # 그래프를 분리할 기준: A / B / C (3개의 그래프가 병렬로 생성됨)
    kind='count',      # 막대그래프 형태
    palette='Set1',
    height=5,
    aspect=0.8
)

# 전체 제목 추가 (약간의 위치 조정 필요)
plt.subplots_adjust(top=0.85)
plt.suptitle('Distribution of Target across feat_31 and feat_36', fontsize=16)

# 그래프 출력
plt.show()