# csv 등 외부파일 읽어들일 때 주로 사용하는 pandas 라이브러리 -> DB 테이블 형태로 출력 가능
import pandas as pd

# 트레인 데이터 셋 != 테스트 데이터 셋 for 명확한 테스트
train= pd.read_csv("https://www.dropbox.com/s/k94xzp4qjuoxcp1/train.csv?dl=1")
train.dropna()
train.head()

test= pd.read_csv("https://www.dropbox.com/s/0mnday26niaqc6i/test.csv?dl=1")
test.dropna()
test.head()

# 테스트 데이터 셋에는 fare_amount 컬럼 누락 -> 최종목적이 이 값을 구하는 것
print("트레인 데이터의 컬럼 목록 : \n", train.columns.tolist())
print("테스트 데이터의 컬럼 목록 : \n", test.columns.tolist())

# 데이터 셋 중 특정 컬럼 분포도로 표현
import seaborn as sns
import matplotlib.pyplot as plt
fig = sns.distplot(train['passenger_count'], kde=False)
plt.title("distribution of passenger_count")
plt.show()

# 변수들간의 correlation 확인
fig = sns.regplot(x="passenger_count", y="fare_amount", data=train)
plt.title("passenger_count vs fare_amount")
plt.show()

# 다양한 머신러닝 최적화 알고리즘 모델을 포함하고 있는 sklearn 라이브러리
from sklearn.linear_model import LinearRegression
lr = LinearRegression()

# 독립변수 X, 종속변수 y 설정
lr.fit(X=train[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count']], y=train['fare_amount'])

LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)

features=['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count']

# 테스트 데이터를 활용하여 학습된 모델 예측 실행
test['fare_amount']=lr.predict(test[features])

# 제출하기기 위해 필요한 컬럼만 추출
tax_submission = test[['key', 'fare_amount']]

# csv 변환하여 파일로 저장
# 저장하고자 하는 local path 설정
tax_submission.to_csv('C:/Users/7040_64bit/new york taxi fare_amount prediction/taxi_sub.csv', index= False)

tax_submission.head()
