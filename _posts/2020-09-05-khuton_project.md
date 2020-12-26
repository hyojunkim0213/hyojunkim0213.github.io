---
layout: post
title: "경희대 해커톤 우수상 수상 - 머신러닝 모델을 활용한 BMTI 사이트 구축 (모델링 편)"
date: 2020-11-30
excerpt: " BMTI는 국민건강영양조사의 식품섭취빈조사 데이터를 통해 BMI 수치를 예측하여 솔루션을 제공하는 사이트입니다."
tags: [Machine learning, classification]
project: true
comments: false
---
## 0\. 개요

- MBTI, SPTI 등 간단하면서 사람들의 흥미를 불러오는 서비스가 많이 등장

- 코로나로 인해 확진자가 아닌, 확찐자라는 용어가 등장 : 사람들의 관심사가 높음<br>

      ‘BMTI’ 사이트 : 식습관 + 운동습관 => BMI 수치 예측 & 식습관 분석

- 해커톤에서는 Domain knowledge를 기반으로 변수를 추출하여 높은 성능이 나오지 않았음
(사람들에게 시각적인 효과와 형태군별 솔루션 제공을 위함)

- **다음의 모델링 프로세스는 데이터 분석에 보완을 하여, 통계적 기법과 다양한 분석 기법을 추가하여 서술하였음.** (해커톤에서 추출한 변수와는 다름)

## 1\.Data preprocessing & feature selection

- 국민건강영양조사 식품섭취빈조사 데이터 활용

식품섭취빈도조 데이터에서 column명이 FA 로 시작하는 경우는 음식에 대한 1회 평균섭취량을 뜻함<BR>
FF 로 시작하는 경우 최근 1년간 평균섭취빈도이므로 FF에 해당하는 column만 추출한다.

![image](https://user-images.githubusercontent.com/28617444/103146990-d426a880-4793-11eb-87aa-1526594d8a7c.png)

또한, BMI 수치를 계산해주기 위해 키와 체중 column을 추가하고 age column을 통해 성인을 대상으로 한다.

![image](https://user-images.githubusercontent.com/28617444/103147019-118b3600-4794-11eb-83eb-5570d3546b97.png)

독립변수 BMI column생성
```python
df_20s['BMI'] = df_20s.HE_wt / (df_20s.HE_ht/100)**2
df_20s['BMI'] = round(df_20s['BMI'],1)
```
<br>

- 데이터 값 (88, 99 : 모름, 무응답) NA 처리 후, 결측치 모두 제거

국민건강영양조사 데이터의 경우 대부분 설문조사 형태로 데이터값이 입력되므로 모름 및 무응답의 경우 NA 로 처리한 후, 결측치에 해당하는 행을 모두 제거하여 분석한다.

```python
for name in df_20s.columns:
    df_20s[[name]] = df_20s[[name]].replace({88:np.nan, 99:np.nan})

df_20s = df_20s.dropna()
```

## 2\. Feature selection

  1. 다중회귀분석

  통계적 접근 (p-value 와 coefficient)으로 target 변수에 대한 설명력으로 변수 제거를 하고자 한다.

```python
import pandas as pd
import numpy as np
import statsmodels.api as sm

# 데이터_설명변수와 독립변수 분할
df_20s_data = df_20s.drop(['BMI'], axis=1)
target = df_20s[["BMI"]]

# for b0, 상수항 추가
X = sm.add_constant(df_20s_data, has_constant = "add")

# OLS 검정
multi_model = sm.OLS(target, X)
fitted_multi_model = multi_model.fit()
fitted_multi_model.summary()
```
![image](https://user-images.githubusercontent.com/28617444/103147206-81022500-4796-11eb-964b-1aed127799fc.png)

모든 변수를 사용했으므로, 전체 모델에 대한 설명력은 **0.181**로 낮을 수 밖에 없다. (F 통계량도 매우 높아 유의하지 않은 모델임)<br>
해당 변수에 대한 p-value값으로 설명력을 독립적으로 판단하고자 한다.<br>
**p-value 값이 0.05 이상**이면 이는 유의하지 않는 변수(Target에 영향을 주지 않는 변수)로 간주하겠다.

```python
#p-value값 확인
results_summary = fitted_multi_model.summary()
#개별 변수에 대한 테이블 결과만을 html문으로 취급하여 도출
results_as_html = results_summary.tables[1].as_html()
df_result = pd.read_html(results_as_html, header=0, index_col=0)[0]
# p value 0.05 이상인 변수 도출
df_result_variabel = df_result[df_result['P>|z|'].sort_values() > 0.05]
df_result_variabel
```
![image](https://user-images.githubusercontent.com/28617444/103147294-68463f00-4797-11eb-97f5-50a946c60d3a.png)

따라서, **97개의 변수 유의하지 않은 변수**

유의한 변수는 다음과 같다.

![image](https://user-images.githubusercontent.com/28617444/103147327-9592ed00-4797-11eb-9187-a7ad2d61bcf9.png)


그러나 변수 간의 교호 작용이 일어날 수 있으므로 이러한 부분도 배제해서는 안 된다.<br>
따라서, 독립변수끼리 상관 관계가 보이면 안 되며 이러한 경우를 분석 대상에서 제거하기 위해 **VIF(분산팽창요인)**을 판단하여 10이 넘으면 제거하고자 한다.

2. 다중공선성 확인

```python
#유의미한 변수만을 추출
df_20s_revise= df_20s[['FF_BIBIM', 'FF_DUMP', 'FF_F_RCAK', 'FF_T_POTA', 'FF_J_KIMC',
       'FF_F_EGG', 'FF_S_EGG', 'FF_R_PORK', 'FF_PORKBY', 'FF_S_CHIC',
       'FF_SCRAB', 'FF_FPASTE', 'FF_GARLIC', 'FF_CORN', 'FF_COFFEE', 'FF_TEA',
       'FF_SODA', 'FF_SOJU']]

 from statsmodels.stats.outliers_influence import variance_inflation_factor
 from statsmodels.tools.tools import add_constant

 train_x_final = add_constant(df_20s_revise)
 vif = pd.DataFrame()
 vif["VIF Factor"] = [variance_inflation_factor( train_x_final.values, i) for i in range(train_x_final.shape[1])]
 vif["features"] = train_x_final.columns
 vif
```
![image](https://user-images.githubusercontent.com/28617444/103147573-3d111f00-479a-11eb-9229-cf27454469ac.png)

**VIF가 10이상인 경우가 없으므로 다중공선성이 없다고 판단**

## 3\. Modeling_RandomForestRegression

예측 모델로는 **랜덤 포레스트 회귀 모델**을 사용하였다. <br>
이 때, **GridSearchCV**를 통해 MSE의 음수값인 neg_mean_squared_error를 기준으로 최적의 파라미터를 도출하여 모델을 수립하였다.

```python
df_20s_All= df_20s[['FF_BIBIM', 'FF_DUMP', 'FF_F_RCAK', 'FF_T_POTA', 'FF_J_KIMC',
       'FF_F_EGG', 'FF_S_EGG', 'FF_R_PORK', 'FF_PORKBY', 'FF_S_CHIC',
       'FF_SCRAB', 'FF_FPASTE', 'FF_GARLIC', 'FF_CORN', 'FF_COFFEE', 'FF_TEA',
       'FF_SODA', 'FF_SOJU','BMI']]

 from sklearn.model_selection import train_test_split
 from sklearn.model_selection import GridSearchCV
 from sklearn.ensemble import RandomForestRegressor

#train,test data set 생성
 X_train, X_test, y_train, y_test = train_test_split(df_20s_X,df_20s_All.BMI, stratify=target, random_state=0)
#modeling
param_grid = [
        {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
        {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
    ]
forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)

grid_search.fit(X_train, y_train)
print('최적 하이퍼 파라미터: ', grid_cv.best_params_)
print('최고 예측 정확도: {:.4f}'.format(grid_cv.best_score_))
```
![image](https://user-images.githubusercontent.com/28617444/103148149-22da3f80-47a0-11eb-92c0-7acc58244099.png)

## 4\. Evaluation

```python
predictions = grid_search.predict(X_test)

# 예측 데이터 시각화
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

fig,(ax1,ax2)= plt.subplots(ncols=2)
fig.set_size_inches(12,5)
sns.distplot(y_train,ax=ax1,bins=50)
ax1.set(title="train")
sns.distplot(predictions,ax=ax2,bins=50)
ax2.set(title="test")
```
예측 결과에 대한 시각화 결과는 다음과 같다.
![image](https://user-images.githubusercontent.com/28617444/103148164-47ceb280-47a0-11eb-82f4-7817b41bce0d.png)

## 5\. 모델 배포

웹페이지 모델 배포를 위해 pickle 형태로 저장하여 배포하였다.
```python
import pickle

with open('./classification.pkl', 'wb') as f:
    pickle.dump(grid_cv, f)

with open('classification.pkl', 'rb') as f:
    data = pickle.load(f)

data
```


## 6\. 연관규칙으로 정상군의 식습관 패턴 파악

- support, confidence, lift (apriori 알고리즘 기반) - mlxtend 패키지

BMI 수치가 정상인 집단의 식습관 패턴을 연관규칙을 통해 파악하고자 한다.<br>
이 때, 연관규칙을 위해 변수들을 모두 **더미변수**로 변경해주어야 한다.
```python
df_all = df_20s[['FF_PIZZA','FF_HAMBER','FF_F_CHIC','FF_INSTNO','FF_ICECM','FF_SNACK','FF_SOJU','FF_BEER','FF_J_SOYP','FF_J_KIMC','FF_KIMCHI','FF_SPROU','FF_VSALAD','FF_MILK','BMI']]
df_all.loc[df_all['BMI'] < 25, 'BMI'] = 0
df_all.loc[df_all['BMI'] >= 25, 'BMI'] = 1
#BMI 정상군
df_normal = df_all[df_all.BMI==0]
a = df_normal.columns
fastfood = ['FF_PIZZA', 'FF_HAMBER', 'FF_F_CHIC', 'FF_INSTNO']
fastfood_df = df_normal[fastfood]
df_normal_X = df_normal.drop('BMI', axis=1)

# 연관규칙을 위해 더미변수로 변경
df_dummy = pd.get_dummies(fastfood_df, columns=fastfood)

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori

frequent_itemsets = apriori(df_dummy, min_support=0.05, use_colnames=True)

from mlxtend.frequent_patterns import association_rules
results = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.3)
results.sort_values('confidence',ascending=False)
```
![image](https://user-images.githubusercontent.com/28617444/103148460-16a3b180-47a3-11eb-95b5-7f4fc6fb9a8f.png)

## 7\. 웹페이지 구성 및 시연 영상 링크

- 형태군 형성 (패스트푸드 과다섭취 타입, 디저트 과다섭취 타입, 주류 과다섭취 타입, 반찬 과다섭취 타입, 운동부족 타입)

- Javascript로 형태군별 해결방안 도출 등 알고리즘 작성

#### 메인화면
![image](https://user-images.githubusercontent.com/28617444/103148247-14405800-47a1-11eb-8acc-9619b893cd1c.png)

#### 설문조사 화면
![image](https://user-images.githubusercontent.com/28617444/103148256-23270a80-47a1-11eb-87d4-bf06a02f8c34.png)

#### 예측결과 화면
![image](https://user-images.githubusercontent.com/28617444/103148263-31752680-47a1-11eb-9b3e-acbc3018731d.png)
![image](https://user-images.githubusercontent.com/28617444/103148265-3934cb00-47a1-11eb-8ab2-04d59a06c72d.png)

#### 분석 및 솔루션 제시 화면
![image](https://user-images.githubusercontent.com/28617444/103148273-3f2aac00-47a1-11eb-98cc-86a661476cbf.png)
![image](https://user-images.githubusercontent.com/28617444/103148275-405bd900-47a1-11eb-9e7d-e862b9b22e9f.png)

![image](https://user-images.githubusercontent.com/28617444/103148276-418d0600-47a1-11eb-97c0-740e18c5f18b.png)

[시연영상보러가기](https://www.youtube.com/watch?v=pcSs7603vLY)
