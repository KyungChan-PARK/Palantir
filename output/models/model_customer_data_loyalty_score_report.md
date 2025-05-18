# 예측 모델링 보고서: model_customer_data_loyalty_score

## 1. 모델 개요

- **파일명**: customer_data.csv
- **문제 유형**: Regression
- **타겟 변수**: loyalty_score
- **데이터 크기**: 500행 × 12열
- **테스트 세트 비율**: 0.2
- **특성 수**: 11

## 2. 최적 모델

- **모델**: linear_regression
- **성능 지표**:
  - R² 점수: 0.7855
  - 평균 제곱 오차(MSE): 68.2924
  - 평균 제곱근 오차(RMSE): 8.2639
  - 평균 절대 오차(MAE): 6.2567

## 3. 모델 성능 비교

| 모델 | R² 점수 | MSE | RMSE | MAE |
|------|---------|-----|------|-----|
| linear_regression | 0.7855 | 68.2924 | 8.2639 | 6.2567 |

## 4. 시각화

### 예측 vs 실제 값

- [linear_regression 예측 vs 실제](..\..\output\viz\models\model_customer_data_loyalty_score_linear_regression_predicted_vs_actual.png)

### 잔차 플롯

- [linear_regression 잔차 플롯](..\..\output\viz\models\model_customer_data_loyalty_score_linear_regression_residuals.png)

### 특성 중요도


## 5. 결론 및 권장 사항

- **최적 모델**: linear_regression
- **모델 성능**: 양호함 (R² >= 0.6)

### 제안 사항

1. 추가 특성 엔지니어링 고려
2. 앙상블 기법 시도
3. 하이퍼파라미터 튜닝 범위 확장
