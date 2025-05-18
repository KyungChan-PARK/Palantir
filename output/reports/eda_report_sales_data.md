# 탐색적 데이터 분석 보고서: eda_report_sales_data

## 1. 데이터 개요

- **파일명**: sales_data.csv
- **파일경로**: data/sales_data.csv
- **데이터 크기**: 1000행 × 7열
- **열 개수**: 7

### 1.1 열 정보

| 열 이름 | 데이터 타입 |
|---------|------------|
| date | object |
| product | object |
| region | object |
| channel | object |
| quantity | int64 |
| price | float64 |
| revenue | float64 |

## 2. 데이터 전처리

### 2.Remove_nulls

- **상태**: 성공
- **제거된 행**: 0

## 3. 분석 결과

### 3.기술 통계

#### 수치형 열 통계 요약

| 열 | 평균 | 중앙값 | 표준편차 | 최소값 | 최대값 |
|-----|------|--------|---------|--------|--------|

#### 결측값 정보

| 열 | 결측값 수 | 결측값 비율(%) |
|-----|-----------|---------------|
| date | 0 | 0.00 |
| product | 0 | 0.00 |
| region | 0 | 0.00 |
| channel | 0 | 0.00 |
| quantity | 0 | 0.00 |
| price | 0 | 0.00 |
| revenue | 0 | 0.00 |

#### 기술 통계 시각화

- [Distribution - 1](../..\output\viz\eda_report_sales_data\descriptive\descriptive_quantity_distribution.png)
- [Distribution - 2](../..\output\viz\eda_report_sales_data\descriptive\descriptive_price_distribution.png)
- [Distribution - 3](../..\output\viz\eda_report_sales_data\descriptive\descriptive_revenue_distribution.png)
- [Frequency - 4](../..\output\viz\eda_report_sales_data\descriptive\descriptive_product_frequency.png)
- [Frequency - 5](../..\output\viz\eda_report_sales_data\descriptive\descriptive_region_frequency.png)

### 3.상관 관계 분석

#### 강한 상관 관계

*강한 상관 관계가 발견되지 않았습니다.*

#### 상관 관계 분석 시각화

- [Heatmap - 1](../..\output\viz\eda_report_sales_data\correlation\correlation_heatmap.png)

## 4. 주요 인사이트

*인사이트가 발견되지 않았습니다.*

## 5. 추천 시각화

### 5.1. product별 quantity 분포

- **유형**: Boxplot
- **열**: product, quantity
- **추천 이유**: 범주별 quantity 분포 차이를 보여줍니다.

### 5.2. product별 price 분포

- **유형**: Boxplot
- **열**: product, price
- **추천 이유**: 범주별 price 분포 차이를 보여줍니다.

### 5.3. region별 quantity 분포

- **유형**: Boxplot
- **열**: region, quantity
- **추천 이유**: 범주별 quantity 분포 차이를 보여줍니다.

### 5.4. region별 price 분포

- **유형**: Boxplot
- **열**: region, price
- **추천 이유**: 범주별 price 분포 차이를 보여줍니다.

