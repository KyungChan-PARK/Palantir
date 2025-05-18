# 탐색적 데이터 분석 보고서: eda_report_marketing_data

## 1. 데이터 개요

- **파일명**: marketing_data.csv
- **파일경로**: C:\Users\packr\OneDrive\palantir\data\marketing_data.csv
- **데이터 크기**: 50행 × 16열
- **열 개수**: 16

### 1.1 열 정보

| 열 이름 | 데이터 타입 |
|---------|------------|
| campaign_id | int64 |
| campaign_name | object |
| type | object |
| channel | object |
| start_date | object |
| end_date | object |
| budget | int64 |
| spend | int64 |
| impressions | int64 |
| clicks | int64 |
| conversions | int64 |
| ctr | float64 |
| cvr | float64 |
| cpa | float64 |
| revenue | int64 |
| roi | float64 |

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
| campaign_id | 0 | 0.00 |
| campaign_name | 0 | 0.00 |
| type | 0 | 0.00 |
| channel | 0 | 0.00 |
| start_date | 0 | 0.00 |
| end_date | 0 | 0.00 |
| budget | 0 | 0.00 |
| spend | 0 | 0.00 |
| impressions | 0 | 0.00 |
| clicks | 0 | 0.00 |
| conversions | 0 | 0.00 |
| ctr | 0 | 0.00 |
| cvr | 0 | 0.00 |
| cpa | 0 | 0.00 |
| revenue | 0 | 0.00 |
| roi | 0 | 0.00 |

#### 기술 통계 시각화

- [Distribution - 1](../..\output\viz\eda_report_marketing_data\descriptive\descriptive_campaign_id_distribution.png)
- [Distribution - 2](../..\output\viz\eda_report_marketing_data\descriptive\descriptive_budget_distribution.png)
- [Distribution - 3](../..\output\viz\eda_report_marketing_data\descriptive\descriptive_spend_distribution.png)
- [Distribution - 4](../..\output\viz\eda_report_marketing_data\descriptive\descriptive_impressions_distribution.png)
- [Distribution - 5](../..\output\viz\eda_report_marketing_data\descriptive\descriptive_clicks_distribution.png)

### 3.상관 관계 분석

#### 강한 상관 관계

| 열1 | 열2 | 상관계수 | 상관관계 유형 |
|-----|-----|---------|------------|
| budget | spend | 0.977 | 양의 (매우 강한) |
| clicks | conversions | 0.922 | 양의 (매우 강한) |
| clicks | revenue | 0.922 | 양의 (매우 강한) |
| conversions | revenue | 1.000 | 양의 (매우 강한) |

#### 상관 관계 분석 시각화

- [Heatmap - 1](../..\output\viz\eda_report_marketing_data\correlation\correlation_heatmap.png)
- [Scatter - 2](../..\output\viz\eda_report_marketing_data\correlation\correlation_scatter_budget_spend.png)
- [Scatter - 3](../..\output\viz\eda_report_marketing_data\correlation\correlation_scatter_clicks_conversions.png)
- [Scatter - 4](../..\output\viz\eda_report_marketing_data\correlation\correlation_scatter_clicks_revenue.png)
- [Scatter - 5](../..\output\viz\eda_report_marketing_data\correlation\correlation_scatter_conversions_revenue.png)

## 4. 주요 인사이트

- [descriptive] 'clicks' 열에 이상치가 많습니다 (6.0%)
- [descriptive] 'cpa' 열에 이상치가 많습니다 (8.0%)
- [descriptive] 'roi' 열에 이상치가 많습니다 (10.0%)
- [descriptive] 'clicks' 열이 오른쪽으로 치우친 분포를 보입니다 (왜도: 1.09)
- [descriptive] 'conversions' 열이 오른쪽으로 치우친 분포를 보입니다 (왜도: 1.53)
- [descriptive] 'cpa' 열이 오른쪽으로 치우친 분포를 보입니다 (왜도: 1.34)
- [descriptive] 'revenue' 열이 오른쪽으로 치우친 분포를 보입니다 (왜도: 1.53)
- [descriptive] 'roi' 열이 오른쪽으로 치우친 분포를 보입니다 (왜도: 3.10)
- [correlation] 'budget'와 'spend' 사이에 매우 강한 양의 상관 관계가 있습니다 (r=0.98)
- [correlation] 'clicks'와 'conversions' 사이에 매우 강한 양의 상관 관계가 있습니다 (r=0.92)
- [correlation] 'clicks'와 'revenue' 사이에 매우 강한 양의 상관 관계가 있습니다 (r=0.92)
- [correlation] 'conversions'와 'revenue' 사이에 매우 강한 양의 상관 관계가 있습니다 (r=1.00)

## 5. 추천 시각화

### 5.1. budget vs spend (r=0.98)

- **유형**: Scatter
- **열**: budget, spend
- **추천 이유**: 매우 강한 양의 상관 관계가 있습니다.

### 5.2. clicks vs conversions (r=0.92)

- **유형**: Scatter
- **열**: clicks, conversions
- **추천 이유**: 매우 강한 양의 상관 관계가 있습니다.

### 5.3. clicks vs revenue (r=0.92)

- **유형**: Scatter
- **열**: clicks, revenue
- **추천 이유**: 매우 강한 양의 상관 관계가 있습니다.

### 5.4. type별 campaign_id 분포

- **유형**: Boxplot
- **열**: type, campaign_id
- **추천 이유**: 범주별 campaign_id 분포 차이를 보여줍니다.

### 5.5. type별 budget 분포

- **유형**: Boxplot
- **열**: type, budget
- **추천 이유**: 범주별 budget 분포 차이를 보여줍니다.

### 5.6. channel별 campaign_id 분포

- **유형**: Boxplot
- **열**: channel, campaign_id
- **추천 이유**: 범주별 campaign_id 분포 차이를 보여줍니다.

### 5.7. channel별 budget 분포

- **유형**: Boxplot
- **열**: channel, budget
- **추천 이유**: 범주별 budget 분포 차이를 보여줍니다.

