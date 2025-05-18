# generate_sample_data.py
"""
테스트용 샘플 데이터 생성
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# 저장 디렉토리
DATA_DIR = os.path.join("C:\\Users\\packr\\OneDrive\\palantir", "data")
os.makedirs(DATA_DIR, exist_ok=True)

def generate_sales_data():
    """
    판매 데이터 생성 (시계열)
    """
    # 매개변수
    n_records = 1000
    start_date = datetime(2023, 1, 1)
    products = ['노트북', '스마트폰', '태블릿', '이어폰', '스마트워치']
    regions = ['서울', '부산', '대구', '인천', '광주', '대전', '울산', '세종']
    channels = ['온라인', '오프라인', '모바일앱']
    
    # 데이터 생성
    dates = [start_date + timedelta(days=i) for i in range(n_records // 10)]
    dates = dates * (n_records // len(dates) + 1)
    dates = sorted(dates[:n_records])
    
    data = {
        'date': dates,
        'product': [random.choice(products) for _ in range(n_records)],
        'region': [random.choice(regions) for _ in range(n_records)],
        'channel': [random.choice(channels) for _ in range(n_records)],
        'quantity': [random.randint(1, 10) for _ in range(n_records)],
        'price': [random.uniform(100000, 2000000) for _ in range(n_records)],
    }
    
    # 계산된 필드 추가
    data['revenue'] = [data['quantity'][i] * data['price'][i] for i in range(n_records)]
    
    # 계절성 추가
    for i in range(n_records):
        month = data['date'][i].month
        if month in [11, 12, 1]:  # 겨울 (연말 시즌)
            data['revenue'][i] *= random.uniform(1.2, 1.5)
        elif month in [6, 7, 8]:  # 여름 (세일 시즌)
            data['revenue'][i] *= random.uniform(1.1, 1.3)
    
    # 트렌드 추가 (전체적으로 상승 추세)
    for i in range(n_records):
        days_passed = (data['date'][i] - start_date).days
        trend_factor = 1.0 + (days_passed / 365) * 0.1  # 연간 10% 성장
        data['revenue'][i] *= trend_factor
    
    # DataFrame 생성
    df = pd.DataFrame(data)
    
    # 저장
    file_path = os.path.join(DATA_DIR, 'sales_data.csv')
    df.to_csv(file_path, index=False)
    print(f"판매 데이터가 생성되었습니다: {file_path}")
    
    return df

def generate_customer_data():
    """
    고객 데이터 생성
    """
    # 매개변수
    n_records = 500
    
    # 데이터 생성
    data = {
        'customer_id': list(range(1, n_records + 1)),
        'age': [random.randint(18, 70) for _ in range(n_records)],
        'gender': [random.choice(['남성', '여성']) for _ in range(n_records)],
        'income': [random.randint(2000000, 10000000) for _ in range(n_records)],
        'education': [random.choice(['고졸', '대졸', '대학원졸']) for _ in range(n_records)],
        'marital_status': [random.choice(['미혼', '기혼', '이혼', '사별']) for _ in range(n_records)],
        'location': [random.choice(['서울', '부산', '대구', '인천', '광주', '대전', '울산', '세종', '경기', '강원', '충북', '충남', '전북', '전남', '경북', '경남', '제주']) for _ in range(n_records)],
    }
    
    # 세그먼트 추가 (소득과 나이 기반)
    segments = []
    for i in range(n_records):
        age = data['age'][i]
        income = data['income'][i]
        
        if age < 30:
            if income < 4000000:
                segment = '젊은 저소득층'
            elif income < 7000000:
                segment = '젊은 중산층'
            else:
                segment = '젊은 고소득층'
        elif age < 50:
            if income < 4000000:
                segment = '중년 저소득층'
            elif income < 7000000:
                segment = '중년 중산층'
            else:
                segment = '중년 고소득층'
        else:
            if income < 4000000:
                segment = '장년 저소득층'
            elif income < 7000000:
                segment = '장년 중산층'
            else:
                segment = '장년 고소득층'
        
        segments.append(segment)
    
    data['segment'] = segments
    
    # 소비 패턴 추가
    data['avg_purchase_amount'] = [0] * n_records
    data['purchase_frequency'] = [0] * n_records
    
    for i in range(n_records):
        segment = data['segment'][i]
        income = data['income'][i]
        
        # 소득과 세그먼트에 따른 구매 금액 및 빈도 조정
        if '저소득층' in segment:
            base_amount = income * 0.02
            base_frequency = random.uniform(1, 3)
        elif '중산층' in segment:
            base_amount = income * 0.03
            base_frequency = random.uniform(2, 5)
        else:  # 고소득층
            base_amount = income * 0.05
            base_frequency = random.uniform(3, 8)
        
        # 무작위성 추가
        data['avg_purchase_amount'][i] = int(base_amount * random.uniform(0.8, 1.2))
        data['purchase_frequency'][i] = round(base_frequency * random.uniform(0.9, 1.1), 1)
    
    # 총 소비 금액 추가
    data['total_spending'] = [
        int(data['avg_purchase_amount'][i] * data['purchase_frequency'][i] * 12)  # 연간 지출
        for i in range(n_records)
    ]
    
    # 고객 충성도 점수 추가 (0-100)
    data['loyalty_score'] = [
        min(100, int(
            (data['purchase_frequency'][i] / 8) * 50 +  # 구매 빈도 기여
            (data['total_spending'][i] / (income * 0.5)) * 50  # 소득 대비 지출 기여
        ))
        for i, income in enumerate(data['income'])
    ]
    
    # DataFrame 생성
    df = pd.DataFrame(data)
    
    # 저장
    file_path = os.path.join(DATA_DIR, 'customer_data.csv')
    df.to_csv(file_path, index=False)
    print(f"고객 데이터가 생성되었습니다: {file_path}")
    
    return df

def generate_product_data():
    """
    제품 데이터 생성
    """
    # 제품 정보
    products = [
        ('노트북', '전자제품', 1500000, 2000000, 15.0, 36),
        ('스마트폰', '전자제품', 800000, 1200000, 20.0, 24),
        ('태블릿', '전자제품', 500000, 800000, 25.0, 24),
        ('이어폰', '액세서리', 150000, 300000, 30.0, 12),
        ('스마트워치', '액세서리', 200000, 400000, 35.0, 18),
        ('데스크톱', '전자제품', 1200000, 1800000, 18.0, 48),
        ('모니터', '전자제품', 300000, 700000, 22.0, 36),
        ('키보드', '액세서리', 50000, 150000, 40.0, 24),
        ('마우스', '액세서리', 30000, 100000, 45.0, 12),
        ('프린터', '전자제품', 200000, 400000, 25.0, 24)
    ]
    
    # 데이터 생성
    data = {
        'product_name': [p[0] for p in products],
        'category': [p[1] for p in products],
        'cost': [p[2] for p in products],
        'price': [p[3] for p in products],
        'profit_margin': [p[4] for p in products],
        'lifecycle_months': [p[5] for p in products],
    }
    
    # 재고 및 판매 데이터 추가
    data['stock'] = [random.randint(50, 500) for _ in range(len(products))]
    data['monthly_sales'] = [random.randint(10, 100) for _ in range(len(products))]
    
    # 계산된 필드: 재고 회전율 및 월간 수익
    data['inventory_turnover'] = [
        data['monthly_sales'][i] / data['stock'][i]
        for i in range(len(products))
    ]
    
    data['monthly_profit'] = [
        data['monthly_sales'][i] * (data['price'][i] - data['cost'][i])
        for i in range(len(products))
    ]
    
    # 제품 평점 추가 (1-5)
    data['rating'] = [round(random.uniform(3.0, 5.0), 1) for _ in range(len(products))]
    
    # 출시일 추가
    today = datetime.now()
    data['release_date'] = [
        (today - timedelta(days=random.randint(30, 730))).strftime('%Y-%m-%d')
        for _ in range(len(products))
    ]
    
    # DataFrame 생성
    df = pd.DataFrame(data)
    
    # 저장
    file_path = os.path.join(DATA_DIR, 'product_data.csv')
    df.to_csv(file_path, index=False)
    print(f"제품 데이터가 생성되었습니다: {file_path}")
    
    return df

def generate_marketing_campaign_data():
    """
    마케팅 캠페인 데이터 생성
    """
    # 매개변수
    n_campaigns = 50
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 5, 1)
    
    # 캠페인 유형 및 채널
    campaign_types = ['할인', '프로모션', '시즌', '신제품', '브랜드', '충성도']
    channels = ['소셜미디어', '이메일', '검색광고', '디스플레이광고', 'TV', '라디오', '오프라인']
    
    # 데이터 생성
    data = {
        'campaign_id': list(range(1, n_campaigns + 1)),
        'campaign_name': [f'캠페인 {i}' for i in range(1, n_campaigns + 1)],
        'type': [random.choice(campaign_types) for _ in range(n_campaigns)],
        'channel': [random.choice(channels) for _ in range(n_campaigns)],
        'start_date': [
            start_date + timedelta(days=random.randint(0, (end_date - start_date).days - 30))
            for _ in range(n_campaigns)
        ],
    }
    
    # 종료일 추가 (시작일로부터 7-60일)
    data['end_date'] = [
        data['start_date'][i] + timedelta(days=random.randint(7, 60))
        for i in range(n_campaigns)
    ]
    
    # 예산 배정
    data['budget'] = [random.randint(5000000, 50000000) for _ in range(n_campaigns)]
    
    # 실제 지출
    data['spend'] = [
        int(data['budget'][i] * random.uniform(0.8, 1.1))
        for i in range(n_campaigns)
    ]
    
    # 성과 지표
    data['impressions'] = [random.randint(50000, 1000000) for _ in range(n_campaigns)]
    data['clicks'] = [int(data['impressions'][i] * random.uniform(0.01, 0.05)) for i in range(n_campaigns)]
    data['conversions'] = [int(data['clicks'][i] * random.uniform(0.05, 0.15)) for i in range(n_campaigns)]
    
    # 계산된 성과 지표
    data['ctr'] = [data['clicks'][i] / data['impressions'][i] for i in range(n_campaigns)]
    data['cvr'] = [data['conversions'][i] / data['clicks'][i] for i in range(n_campaigns)]
    data['cpa'] = [data['spend'][i] / data['conversions'][i] if data['conversions'][i] > 0 else None for i in range(n_campaigns)]
    
    # ROI 계산 (평균 주문 가치 가정)
    aov = 150000  # 평균 주문 가치
    data['revenue'] = [data['conversions'][i] * aov for i in range(n_campaigns)]
    data['roi'] = [(data['revenue'][i] - data['spend'][i]) / data['spend'][i] for i in range(n_campaigns)]
    
    # DataFrame 생성
    df = pd.DataFrame(data)
    
    # 날짜 형식 변환
    df['start_date'] = pd.to_datetime(df['start_date']).dt.strftime('%Y-%m-%d')
    df['end_date'] = pd.to_datetime(df['end_date']).dt.strftime('%Y-%m-%d')
    
    # 저장
    file_path = os.path.join(DATA_DIR, 'marketing_data.csv')
    df.to_csv(file_path, index=False)
    print(f"마케팅 캠페인 데이터가 생성되었습니다: {file_path}")
    
    return df

if __name__ == "__main__":
    print("테스트용 샘플 데이터를 생성합니다...")
    generate_sales_data()
    generate_customer_data()
    generate_product_data()
    generate_marketing_campaign_data()
    print("모든 샘플 데이터 생성이 완료되었습니다.")
