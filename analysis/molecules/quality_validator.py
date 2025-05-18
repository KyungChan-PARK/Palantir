"""
품질 검증기 클래스 - 데이터 품질 검증을 위한 모듈

이 모듈은 Great Expectations를 기반으로 하는 데이터 품질 검증 기능을 제공합니다.
JSON으로 정의된 기대치를 기반으로 데이터프레임 또는 CSV 파일의 품질을 검증합니다.
"""

import os
import json
import pandas as pd
from datetime import datetime
import logging

# 로깅 설정
logger = logging.getLogger(__name__)

class QualityValidator:
    """
    데이터 품질 검증을 위한 클래스
    Great Expectations 없이 독립적으로 동작하는 간소화된 버전
    """
    
    def __init__(self):
        """
        QualityValidator 초기화
        """
        self.logger = logger
    
    def validate_pandas_df(self, df, expectations):
        """
        판다스 데이터프레임에 대한 검증 실행
        
        Args:
            df (pandas.DataFrame): 검증할 데이터프레임
            expectations (dict): 컬럼 및 테이블 기대치 정의
            
        Returns:
            dict: 검증 결과
        """
        # 검증 결과를 저장할 딕셔너리
        results = {
            'summary': {
                'total_expectations': 0,
                'passed_expectations': 0,
                'failed_expectations': 0,
                'success_percent': 0.0
            },
            'column_results': {},
            'table_results': {},
            'details': []
        }
        
        # 컬럼 기대치 검증
        if 'column_expectations' in expectations:
            for column, column_expectations in expectations['column_expectations'].items():
                column_result = self._validate_column(df, column, column_expectations, results['details'])
                results['column_results'][column] = column_result
                
                # 전체 통계 업데이트
                results['summary']['total_expectations'] += column_result['total_expectations']
                results['summary']['passed_expectations'] += column_result['passed_expectations']
                results['summary']['failed_expectations'] += column_result['failed_expectations']
        
        # 테이블 기대치 검증
        if 'table_expectations' in expectations:
            table_result = self._validate_table(df, expectations['table_expectations'], results['details'])
            results['table_results'] = table_result
            
            # 전체 통계 업데이트
            results['summary']['total_expectations'] += table_result['total_expectations']
            results['summary']['passed_expectations'] += table_result['passed_expectations']
            results['summary']['failed_expectations'] += table_result['failed_expectations']
        
        # 전체 성공률 계산
        if results['summary']['total_expectations'] > 0:
            results['summary']['success_percent'] = (
                results['summary']['passed_expectations'] / results['summary']['total_expectations'] * 100
            )
        
        return results
    
    def validate_csv(self, csv_path, expectations):
        """
        CSV 파일에 대한 검증 실행
        
        Args:
            csv_path (str): 검증할 CSV 파일 경로
            expectations (dict): 컬럼 및 테이블 기대치 정의
            
        Returns:
            dict: 검증 결과
        """
        try:
            df = pd.read_csv(csv_path)
            return self.validate_pandas_df(df, expectations)
        except Exception as e:
            self.logger.error(f"CSV 파일 검증 중 오류 발생: {str(e)}")
            return {
                'summary': {
                    'total_expectations': 0,
                    'passed_expectations': 0,
                    'failed_expectations': 0,
                    'success_percent': 0.0,
                    'error': str(e)
                },
                'column_results': {},
                'table_results': {},
                'details': [{
                    'expectation_type': 'csv_readable',
                    'success': False,
                    'message': f"CSV 파일을 읽을 수 없음: {str(e)}"
                }]
            }
    
    def validate_from_expectation_file(self, data, expectation_file_path):
        """
        JSON 파일에서 기대치를 로드하여 데이터 검증
        
        Args:
            data (pandas.DataFrame or str): 검증할 데이터프레임 또는 CSV 파일 경로
            expectation_file_path (str): 기대치 정의 JSON 파일 경로
            
        Returns:
            dict: 검증 결과
        """
        try:
            with open(expectation_file_path, 'r', encoding='utf-8') as f:
                expectations = json.load(f)
            
            if isinstance(data, pd.DataFrame):
                return self.validate_pandas_df(data, expectations)
            elif isinstance(data, str) and os.path.isfile(data):
                return self.validate_csv(data, expectations)
            else:
                raise ValueError("data는 판다스 데이터프레임 또는 CSV 파일 경로여야 합니다.")
        except Exception as e:
            self.logger.error(f"기대치 파일로부터 검증 중 오류 발생: {str(e)}")
            return {
                'summary': {
                    'total_expectations': 0,
                    'passed_expectations': 0,
                    'failed_expectations': 0,
                    'success_percent': 0.0,
                    'error': str(e)
                },
                'column_results': {},
                'table_results': {},
                'details': [{
                    'expectation_type': 'expectations_file_readable',
                    'success': False,
                    'message': f"기대치 파일을 처리할 수 없음: {str(e)}"
                }]
            }
    
    def _validate_column(self, df, column, expectations, details):
        """
        단일 컬럼에 대한 기대치 검증
        
        Args:
            df (pandas.DataFrame): 데이터프레임
            column (str): 컬럼 이름
            expectations (dict): 컬럼 기대치
            details (list): 상세 결과를 추가할 리스트
            
        Returns:
            dict: 컬럼 검증 결과
        """
        result = {
            'total_expectations': 0,
            'passed_expectations': 0,
            'failed_expectations': 0,
            'success_rate': 0.0
        }
        
        # 컬럼 존재 여부 확인
        if column not in df.columns:
            result['total_expectations'] += 1
            result['failed_expectations'] += 1
            details.append({
                'expectation_type': 'column_exists',
                'column': column,
                'success': False,
                'message': f"컬럼 '{column}'이 데이터셋에 존재하지 않습니다."
            })
            return result
        
        # 각 기대치 검증
        for expectation_type, expectation_value in expectations.items():
            result['total_expectations'] += 1
            
            # 기대치 유형별 검증 로직
            if expectation_type == 'not_null':
                success = self._validate_not_null(df, column, expectation_value)
                message = f"컬럼 '{column}'의 모든 값이 null이 아닙니다." if success else \
                         f"컬럼 '{column}'에 null 값이 존재합니다."
            
            elif expectation_type == 'unique':
                success = self._validate_unique(df, column, expectation_value)
                message = f"컬럼 '{column}'의 모든 값이 고유합니다." if success else \
                         f"컬럼 '{column}'에 중복 값이 존재합니다."
            
            elif expectation_type == 'allowed_values':
                success = self._validate_allowed_values(df, column, expectation_value)
                message = f"컬럼 '{column}'의 모든 값이 허용된 값 목록에 포함됩니다." if success else \
                         f"컬럼 '{column}'에 허용되지 않은 값이 존재합니다."
            
            elif expectation_type == 'min_value':
                success = self._validate_min_value(df, column, expectation_value)
                message = f"컬럼 '{column}'의 모든 값이 최소값 {expectation_value} 이상입니다." if success else \
                         f"컬럼 '{column}'에 최소값 {expectation_value}보다 작은 값이 존재합니다."
            
            elif expectation_type == 'max_value':
                success = self._validate_max_value(df, column, expectation_value)
                message = f"컬럼 '{column}'의 모든 값이 최대값 {expectation_value} 이하입니다." if success else \
                         f"컬럼 '{column}'에 최대값 {expectation_value}보다 큰 값이 존재합니다."
            
            else:
                success = False
                message = f"지원되지 않는 기대치 유형: {expectation_type}"
            
            # 결과 기록
            if success:
                result['passed_expectations'] += 1
            else:
                result['failed_expectations'] += 1
            
            details.append({
                'expectation_type': expectation_type,
                'column': column,
                'success': success,
                'message': message
            })
        
        # 컬럼 성공률 계산
        if result['total_expectations'] > 0:
            result['success_rate'] = result['passed_expectations'] / result['total_expectations'] * 100
        
        return result
    
    def _validate_table(self, df, expectations, details):
        """
        테이블 수준 기대치 검증
        
        Args:
            df (pandas.DataFrame): 데이터프레임
            expectations (dict): 테이블 기대치
            details (list): 상세 결과를 추가할 리스트
            
        Returns:
            dict: 테이블 검증 결과
        """
        result = {
            'total_expectations': 0,
            'passed_expectations': 0,
            'failed_expectations': 0,
            'success_rate': 0.0
        }
        
        # 각 기대치 검증
        for expectation_type, expectation_value in expectations.items():
            result['total_expectations'] += 1
            
            # 기대치 유형별 검증 로직
            if expectation_type == 'row_count_min':
                success = len(df) >= expectation_value
                message = f"데이터셋의 행 수({len(df)})가 최소 요구치({expectation_value})를 충족합니다." if success else \
                         f"데이터셋의 행 수({len(df)})가 최소 요구치({expectation_value})보다 적습니다."
            
            elif expectation_type == 'row_count_max':
                success = len(df) <= expectation_value
                message = f"데이터셋의 행 수({len(df)})가 최대 제한({expectation_value})을 초과하지 않습니다." if success else \
                         f"데이터셋의 행 수({len(df)})가 최대 제한({expectation_value})을 초과합니다."
            
            elif expectation_type == 'required_columns':
                missing_columns = [col for col in expectation_value if col not in df.columns]
                success = len(missing_columns) == 0
                message = "모든 필수 컬럼이 데이터셋에 존재합니다." if success else \
                         f"다음 필수 컬럼이 누락되었습니다: {', '.join(missing_columns)}"
            
            else:
                success = False
                message = f"지원되지 않는 테이블 기대치 유형: {expectation_type}"
            
            # 결과 기록
            if success:
                result['passed_expectations'] += 1
            else:
                result['failed_expectations'] += 1
            
            details.append({
                'expectation_type': expectation_type,
                'success': success,
                'message': message
            })
        
        # 테이블 성공률 계산
        if result['total_expectations'] > 0:
            result['success_rate'] = result['passed_expectations'] / result['total_expectations'] * 100
        
        return result
    
    # 개별 검증 메서드들
    def _validate_not_null(self, df, column, expectation_value):
        """null 값 없음 검증"""
        if expectation_value:
            return df[column].notna().all()
        return True
    
    def _validate_unique(self, df, column, expectation_value):
        """고유값 검증"""
        if expectation_value:
            return df[column].is_unique
        return True
    
    def _validate_allowed_values(self, df, column, expectation_value):
        """허용된 값 목록 검증"""
        return df[column].isin(expectation_value).all()
    
    def _validate_min_value(self, df, column, expectation_value):
        """최소값 검증"""
        return (df[column] >= expectation_value).all()
    
    def _validate_max_value(self, df, column, expectation_value):
        """최대값 검증"""
        return (df[column] <= expectation_value).all()
