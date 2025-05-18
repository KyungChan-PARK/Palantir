# predictive_modeling.py
"""
예측 모델링 워크플로우
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
import json
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    r2_score, mean_squared_error, mean_absolute_error,
    confusion_matrix, classification_report
)

# 모델 임포트
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from analysis.mcp_init import mcp
from analysis.tools.atoms.data_reader import read_data
from analysis.tools.atoms.data_processor import preprocess_data

@mcp.workflow(
    name="build_predictive_model",
    description="데이터를 읽고 예측 모델을 구축합니다. 자동으로 데이터 전처리, 특성 엔지니어링, 모델 선택 및 하이퍼파라미터 튜닝을 수행합니다."
)
async def build_predictive_model(
    file_path: str,
    target_column: str,
    problem_type: Optional[str] = None,  # "classification" 또는 "regression"
    features: Optional[List[str]] = None,
    test_size: float = 0.2,
    models_to_try: Optional[List[str]] = None,
    feature_engineering: Optional[List[str]] = None,
    hyperparameter_tuning: bool = True,
    output_dir: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    데이터를 읽고 예측 모델을 구축합니다.
    
    Parameters:
    -----------
    file_path : str
        읽을 파일의 경로 (상대 경로는 프로젝트 루트 기준)
    target_column : str
        예측할 타겟 열 이름
    problem_type : str, optional
        문제 유형 ("classification" 또는 "regression"), 지정하지 않으면 자동 감지
    features : List[str], optional
        사용할 특성 열 목록 (지정하지 않으면 타겟을 제외한 모든 열 사용)
    test_size : float, default=0.2
        테스트 세트 비율 (0.0 ~ 1.0)
    models_to_try : List[str], optional
        시도할 모델 목록, 지정하지 않으면 기본 모델 세트 사용
    feature_engineering : List[str], optional
        적용할 특성 엔지니어링 기법 목록
    hyperparameter_tuning : bool, default=True
        하이퍼파라미터 튜닝 수행 여부
    output_dir : str, optional
        결과를 저장할 디렉토리 경로
    params : Dict[str, Any], optional
        워크플로우 매개변수를 포함하는 딕셔너리
        
    Returns:
    --------
    Dict[str, Any]
        모델링 결과와 메타데이터를 포함하는 딕셔너리
    """
    # 기본값 설정
    params = params or {}
    feature_engineering = feature_engineering or ["remove_nulls", "normalize"]
    
    if output_dir is None:
        output_dir = os.path.join("C:\\Users\\packr\\OneDrive\\palantir", "output", "models")
    
    # 디렉토리가 없으면 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 모델 디렉토리
    models_dir = os.path.join(output_dir, "saved_models")
    os.makedirs(models_dir, exist_ok=True)
    
    # 시각화 디렉토리
    viz_dir = os.path.join("C:\\Users\\packr\\OneDrive\\palantir", "output", "viz", "models")
    os.makedirs(viz_dir, exist_ok=True)
    
    # 파일 이름에서 모델 이름 생성
    file_name = os.path.basename(file_path)
    model_name = f"model_{os.path.splitext(file_name)[0]}_{target_column}"
    
    try:
        # 1. 데이터 로드
        load_result = await read_data(file_path)
        
        if not load_result.get("success", False):
            return {
                "success": False,
                "error": f"데이터 로드 실패: {load_result.get('error', '알 수 없는 오류')}"
            }
        
        # 데이터 추출
        df = load_result["data"]
        
        # 타겟 열 확인
        if target_column not in df.columns:
            return {
                "success": False,
                "error": f"타겟 열을 찾을 수 없습니다: {target_column}"
            }
        
        # 특성 열 설정
        if features is None:
            features = [col for col in df.columns if col != target_column]
        else:
            # 존재하는 열만 선택
            features = [col for col in features if col in df.columns]
            
            if not features:
                return {
                    "success": False,
                    "error": "유효한 특성 열이 없습니다."
                }
        
        # 2. 문제 유형 감지 (분류 또는 회귀)
        if problem_type is None:
            # 타겟이 범주형이면 분류, 수치형이면 회귀
            if pd.api.types.is_numeric_dtype(df[target_column]):
                unique_count = df[target_column].nunique()
                total_count = df[target_column].count()
                
                # 고유값 비율이 낮으면 분류로 간주
                if unique_count / total_count < 0.05 or unique_count < 10:
                    problem_type = "classification"
                else:
                    problem_type = "regression"
            else:
                problem_type = "classification"
        
        # 3. 데이터 전처리
        # 데이터프레임 복사
        X = df[features].copy()
        y = df[target_column].copy()
        
        # 전처리 파이프라인 구성
        numeric_features = [col for col in features if pd.api.types.is_numeric_dtype(df[col])]
        categorical_features = [col for col in features if not pd.api.types.is_numeric_dtype(df[col])]
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]), numeric_features),
                ('cat', Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))
                ]), categorical_features)
            ]
        )
        
        # 4. 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # 5. 모델 선택 및 학습
        if models_to_try is None:
            if problem_type == "classification":
                models_to_try = ["logistic_regression", "decision_tree", "random_forest", "gradient_boosting"]
            else:  # regression
                models_to_try = ["linear_regression", "decision_tree", "random_forest", "gradient_boosting"]
        
        available_models = {
            # 분류 모델
            "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
            "decision_tree_clf": DecisionTreeClassifier(random_state=42),
            "random_forest_clf": RandomForestClassifier(random_state=42),
            "gradient_boosting_clf": GradientBoostingClassifier(random_state=42),
            "svm_clf": SVC(probability=True, random_state=42),
            "knn_clf": KNeighborsClassifier(),
            
            # 회귀 모델
            "linear_regression": LinearRegression(),
            "ridge": Ridge(random_state=42),
            "lasso": Lasso(random_state=42),
            "decision_tree_reg": DecisionTreeRegressor(random_state=42),
            "random_forest_reg": RandomForestRegressor(random_state=42),
            "gradient_boosting_reg": GradientBoostingRegressor(random_state=42),
            "svm_reg": SVR(),
            "knn_reg": KNeighborsRegressor()
        }
        
        # 문제 유형에 맞는 모델 필터링
        if problem_type == "classification":
            model_keys = [key for key in models_to_try if key in available_models and (key.endswith("_clf") or key == "logistic_regression")]
            if not model_keys:
                model_keys = ["logistic_regression", "random_forest_clf", "gradient_boosting_clf"]
        else:  # regression
            model_keys = [key for key in models_to_try if key in available_models and (key.endswith("_reg") or key == "linear_regression")]
            if not model_keys:
                model_keys = ["linear_regression", "random_forest_reg", "gradient_boosting_reg"]
        
        # 하이퍼파라미터 그리드
        param_grids = {
            "logistic_regression": {
                "classifier__C": [0.01, 0.1, 1, 10, 100],
                "classifier__solver": ["liblinear", "lbfgs"]
            },
            "decision_tree_clf": {
                "classifier__max_depth": [None, 5, 10, 15],
                "classifier__min_samples_split": [2, 5, 10]
            },
            "random_forest_clf": {
                "classifier__n_estimators": [50, 100, 200],
                "classifier__max_depth": [None, 10, 20]
            },
            "gradient_boosting_clf": {
                "classifier__n_estimators": [50, 100, 200],
                "classifier__learning_rate": [0.01, 0.1, 0.2]
            },
            "svm_clf": {
                "classifier__C": [0.1, 1, 10],
                "classifier__kernel": ["linear", "rbf"]
            },
            "knn_clf": {
                "classifier__n_neighbors": [3, 5, 7, 9],
                "classifier__weights": ["uniform", "distance"]
            },
            "linear_regression": {},  # 단순 선형 회귀는 튜닝할 하이퍼파라미터가 없음
            "ridge": {
                "regressor__alpha": [0.01, 0.1, 1, 10, 100]
            },
            "lasso": {
                "regressor__alpha": [0.01, 0.1, 1, 10, 100]
            },
            "decision_tree_reg": {
                "regressor__max_depth": [None, 5, 10, 15],
                "regressor__min_samples_split": [2, 5, 10]
            },
            "random_forest_reg": {
                "regressor__n_estimators": [50, 100, 200],
                "regressor__max_depth": [None, 10, 20]
            },
            "gradient_boosting_reg": {
                "regressor__n_estimators": [50, 100, 200],
                "regressor__learning_rate": [0.01, 0.1, 0.2]
            },
            "svm_reg": {
                "regressor__C": [0.1, 1, 10],
                "regressor__kernel": ["linear", "rbf"]
            },
            "knn_reg": {
                "regressor__n_neighbors": [3, 5, 7, 9],
                "regressor__weights": ["uniform", "distance"]
            }
        }
        
        # 모델 평가 결과 저장
        model_results = []
        
        for model_key in model_keys:
            model = available_models[model_key]
            
            # 파이프라인 생성
            if problem_type == "classification":
                pipeline = Pipeline([
                    ('preprocessor', preprocessor),
                    ('classifier', model)
                ])
                param_grid = param_grids.get(model_key, {})
                scoring = 'f1_weighted'
            else:  # regression
                pipeline = Pipeline([
                    ('preprocessor', preprocessor),
                    ('regressor', model)
                ])
                param_grid = param_grids.get(model_key, {})
                scoring = 'neg_mean_squared_error'
            
            # 하이퍼파라미터 튜닝
            best_estimator = None
            best_params = {}
            cv_scores = []
            
            if hyperparameter_tuning and param_grid:
                grid_search = GridSearchCV(
                    pipeline, param_grid, 
                    cv=5, scoring=scoring, 
                    n_jobs=-1, verbose=0
                )
                grid_search.fit(X_train, y_train)
                
                best_estimator = grid_search.best_estimator_
                best_params = grid_search.best_params_
                cv_scores = grid_search.cv_results_['mean_test_score']
            else:
                pipeline.fit(X_train, y_train)
                cv_results = cross_val_score(pipeline, X_train, y_train, cv=5, scoring=scoring)
                cv_scores = cv_results
                best_estimator = pipeline
            
            # 테스트 세트에서 평가
            y_pred = best_estimator.predict(X_test)
            
            # 모델 성능 평가
            performance_metrics = {}
            
            if problem_type == "classification":
                # 분류 메트릭
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                # 혼동 행렬
                cm = confusion_matrix(y_test, y_pred)
                
                performance_metrics = {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                    "confusion_matrix": cm.tolist()
                }
                
                # 시각화: 혼동 행렬
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=sorted(y.unique()),
                           yticklabels=sorted(y.unique()))
                plt.title(f'Confusion Matrix - {model_key}')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                
                viz_file = os.path.join(viz_dir, f"{model_name}_{model_key}_confusion_matrix.png")
                plt.tight_layout()
                plt.savefig(viz_file)
                plt.close()
                
                # 분류 보고서
                report = classification_report(y_test, y_pred, output_dict=True)
                performance_metrics["classification_report"] = report
                
            else:  # regression
                # 회귀 메트릭
                r2 = r2_score(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                
                performance_metrics = {
                    "r2_score": r2,
                    "mean_squared_error": mse,
                    "root_mean_squared_error": rmse,
                    "mean_absolute_error": mae
                }
                
                # 시각화: 예측 vs 실제
                plt.figure(figsize=(8, 6))
                plt.scatter(y_test, y_pred, alpha=0.5)
                plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
                plt.title(f'Actual vs Predicted - {model_key}')
                plt.xlabel('Actual')
                plt.ylabel('Predicted')
                
                viz_file = os.path.join(viz_dir, f"{model_name}_{model_key}_predicted_vs_actual.png")
                plt.tight_layout()
                plt.savefig(viz_file)
                plt.close()
                
                # 잔차 플롯
                residuals = y_test - y_pred
                plt.figure(figsize=(8, 6))
                plt.scatter(y_pred, residuals, alpha=0.5)
                plt.axhline(y=0, color='r', linestyle='--')
                plt.title(f'Residuals Plot - {model_key}')
                plt.xlabel('Predicted')
                plt.ylabel('Residuals')
                
                viz_file = os.path.join(viz_dir, f"{model_name}_{model_key}_residuals.png")
                plt.tight_layout()
                plt.savefig(viz_file)
                plt.close()
            
            # 특성 중요도 (가능한 경우)
            feature_importance = {}
            if hasattr(best_estimator, 'feature_importances_'):
                importances = best_estimator.feature_importances_
                feature_importance = dict(zip(features, importances))
                
                # 시각화: 특성 중요도
                plt.figure(figsize=(10, 6))
                sorted_idx = np.argsort(importances)
                plt.barh(range(len(sorted_idx)), importances[sorted_idx])
                plt.yticks(range(len(sorted_idx)), [features[i] for i in sorted_idx])
                plt.title(f'Feature Importance - {model_key}')
                
                viz_file = os.path.join(viz_dir, f"{model_name}_{model_key}_feature_importance.png")
                plt.tight_layout()
                plt.savefig(viz_file)
                plt.close()
            
            # 모델 저장
            model_file = os.path.join(models_dir, f"{model_name}_{model_key}.pkl")
            with open(model_file, 'wb') as f:
                pickle.dump(best_estimator, f)
            
            # 결과 저장
            model_results.append({
                "model_key": model_key,
                "performance": performance_metrics,
                "cv_scores": cv_scores.tolist() if isinstance(cv_scores, np.ndarray) else cv_scores,
                "best_params": best_params,
                "feature_importance": feature_importance,
                "model_file": model_file
            })
        
        # 최적 모델 선택
        if problem_type == "classification":
            best_model = max(model_results, key=lambda x: x["performance"]["f1_score"])
        else:  # regression
            best_model = max(model_results, key=lambda x: x["performance"]["r2_score"])
        
        # 결과 저장
        result_file = os.path.join(output_dir, f"{model_name}_results.json")
        
        # JSON 직렬화 가능한 형태로 변환
        serializable_results = []
        for result in model_results:
            result_copy = result.copy()
            # numpy 배열이나 JSON으로 직렬화할 수 없는 객체 처리
            if "cv_scores" in result_copy:
                if isinstance(result_copy["cv_scores"], np.ndarray):
                    result_copy["cv_scores"] = result_copy["cv_scores"].tolist()
            
            if "feature_importance" in result_copy:
                feature_imp = {}
                for feat, imp in result_copy["feature_importance"].items():
                    if isinstance(imp, np.ndarray):
                        feature_imp[feat] = imp.tolist()
                    else:
                        feature_imp[feat] = imp
                result_copy["feature_importance"] = feature_imp
            
            serializable_results.append(result_copy)
        
        result_data = {
            "file_name": file_name,
            "model_name": model_name,
            "problem_type": problem_type,
            "target_column": target_column,
            "features": features,
            "data_shape": df.shape,
            "test_size": test_size,
            "model_results": serializable_results,
            "best_model": best_model["model_key"]
        }
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        
        # 보고서 요약 Markdown 생성
        markdown_summary = generate_model_markdown_summary(result_data, model_name)
        markdown_file = os.path.join(output_dir, f"{model_name}_report.md")
        
        with open(markdown_file, 'w', encoding='utf-8') as f:
            f.write(markdown_summary)
        
        return {
            "success": True,
            "model_name": model_name,
            "problem_type": problem_type,
            "result_file": result_file,
            "markdown_summary": markdown_file,
            "best_model": {
                "model_key": best_model["model_key"],
                "performance": best_model["performance"],
                "model_file": best_model["model_file"]
            },
            "model_results": model_results
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": f"예측 모델 구축 중 오류 발생: {str(e)}"
        }

def generate_model_markdown_summary(result_data: Dict[str, Any], model_name: str) -> str:
    """
    모델링 결과의 Markdown 요약을 생성합니다.
    
    Parameters:
    -----------
    result_data : Dict[str, Any]
        모델링 결과 데이터
    model_name : str
        모델 이름
        
    Returns:
    --------
    str
        Markdown 형식의 요약
    """
    markdown = f"# 예측 모델링 보고서: {model_name}\n\n"
    
    # 모델 개요
    markdown += "## 1. 모델 개요\n\n"
    markdown += f"- **파일명**: {result_data['file_name']}\n"
    markdown += f"- **문제 유형**: {result_data['problem_type'].capitalize()}\n"
    markdown += f"- **타겟 변수**: {result_data['target_column']}\n"
    markdown += f"- **데이터 크기**: {result_data['data_shape'][0]}행 × {result_data['data_shape'][1]}열\n"
    markdown += f"- **테스트 세트 비율**: {result_data['test_size']}\n"
    markdown += f"- **특성 수**: {len(result_data['features'])}\n\n"
    
    # 최적 모델
    best_model_key = result_data['best_model']
    best_model = next((model for model in result_data['model_results'] if model['model_key'] == best_model_key), None)
    
    if best_model:
        markdown += "## 2. 최적 모델\n\n"
        markdown += f"- **모델**: {best_model['model_key']}\n"
        
        # 성능 지표
        markdown += "- **성능 지표**:\n"
        
        if result_data['problem_type'] == "classification":
            performance = best_model['performance']
            markdown += f"  - 정확도(Accuracy): {performance['accuracy']:.4f}\n"
            markdown += f"  - 정밀도(Precision): {performance['precision']:.4f}\n"
            markdown += f"  - 재현율(Recall): {performance['recall']:.4f}\n"
            markdown += f"  - F1 점수: {performance['f1_score']:.4f}\n"
        else:  # regression
            performance = best_model['performance']
            markdown += f"  - R² 점수: {performance['r2_score']:.4f}\n"
            markdown += f"  - 평균 제곱 오차(MSE): {performance['mean_squared_error']:.4f}\n"
            markdown += f"  - 평균 제곱근 오차(RMSE): {performance['root_mean_squared_error']:.4f}\n"
            markdown += f"  - 평균 절대 오차(MAE): {performance['mean_absolute_error']:.4f}\n"
        
        # 하이퍼파라미터
        if best_model.get('best_params'):
            markdown += "- **최적 하이퍼파라미터**:\n"
            for param, value in best_model['best_params'].items():
                markdown += f"  - {param}: {value}\n"
        
        # 특성 중요도
        if best_model.get('feature_importance'):
            markdown += "\n### 특성 중요도\n\n"
            markdown += "| 특성 | 중요도 |\n"
            markdown += "|------|--------|\n"
            
            # 중요도 순으로 정렬
            importances = best_model['feature_importance']
            sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
            
            for feature, importance in sorted_features:
                if isinstance(importance, list):
                    importance = sum(importance) / len(importance)
                markdown += f"| {feature} | {importance:.4f} |\n"
        
        markdown += "\n"
    
    # 모든 모델 성능 비교
    markdown += "## 3. 모델 성능 비교\n\n"
    
    if result_data['problem_type'] == "classification":
        markdown += "| 모델 | 정확도 | 정밀도 | 재현율 | F1 점수 |\n"
        markdown += "|------|--------|--------|--------|--------|\n"
        
        for model in result_data['model_results']:
            performance = model['performance']
            markdown += f"| {model['model_key']} | {performance['accuracy']:.4f} | {performance['precision']:.4f} | {performance['recall']:.4f} | {performance['f1_score']:.4f} |\n"
    
    else:  # regression
        markdown += "| 모델 | R² 점수 | MSE | RMSE | MAE |\n"
        markdown += "|------|---------|-----|------|-----|\n"
        
        for model in result_data['model_results']:
            performance = model['performance']
            markdown += f"| {model['model_key']} | {performance['r2_score']:.4f} | {performance['mean_squared_error']:.4f} | {performance['root_mean_squared_error']:.4f} | {performance['mean_absolute_error']:.4f} |\n"
    
    markdown += "\n"
    
    # 시각화 참조
    markdown += "## 4. 시각화\n\n"
    
    viz_dir = os.path.join("..", "..", "output", "viz", "models")
    
    if result_data['problem_type'] == "classification":
        markdown += "### 혼동 행렬\n\n"
        for model in result_data['model_results']:
            model_key = model['model_key']
            viz_file = f"{model_name}_{model_key}_confusion_matrix.png"
            markdown += f"- [{model_key} 혼동 행렬]({os.path.join(viz_dir, viz_file)})\n"
    
    else:  # regression
        markdown += "### 예측 vs 실제 값\n\n"
        for model in result_data['model_results']:
            model_key = model['model_key']
            viz_file = f"{model_name}_{model_key}_predicted_vs_actual.png"
            markdown += f"- [{model_key} 예측 vs 실제]({os.path.join(viz_dir, viz_file)})\n"
        
        markdown += "\n### 잔차 플롯\n\n"
        for model in result_data['model_results']:
            model_key = model['model_key']
            viz_file = f"{model_name}_{model_key}_residuals.png"
            markdown += f"- [{model_key} 잔차 플롯]({os.path.join(viz_dir, viz_file)})\n"
    
    markdown += "\n### 특성 중요도\n\n"
    for model in result_data['model_results']:
        if model.get('feature_importance'):
            model_key = model['model_key']
            viz_file = f"{model_name}_{model_key}_feature_importance.png"
            markdown += f"- [{model_key} 특성 중요도]({os.path.join(viz_dir, viz_file)})\n"
    
    # 결론 및 다음 단계
    markdown += "\n## 5. 결론 및 권장 사항\n\n"
    
    best_model_key = result_data['best_model']
    
    markdown += f"- **최적 모델**: {best_model_key}\n"
    
    if result_data['problem_type'] == "classification":
        best_performance = next((model['performance'] for model in result_data['model_results'] if model['model_key'] == best_model_key), None)
        if best_performance:
            if best_performance['f1_score'] >= 0.9:
                markdown += "- **모델 성능**: 매우 우수함 (F1 점수 >= 0.9)\n"
            elif best_performance['f1_score'] >= 0.8:
                markdown += "- **모델 성능**: 우수함 (F1 점수 >= 0.8)\n"
            elif best_performance['f1_score'] >= 0.7:
                markdown += "- **모델 성능**: 양호함 (F1 점수 >= 0.7)\n"
            else:
                markdown += "- **모델 성능**: 개선 필요 (F1 점수 < 0.7)\n"
    else:  # regression
        best_performance = next((model['performance'] for model in result_data['model_results'] if model['model_key'] == best_model_key), None)
        if best_performance:
            if best_performance['r2_score'] >= 0.9:
                markdown += "- **모델 성능**: 매우 우수함 (R² >= 0.9)\n"
            elif best_performance['r2_score'] >= 0.8:
                markdown += "- **모델 성능**: 우수함 (R² >= 0.8)\n"
            elif best_performance['r2_score'] >= 0.6:
                markdown += "- **모델 성능**: 양호함 (R² >= 0.6)\n"
            else:
                markdown += "- **모델 성능**: 개선 필요 (R² < 0.6)\n"
    
    markdown += "\n### 제안 사항\n\n"
    markdown += "1. 추가 특성 엔지니어링 고려\n"
    markdown += "2. 앙상블 기법 시도\n"
    markdown += "3. 하이퍼파라미터 튜닝 범위 확장\n"
    
    return markdown
