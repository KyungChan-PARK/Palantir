"""
Feedback Collector Module

This module provides tools for collecting and storing model performance feedback,
enabling a self-improving feedback loop for the data analysis system.
"""

import os
import json
import datetime
from common.path_utils import get_palantir_root
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
import logging

from analysis.mcp_init import mcp

# Setup logging
logger = logging.getLogger(__name__)

@mcp.tool(
    name="collect_model_feedback",
    description="Collect and store model performance feedback for self-improvement",
    tags=["feedback", "performance", "enhancement"]
)
async def collect_model_feedback(
    model_id: str,
    metrics: Dict[str, float],
    dataset_id: str,
    model_type: str,
    notes: Optional[str] = None,
    storage_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Collect and store model performance feedback to enable self-improvement loops.
    
    Parameters:
    -----------
    model_id : str
        Unique identifier for the model
    metrics : Dict[str, float]
        Performance metrics (e.g., {'accuracy': 0.95, 'f1_score': 0.92})
    dataset_id : str
        Identifier for the dataset used
    model_type : str
        Type of model (e.g., 'classification', 'regression')
    notes : str, optional
        Additional notes or context about this feedback
    storage_path : str, optional
        Path to store feedback data (default: output/feedback)
        
    Returns:
    --------
    Dict[str, Any]
        Result information including success status and feedback ID
    """
    try:
        # Setup default storage path
        if storage_path is None:
            storage_path = os.path.join(get_palantir_root().as_posix(), "output", "feedback")
        
        # Create directory if it doesn't exist
        os.makedirs(storage_path, exist_ok=True)
        
        # Generate feedback ID
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        feedback_id = f"feedback_{model_id}_{timestamp}"
        
        # Prepare feedback data
        feedback_data = {
            "feedback_id": feedback_id,
            "timestamp": timestamp,
            "model_id": model_id,
            "model_type": model_type,
            "dataset_id": dataset_id,
            "metrics": metrics,
            "notes": notes
        }
        
        # Save feedback to JSON file
        feedback_file = os.path.join(storage_path, f"{feedback_id}.json")
        with open(feedback_file, 'w', encoding='utf-8') as f:
            json.dump(feedback_data, f, ensure_ascii=False, indent=2)
        
        # Update feedback log (create or append)
        log_file = os.path.join(storage_path, "feedback_log.csv")
        
        # Flatten metrics for CSV storage
        flat_data = {
            "feedback_id": feedback_id,
            "timestamp": timestamp,
            "model_id": model_id,
            "model_type": model_type,
            "dataset_id": dataset_id
        }
        
        # Add metrics with prefix to avoid column collisions
        for metric_name, metric_value in metrics.items():
            flat_data[f"metric_{metric_name}"] = metric_value
        
        # Check if log file exists
        if os.path.exists(log_file):
            # Append to existing log
            log_df = pd.read_csv(log_file)
            new_row = pd.DataFrame([flat_data])
            updated_df = pd.concat([log_df, new_row], ignore_index=True)
            updated_df.to_csv(log_file, index=False)
        else:
            # Create new log
            pd.DataFrame([flat_data]).to_csv(log_file, index=False)
        
        logger.info(f"Successfully collected feedback with ID: {feedback_id}")
        
        return {
            "success": True,
            "feedback_id": feedback_id,
            "storage_path": storage_path,
            "feedback_file": feedback_file
        }
    
    except Exception as e:
        logger.error(f"Error collecting model feedback: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

@mcp.tool(
    name="analyze_performance_trends",
    description="Analyze performance trends from collected feedback",
    tags=["feedback", "analysis", "trends", "enhancement"]
)
async def analyze_performance_trends(
    model_id: Optional[str] = None,
    model_type: Optional[str] = None,
    metric_name: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    feedback_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze performance trends from collected model feedback.
    
    Parameters:
    -----------
    model_id : str, optional
        Filter by model ID
    model_type : str, optional
        Filter by model type
    metric_name : str, optional
        Specific metric to analyze (e.g., 'accuracy')
    start_date : str, optional
        Start date for analysis (format: YYYYMMDD)
    end_date : str, optional
        End date for analysis (format: YYYYMMDD)
    feedback_path : str, optional
        Path to feedback data (default: output/feedback)
        
    Returns:
    --------
    Dict[str, Any]
        Result including trend analysis and statistics
    """
    try:
        # Setup default feedback path
        if feedback_path is None:
            feedback_path = os.path.join(get_palantir_root().as_posix(), "output", "feedback")
        
        # Check if feedback log exists
        log_file = os.path.join(feedback_path, "feedback_log.csv")
        if not os.path.exists(log_file):
            return {
                "success": False,
                "error": "Feedback log not found. No feedback has been collected yet."
            }
        
        # Load feedback log
        log_df = pd.read_csv(log_file)
        
        # Apply filters
        filtered_df = log_df.copy()
        
        if model_id:
            filtered_df = filtered_df[filtered_df["model_id"] == model_id]
        
        if model_type:
            filtered_df = filtered_df[filtered_df["model_type"] == model_type]
        
        if start_date:
            # Convert timestamp string to datetime for comparison
            filtered_df["date"] = pd.to_datetime(filtered_df["timestamp"].str.split("_").str[0], format="%Y%m%d")
            filtered_df = filtered_df[filtered_df["date"] >= pd.to_datetime(start_date, format="%Y%m%d")]
        
        if end_date:
            if "date" not in filtered_df.columns:
                filtered_df["date"] = pd.to_datetime(filtered_df["timestamp"].str.split("_").str[0], format="%Y%m%d")
            filtered_df = filtered_df[filtered_df["date"] <= pd.to_datetime(end_date, format="%Y%m%d")]
        
        # Check if we have data after filtering
        if filtered_df.empty:
            return {
                "success": False,
                "error": "No feedback data matches the specified criteria."
            }
        
        # Analyze trends
        result = {
            "success": True,
            "count": len(filtered_df),
            "model_types": filtered_df["model_type"].value_counts().to_dict(),
            "date_range": {
                "min": filtered_df["timestamp"].min(),
                "max": filtered_df["timestamp"].max()
            },
            "metrics": {},
            "trends": {}
        }
        
        # Find all metric columns
        metric_columns = [col for col in filtered_df.columns if col.startswith("metric_")]
        
        # If specific metric is requested
        if metric_name:
            metric_col = f"metric_{metric_name}"
            if metric_col in metric_columns:
                metric_columns = [metric_col]
            else:
                return {
                    "success": False,
                    "error": f"Requested metric '{metric_name}' not found in feedback data."
                }
        
        # Analyze each metric
        for metric_col in metric_columns:
            # Extract the clean metric name
            clean_name = metric_col[7:]  # Remove "metric_" prefix
            
            # Calculate statistics
            stats = {
                "mean": filtered_df[metric_col].mean(),
                "median": filtered_df[metric_col].median(),
                "min": filtered_df[metric_col].min(),
                "max": filtered_df[metric_col].max(),
                "std": filtered_df[metric_col].std()
            }
            
            result["metrics"][clean_name] = stats
            
            # Calculate trend if we have timestamp information
            if "date" in filtered_df.columns and len(filtered_df) > 1:
                # Sort by date
                trend_df = filtered_df.sort_values("date")
                
                # Check for improvement or decline
                first_value = trend_df[metric_col].iloc[0]
                last_value = trend_df[metric_col].iloc[-1]
                change = last_value - first_value
                percent_change = (change / first_value * 100) if first_value != 0 else 0
                
                # Determine trend
                if change > 0:
                    trend = "improving"
                elif change < 0:
                    trend = "declining"
                else:
                    trend = "stable"
                
                result["trends"][clean_name] = {
                    "first_value": float(first_value),
                    "last_value": float(last_value),
                    "change": float(change),
                    "percent_change": float(percent_change),
                    "trend": trend
                }
        
        return result
    
    except Exception as e:
        logger.error(f"Error analyzing performance trends: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

@mcp.tool(
    name="identify_improvement_areas",
    description="Identify areas for model improvement based on feedback",
    tags=["feedback", "improvement", "enhancement"]
)
async def identify_improvement_areas(
    model_id: Optional[str] = None,
    model_type: Optional[str] = None,
    feedback_path: Optional[str] = None,
    improvement_threshold: float = 0.05
) -> Dict[str, Any]:
    """
    Identify areas for model improvement based on collected feedback.
    
    Parameters:
    -----------
    model_id : str, optional
        Filter by model ID
    model_type : str, optional
        Filter by model type
    feedback_path : str, optional
        Path to feedback data (default: output/feedback)
    improvement_threshold : float, optional
        Threshold for identifying improvement opportunities (default: 0.05)
        
    Returns:
    --------
    Dict[str, Any]
        Result including identified improvement areas and recommendations
    """
    try:
        # Setup default feedback path
        if feedback_path is None:
            feedback_path = os.path.join(get_palantir_root().as_posix(), "output", "feedback")
        
        # First, get performance trends
        trends_result = await analyze_performance_trends(
            model_id=model_id,
            model_type=model_type,
            feedback_path=feedback_path
        )
        
        if not trends_result.get("success", False):
            return trends_result
        
        # Identify improvement areas
        improvement_areas = []
        
        # Check metrics for improvement opportunities
        for metric_name, metric_stats in trends_result.get("metrics", {}).items():
            std_deviation = metric_stats.get("std", 0)
            mean_value = metric_stats.get("mean", 0)
            max_value = metric_stats.get("max", 0)
            
            # Calculate relative gap to optimal performance
            # For metrics like accuracy, higher is better (assuming all metrics follow this pattern)
            performance_gap = max_value - mean_value
            relative_gap = performance_gap / max_value if max_value > 0 else 0
            
            if relative_gap > improvement_threshold:
                # This metric has room for improvement
                improvement_areas.append({
                    "metric": metric_name,
                    "current_avg": mean_value,
                    "best_observed": max_value,
                    "gap": performance_gap,
                    "relative_gap": relative_gap,
                    "volatility": std_deviation,
                    "priority": "high" if relative_gap > 0.1 else "medium"
                })
        
        # Generate recommendations based on identified areas
        recommendations = []
        
        # General recommendations based on model type
        if model_type == "classification":
            recommendations.append({
                "type": "general",
                "description": "Consider class balancing techniques if dealing with imbalanced datasets",
                "priority": "medium"
            })
            recommendations.append({
                "type": "general",
                "description": "Evaluate precision-recall tradeoff for different decision thresholds",
                "priority": "medium"
            })
        elif model_type == "regression":
            recommendations.append({
                "type": "general",
                "description": "Check for non-linear relationships that might not be captured by linear models",
                "priority": "medium"
            })
            recommendations.append({
                "type": "general",
                "description": "Consider feature scaling or normalization for better convergence",
                "priority": "medium"
            })
        
        # Specific recommendations based on metrics
        for area in improvement_areas:
            metric = area["metric"]
            priority = area["priority"]
            
            if metric == "accuracy":
                recommendations.append({
                    "type": "specific",
                    "metric": metric,
                    "description": "Consider feature engineering to improve model accuracy",
                    "priority": priority
                })
                recommendations.append({
                    "type": "specific",
                    "metric": metric,
                    "description": "Try ensemble methods to boost accuracy",
                    "priority": priority
                })
            elif metric == "f1_score":
                recommendations.append({
                    "type": "specific",
                    "metric": metric,
                    "description": "Adjust class weights to improve balance between precision and recall",
                    "priority": priority
                })
            elif metric == "r2_score":
                recommendations.append({
                    "type": "specific",
                    "metric": metric,
                    "description": "Consider more complex model architectures to capture non-linear relationships",
                    "priority": priority
                })
                recommendations.append({
                    "type": "specific",
                    "metric": metric,
                    "description": "Feature selection may help improve R² score by removing noise",
                    "priority": priority
                })
            elif metric == "rmse" or metric == "mae":
                recommendations.append({
                    "type": "specific",
                    "metric": metric,
                    "description": f"Focus on reducing outlier influence to improve {metric.upper()}",
                    "priority": priority
                })
            else:
                recommendations.append({
                    "type": "specific",
                    "metric": metric,
                    "description": f"Review hyperparameter optimization to improve {metric}",
                    "priority": priority
                })
        
        return {
            "success": True,
            "improvement_areas": improvement_areas,
            "recommendations": recommendations,
            "model_stats": {
                "model_id": model_id,
                "model_type": model_type,
                "feedback_count": trends_result.get("count", 0)
            }
        }
    
    except Exception as e:
        logger.error(f"Error identifying improvement areas: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }
