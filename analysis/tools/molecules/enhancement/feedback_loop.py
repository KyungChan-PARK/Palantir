"""
Feedback Loop Workflow Module

This module implements a self-improving feedback loop workflow that collects performance metrics,
analyzes trends, and provides actionable recommendations for model improvement.
"""

import os
import json
import datetime
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
import logging

from analysis.mcp_init import mcp
from analysis.tools.atoms.enhancement.feedback_collector import (
    collect_model_feedback,
    analyze_performance_trends,
    identify_improvement_areas
)

# Setup logging
logger = logging.getLogger(__name__)

@mcp.workflow(
    name="self_improving_workflow",
    description="Execute a complete self-improving feedback loop for model optimization"
)
async def self_improving_workflow(
    model_id: str,
    performance_metrics: Dict[str, float],
    dataset_id: str,
    model_type: str,
    model_params: Optional[Dict[str, Any]] = None,
    analysis_window: Optional[int] = None,
    improvement_threshold: float = 0.05,
    feedback_path: Optional[str] = None,
    notes: Optional[str] = None
) -> Dict[str, Any]:
    """
    Execute a complete self-improving feedback loop for model optimization.
    
    Parameters:
    -----------
    model_id : str
        Unique identifier for the model
    performance_metrics : Dict[str, float]
        Performance metrics (e.g., {'accuracy': 0.95, 'f1_score': 0.92})
    dataset_id : str
        Identifier for the dataset used
    model_type : str
        Type of model (e.g., 'classification', 'regression')
    model_params : Dict[str, Any], optional
        Model parameters and hyperparameters
    analysis_window : int, optional
        Number of recent feedback entries to analyze (default: all)
    improvement_threshold : float, optional
        Threshold for identifying improvement opportunities (default: 0.05)
    feedback_path : str, optional
        Path to store feedback data (default: output/feedback)
    notes : str, optional
        Additional notes or context about this feedback
        
    Returns:
    --------
    Dict[str, Any]
        Result including feedback ID, improvement areas, and recommendations
    """
    try:
        # 1. Collect and store model performance feedback
        feedback_result = await collect_model_feedback(
            model_id=model_id,
            metrics=performance_metrics,
            dataset_id=dataset_id,
            model_type=model_type,
            notes=notes,
            storage_path=feedback_path
        )
        
        if not feedback_result.get("success", False):
            return feedback_result  # Return error
        
        feedback_id = feedback_result.get("feedback_id")
        
        # Prepare result
        result = {
            "success": True,
            "feedback_id": feedback_id,
            "model_id": model_id,
            "model_type": model_type,
            "timestamp": datetime.datetime.now().isoformat(),
            "current_metrics": performance_metrics
        }
        
        # Add model parameters if provided
        if model_params:
            # Store model parameters with feedback
            params_file = os.path.join(feedback_result.get("storage_path"), f"{feedback_id}_params.json")
            with open(params_file, 'w', encoding='utf-8') as f:
                json.dump(model_params, f, ensure_ascii=False, indent=2)
                
            result["model_params"] = model_params
            result["params_file"] = params_file
        
        # 2. Analyze performance trends
        trends_params = {
            "model_id": model_id,
            "model_type": model_type,
            "feedback_path": feedback_path
        }
        
        if analysis_window is not None:
            # Calculate start date based on analysis window (in days)
            current_date = datetime.datetime.now()
            start_date = (current_date - datetime.timedelta(days=analysis_window)).strftime("%Y%m%d")
            trends_params["start_date"] = start_date
        
        trends_result = await analyze_performance_trends(**trends_params)
        
        if not trends_result.get("success", False):
            # We can still continue even if trend analysis fails
            logger.warning(f"Performance trend analysis failed: {trends_result.get('error')}")
            result["trends_analysis"] = {
                "success": False,
                "error": trends_result.get("error")
            }
        else:
            result["trends_analysis"] = trends_result
        
        # 3. Identify improvement areas
        if trends_result.get("success", False):
            improvement_result = await identify_improvement_areas(
                model_id=model_id,
                model_type=model_type,
                feedback_path=feedback_path,
                improvement_threshold=improvement_threshold
            )
            
            if improvement_result.get("success", False):
                result["improvement_areas"] = improvement_result.get("improvement_areas", [])
                result["recommendations"] = improvement_result.get("recommendations", [])
            else:
                logger.warning(f"Improvement area identification failed: {improvement_result.get('error')}")
                result["improvement_areas"] = []
                result["recommendations"] = []
        else:
            # Generate basic recommendations even without trend analysis
            result["improvement_areas"] = []
            result["recommendations"] = [
                {
                    "type": "general",
                    "description": "Continue collecting performance data to enable trend analysis",
                    "priority": "high"
                }
            ]
        
        # 4. Generate improvement report
        report = generate_improvement_report(result)
        
        # Save report
        report_path = os.path.join(feedback_result.get("storage_path"), f"{feedback_id}_report.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        result["report"] = report_path
        
        return result
    
    except Exception as e:
        logger.error(f"Error in self-improving workflow: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "feedback_id": feedback_id if 'feedback_id' in locals() else None
        }

@mcp.workflow(
    name="process_optimization_workflow",
    description="Optimize analysis process based on historical performance"
)
async def process_optimization_workflow(
    process_id: str,
    process_type: str,
    performance_metrics: Dict[str, float],
    process_params: Dict[str, Any],
    optimization_target: str,
    feedback_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Optimize analysis process based on historical performance data.
    
    Parameters:
    -----------
    process_id : str
        Unique identifier for the process
    process_type : str
        Type of process (e.g., 'eda', 'modeling', 'decision_support')
    performance_metrics : Dict[str, float]
        Performance metrics (e.g., {'execution_time': 5.2, 'memory_usage': 128.5})
    process_params : Dict[str, Any]
        Process parameters and settings
    optimization_target : str
        Target metric to optimize (e.g., 'execution_time', 'memory_usage')
    feedback_path : str, optional
        Path to store feedback data (default: output/feedback/process)
        
    Returns:
    --------
    Dict[str, Any]
        Result including optimization recommendations
    """
    try:
        # Setup default feedback path
        if feedback_path is None:
            feedback_path = os.path.join("C:\\Users\\packr\\OneDrive\\palantir", "output", "feedback", "process")
        
        # Create directory if it doesn't exist
        os.makedirs(feedback_path, exist_ok=True)
        
        # Generate process feedback ID
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        feedback_id = f"process_{process_id}_{timestamp}"
        
        # Prepare feedback data
        feedback_data = {
            "feedback_id": feedback_id,
            "timestamp": timestamp,
            "process_id": process_id,
            "process_type": process_type,
            "metrics": performance_metrics,
            "parameters": process_params,
            "optimization_target": optimization_target
        }
        
        # Save feedback to JSON file
        feedback_file = os.path.join(feedback_path, f"{feedback_id}.json")
        with open(feedback_file, 'w', encoding='utf-8') as f:
            json.dump(feedback_data, f, ensure_ascii=False, indent=2)
        
        # Update process feedback log (create or append)
        log_file = os.path.join(feedback_path, "process_feedback_log.csv")
        
        # Flatten metrics and parameters for CSV storage
        flat_data = {
            "feedback_id": feedback_id,
            "timestamp": timestamp,
            "process_id": process_id,
            "process_type": process_type,
            "optimization_target": optimization_target
        }
        
        # Add metrics with prefix
        for metric_name, metric_value in performance_metrics.items():
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
        
        # Load historical process data
        process_history = []
        
        # Iterate through all JSON files in the feedback directory
        for filename in os.listdir(feedback_path):
            if filename.endswith(".json") and filename.startswith(f"process_{process_id}_"):
                file_path = os.path.join(feedback_path, filename)
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    try:
                        process_data = json.load(f)
                        process_history.append(process_data)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse process feedback file: {filename}")
        
        # Sort process history by timestamp
        process_history.sort(key=lambda x: x.get("timestamp", ""))
        
        # Analyze historical data
        target_values = []
        param_values = {}
        
        for entry in process_history:
            metrics = entry.get("metrics", {})
            params = entry.get("parameters", {})
            
            if optimization_target in metrics:
                target_values.append(metrics[optimization_target])
                
                # Track parameter values
                for param_name, param_value in params.items():
                    if param_name not in param_values:
                        param_values[param_name] = []
                    
                    param_values[param_name].append(param_value)
        
        # Generate recommendations
        recommendations = []
        current_value = performance_metrics.get(optimization_target)
        
        if current_value is not None and len(target_values) > 0:
            min_value = min(target_values)
            best_idx = target_values.index(min_value)
            
            # Check if current run is better or worse than best
            if current_value > min_value:  # Assuming lower is better
                # Current run is worse, recommend parameters from best run
                best_params = process_history[best_idx].get("parameters", {})
                
                recommendations.append({
                    "type": "parameter_optimization",
                    "description": f"Consider using parameters from best performing run (ID: {process_history[best_idx].get('feedback_id')})",
                    "best_value": min_value,
                    "current_value": current_value,
                    "improvement_potential": current_value - min_value,
                    "recommended_parameters": best_params,
                    "priority": "high"
                })
            else:
                # Current run is the best so far
                recommendations.append({
                    "type": "confirmation",
                    "description": "Current parameters are optimal based on historical data",
                    "priority": "low"
                })
        
        # Parameter correlation analysis
        if len(target_values) > 5:  # Need enough data points
            param_correlations = {}
            
            for param_name, values in param_values.items():
                if len(values) == len(target_values):
                    # Check if parameter has numeric values
                    if all(isinstance(v, (int, float)) for v in values):
                        correlation = np.corrcoef(values, target_values)[0, 1]
                        param_correlations[param_name] = correlation
            
            # Find parameters with strong correlation (absolute value > 0.5)
            strong_correlations = {k: v for k, v in param_correlations.items() if abs(v) > 0.5}
            
            if strong_correlations:
                correlation_insights = []
                
                for param_name, corr_value in sorted(strong_correlations.items(), key=lambda x: abs(x[1]), reverse=True):
                    direction = "lower" if corr_value > 0 else "higher"  # Assuming lower target is better
                    correlation_insights.append(f"{param_name} should be {direction} (correlation: {corr_value:.2f})")
                
                recommendations.append({
                    "type": "correlation_analysis",
                    "description": "Parameters with strong correlation to performance",
                    "correlations": strong_correlations,
                    "insights": correlation_insights,
                    "priority": "medium"
                })
        
        # If we don't have enough data yet
        if len(target_values) < 5:
            recommendations.append({
                "type": "data_collection",
                "description": "Continue collecting process data to enable robust analysis",
                "current_samples": len(target_values),
                "recommended_samples": 5,
                "priority": "high"
            })
        
        # Generate report
        report = generate_optimization_report(
            process_id=process_id,
            process_type=process_type,
            optimization_target=optimization_target,
            current_metrics=performance_metrics,
            recommendations=recommendations,
            history_length=len(process_history)
        )
        
        # Save report
        report_path = os.path.join(feedback_path, f"{feedback_id}_report.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        return {
            "success": True,
            "feedback_id": feedback_id,
            "process_id": process_id,
            "process_type": process_type,
            "optimization_target": optimization_target,
            "current_value": current_value,
            "history_length": len(process_history),
            "recommendations": recommendations,
            "report": report_path
        }
    
    except Exception as e:
        logger.error(f"Error in process optimization workflow: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

@mcp.workflow(
    name="parameter_optimization_workflow",
    description="Optimize model hyperparameters based on historical performance"
)
async def parameter_optimization_workflow(
    model_id: str,
    model_type: str,
    current_params: Dict[str, Any],
    performance_metrics: Dict[str, float],
    optimization_target: str,
    param_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    feedback_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Optimize model hyperparameters based on historical performance data.
    
    Parameters:
    -----------
    model_id : str
        Unique identifier for the model
    model_type : str
        Type of model (e.g., 'classification', 'regression')
    current_params : Dict[str, Any]
        Current model hyperparameters
    performance_metrics : Dict[str, float]
        Performance metrics (e.g., {'accuracy': 0.95, 'f1_score': 0.92})
    optimization_target : str
        Target metric to optimize (e.g., 'accuracy', 'f1_score')
    param_ranges : Dict[str, Tuple[float, float]], optional
        Valid ranges for parameters to optimize
    feedback_path : str, optional
        Path to store feedback data (default: output/feedback/params)
        
    Returns:
    --------
    Dict[str, Any]
        Result including parameter optimization recommendations
    """
    try:
        # Setup default feedback path
        if feedback_path is None:
            feedback_path = os.path.join("C:\\Users\\packr\\OneDrive\\palantir", "output", "feedback", "params")
        
        # Create directory if it doesn't exist
        os.makedirs(feedback_path, exist_ok=True)
        
        # Generate parameter feedback ID
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        feedback_id = f"params_{model_id}_{timestamp}"
        
        # Prepare feedback data
        feedback_data = {
            "feedback_id": feedback_id,
            "timestamp": timestamp,
            "model_id": model_id,
            "model_type": model_type,
            "parameters": current_params,
            "metrics": performance_metrics,
            "optimization_target": optimization_target
        }
        
        # Save feedback to JSON file
        feedback_file = os.path.join(feedback_path, f"{feedback_id}.json")
        with open(feedback_file, 'w', encoding='utf-8') as f:
            json.dump(feedback_data, f, ensure_ascii=False, indent=2)
        
        # Check if optimization target exists in metrics
        if optimization_target not in performance_metrics:
            return {
                "success": False,
                "error": f"Optimization target '{optimization_target}' not found in performance metrics"
            }
        
        current_value = performance_metrics[optimization_target]
        
        # Load historical parameter data
        param_history = []
        
        # Iterate through all JSON files in the feedback directory
        for filename in os.listdir(feedback_path):
            if filename.endswith(".json") and filename.startswith(f"params_{model_id}_"):
                file_path = os.path.join(feedback_path, filename)
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    try:
                        param_data = json.load(f)
                        
                        # Check if this entry has the target metric
                        if param_data.get("optimization_target") == optimization_target and \
                           optimization_target in param_data.get("metrics", {}):
                            param_history.append(param_data)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse parameter feedback file: {filename}")
        
        # Sort parameter history by timestamp
        param_history.sort(key=lambda x: x.get("timestamp", ""))
        
        # Initialize results
        result = {
            "success": True,
            "feedback_id": feedback_id,
            "model_id": model_id,
            "model_type": model_type,
            "optimization_target": optimization_target,
            "current_value": current_value,
            "history_length": len(param_history),
            "recommendations": []
        }
        
        # If we don't have enough historical data
        if len(param_history) < 3:
            result["recommendations"].append({
                "type": "data_collection",
                "description": "Not enough historical data for parameter optimization",
                "current_samples": len(param_history),
                "recommended_samples": 3,
                "priority": "high"
            })
            
            # If we have param ranges, suggest exploration
            if param_ranges:
                exploration_suggestions = []
                
                for param_name, (min_val, max_val) in param_ranges.items():
                    current_val = current_params.get(param_name)
                    
                    if current_val is not None and isinstance(current_val, (int, float)):
                        # Simple suggestion: try midpoint between current and boundaries
                        if current_val < max_val - (max_val - min_val) * 0.1:
                            new_val = (current_val + max_val) / 2
                            exploration_suggestions.append({
                                "parameter": param_name,
                                "current_value": current_val,
                                "suggested_value": new_val,
                                "direction": "increase",
                                "confidence": "low"
                            })
                        elif current_val > min_val + (max_val - min_val) * 0.1:
                            new_val = (current_val + min_val) / 2
                            exploration_suggestions.append({
                                "parameter": param_name,
                                "current_value": current_val,
                                "suggested_value": new_val,
                                "direction": "decrease",
                                "confidence": "low"
                            })
                
                if exploration_suggestions:
                    result["recommendations"].append({
                        "type": "parameter_exploration",
                        "description": "Initial parameter exploration suggestions",
                        "suggestions": exploration_suggestions,
                        "priority": "medium"
                    })
            
            # Return early - not enough data for thorough analysis
            return result
        
        # Extract parameter and metric data
        parameter_data = {}
        target_values = []
        
        for entry in param_history:
            params = entry.get("parameters", {})
            metrics = entry.get("metrics", {})
            
            if optimization_target in metrics:
                target_values.append(metrics[optimization_target])
                
                # Track parameter values
                for param_name, param_value in params.items():
                    if isinstance(param_value, (int, float)):  # Only analyze numeric parameters
                        if param_name not in parameter_data:
                            parameter_data[param_name] = []
                        
                        parameter_data[param_name].append(param_value)
        
        # Find the best historical performance
        # Assuming higher values are better (e.g., accuracy, f1)
        # This can be inverted for metrics where lower is better
        is_higher_better = True
        if optimization_target in ['error', 'loss', 'rmse', 'mae', 'mse']:
            is_higher_better = False
        
        if is_higher_better:
            best_value = max(target_values)
            best_idx = target_values.index(best_value)
        else:
            best_value = min(target_values)
            best_idx = target_values.index(best_value)
        
        best_params = param_history[best_idx].get("parameters", {})
        
        # Compare current performance with best
        if (is_higher_better and current_value < best_value) or \
           (not is_higher_better and current_value > best_value):
            # Current performance is worse than best
            result["recommendations"].append({
                "type": "best_parameters",
                "description": f"Consider using parameters from best performing run (ID: {param_history[best_idx].get('feedback_id')})",
                "best_value": best_value,
                "current_value": current_value,
                "improvement_potential": abs(current_value - best_value),
                "recommended_parameters": best_params,
                "priority": "high"
            })
        else:
            # Current performance is the best so far
            result["recommendations"].append({
                "type": "confirmation",
                "description": "Current parameters are optimal based on historical data",
                "priority": "low"
            })
        
        # Parameter correlation analysis
        param_correlations = {}
        
        for param_name, values in parameter_data.items():
            if len(values) == len(target_values):
                correlation = np.corrcoef(values, target_values)[0, 1]
                
                # Adjust correlation sign based on optimization direction
                if not is_higher_better:
                    correlation = -correlation
                
                param_correlations[param_name] = correlation
        
        # Find parameters with strong correlation (absolute value > 0.3)
        strong_correlations = {k: v for k, v in param_correlations.items() if abs(v) > 0.3}
        
        if strong_correlations:
            correlation_insights = []
            parameter_suggestions = []
            
            for param_name, corr_value in sorted(strong_correlations.items(), key=lambda x: abs(x[1]), reverse=True):
                current_val = current_params.get(param_name)
                
                if current_val is not None and isinstance(current_val, (int, float)):
                    # Determine direction to adjust
                    direction = "increase" if corr_value > 0 else "decrease"
                    
                    # Check parameter ranges if provided
                    can_adjust = True
                    adjustment_factor = 0.1  # Default adjustment factor
                    
                    if param_ranges and param_name in param_ranges:
                        min_val, max_val = param_ranges[param_name]
                        
                        if direction == "increase" and current_val >= max_val:
                            can_adjust = False
                        elif direction == "decrease" and current_val <= min_val:
                            can_adjust = False
                        
                        # Adjust based on range
                        param_range = max_val - min_val
                        adjustment_factor = min(0.1, param_range * 0.1 / current_val)
                    
                    if can_adjust:
                        # Calculate suggested value
                        suggested_val = None
                        
                        if direction == "increase":
                            suggested_val = current_val * (1 + adjustment_factor)
                        else:
                            suggested_val = current_val * (1 - adjustment_factor)
                        
                        # Round to appropriate precision
                        if isinstance(current_val, int):
                            suggested_val = int(round(suggested_val))
                        else:
                            suggested_val = round(suggested_val, 6)
                        
                        # Enforce parameter ranges
                        if param_ranges and param_name in param_ranges:
                            min_val, max_val = param_ranges[param_name]
                            suggested_val = max(min_val, min(suggested_val, max_val))
                        
                        correlation_insights.append(
                            f"{param_name} should be {direction}d (correlation: {corr_value:.2f})"
                        )
                        
                        parameter_suggestions.append({
                            "parameter": param_name,
                            "current_value": current_val,
                            "suggested_value": suggested_val,
                            "direction": direction,
                            "correlation": corr_value,
                            "confidence": "high" if abs(corr_value) > 0.6 else "medium"
                        })
            
            if parameter_suggestions:
                result["recommendations"].append({
                    "type": "parameter_tuning",
                    "description": "Parameter adjustment recommendations based on correlation analysis",
                    "correlations": strong_correlations,
                    "insights": correlation_insights,
                    "suggestions": parameter_suggestions,
                    "priority": "high"
                })
        
        # Generate report
        report = generate_parameter_optimization_report(
            model_id=model_id,
            model_type=model_type,
            optimization_target=optimization_target,
            current_params=current_params,
            current_value=current_value,
            best_params=best_params,
            best_value=best_value,
            recommendations=result["recommendations"],
            history_length=len(param_history)
        )
        
        # Save report
        report_path = os.path.join(feedback_path, f"{feedback_id}_report.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        result["report"] = report_path
        
        return result
    
    except Exception as e:
        logger.error(f"Error in parameter optimization workflow: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

# Utility functions for report generation

def generate_improvement_report(result: Dict[str, Any]) -> str:
    """Generate a markdown report for model improvement"""
    model_id = result.get("model_id", "unknown")
    model_type = result.get("model_type", "unknown")
    timestamp = result.get("timestamp", datetime.datetime.now().isoformat())
    
    # Format timestamp for display
    try:
        dt = datetime.datetime.fromisoformat(timestamp)
        formatted_date = dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        formatted_date = timestamp
    
    report = f"# Model Improvement Report: {model_id}\n\n"
    report += f"**Model Type:** {model_type}\n"
    report += f"**Report Date:** {formatted_date}\n\n"
    
    # Current performance metrics
    report += "## Current Performance\n\n"
    
    metrics = result.get("current_metrics", {})
    
    report += "| Metric | Value |\n"
    report += "|--------|-------|\n"
    
    for metric_name, metric_value in metrics.items():
        report += f"| {metric_name} | {metric_value:.4f} |\n"
    
    report += "\n"
    
    # Performance trends if available
    trends_analysis = result.get("trends_analysis", {})
    
    if trends_analysis.get("success", False):
        report += "## Performance Trends\n\n"
        
        trends = trends_analysis.get("trends", {})
        
        if trends:
            report += "| Metric | Current | Best | Change | Trend |\n"
            report += "|--------|---------|------|--------|-------|\n"
            
            for metric_name, trend_data in trends.items():
                current = trend_data.get("last_value", 0)
                best = trend_data.get("first_value", 0)
                change = trend_data.get("percent_change", 0)
                trend = trend_data.get("trend", "stable")
                
                # Format trend indicator
                trend_indicator = "→"
                if trend == "improving":
                    trend_indicator = "↑"
                elif trend == "declining":
                    trend_indicator = "↓"
                
                report += f"| {metric_name} | {current:.4f} | {best:.4f} | {change:.2f}% | {trend_indicator} |\n"
        
        report += "\n"
    
    # Improvement areas
    improvement_areas = result.get("improvement_areas", [])
    
    if improvement_areas:
        report += "## Identified Improvement Areas\n\n"
        
        for area in improvement_areas:
            metric = area.get("metric", "unknown")
            current_avg = area.get("current_avg", 0)
            best_observed = area.get("best_observed", 0)
            gap = area.get("gap", 0)
            relative_gap = area.get("relative_gap", 0)
            
            report += f"### {metric.capitalize()}\n\n"
            report += f"- Current Average: {current_avg:.4f}\n"
            report += f"- Best Observed: {best_observed:.4f}\n"
            report += f"- Improvement Gap: {gap:.4f} ({relative_gap*100:.2f}%)\n\n"
    
    # Recommendations
    recommendations = result.get("recommendations", [])
    
    if recommendations:
        report += "## Recommendations\n\n"
        
        # Group by priority
        priority_groups = {
            "high": [],
            "medium": [],
            "low": []
        }
        
        for rec in recommendations:
            priority = rec.get("priority", "medium")
            priority_groups[priority].append(rec)
        
        # Process each priority group
        for priority in ["high", "medium", "low"]:
            if priority_groups[priority]:
                report += f"### {priority.capitalize()} Priority\n\n"
                
                for rec in priority_groups[priority]:
                    rec_type = rec.get("type", "")
                    description = rec.get("description", "")
                    
                    report += f"**{rec_type.replace('_', ' ').capitalize()}:** {description}\n\n"
                    
                    # Handle specific recommendation types
                    if rec_type == "specific":
                        report += f"Target Metric: {rec.get('metric', '')}\n\n"
                    elif rec_type == "parameter_optimization":
                        params = rec.get("recommended_parameters", {})
                        
                        if params:
                            report += "Recommended parameters:\n\n"
                            report += "```\n"
                            
                            for param_name, param_value in params.items():
                                report += f"{param_name}: {param_value}\n"
                            
                            report += "```\n\n"
                    elif rec_type == "correlation_analysis":
                        insights = rec.get("insights", [])
                        
                        if insights:
                            report += "Correlation insights:\n\n"
                            
                            for insight in insights:
                                report += f"- {insight}\n"
                            
                            report += "\n"
    else:
        report += "## Recommendations\n\n"
        report += "_No recommendations available. Continue collecting performance data._\n\n"
    
    # Next steps
    report += "## Next Steps\n\n"
    
    if recommendations:
        high_priority = priority_groups.get("high", [])
        
        if high_priority:
            report += "1. Implement high priority recommendations:\n"
            
            for i, rec in enumerate(high_priority[:3]):
                report += f"   - {rec.get('description')}\n"
        else:
            report += "1. Consider implementing medium priority recommendations\n"
    
    report += "2. Continue tracking model performance\n"
    report += "3. Re-run improvement analysis after implementing changes\n"
    
    return report

def generate_optimization_report(
    process_id: str,
    process_type: str,
    optimization_target: str,
    current_metrics: Dict[str, float],
    recommendations: List[Dict[str, Any]],
    history_length: int
) -> str:
    """Generate a markdown report for process optimization"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"# Process Optimization Report: {process_id}\n\n"
    report += f"**Process Type:** {process_type}\n"
    report += f"**Report Date:** {timestamp}\n"
    report += f"**Optimization Target:** {optimization_target}\n"
    report += f"**Historical Samples:** {history_length}\n\n"
    
    # Current metrics
    report += "## Current Performance\n\n"
    
    report += "| Metric | Value |\n"
    report += "|--------|-------|\n"
    
    for metric_name, metric_value in current_metrics.items():
        if isinstance(metric_value, float):
            report += f"| {metric_name} | {metric_value:.4f} |\n"
        else:
            report += f"| {metric_name} | {metric_value} |\n"
    
    report += "\n"
    
    # Recommendations
    if recommendations:
        report += "## Optimization Recommendations\n\n"
        
        # Group by priority
        priority_groups = {
            "high": [],
            "medium": [],
            "low": []
        }
        
        for rec in recommendations:
            priority = rec.get("priority", "medium")
            priority_groups[priority].append(rec)
        
        # Process each priority group
        for priority in ["high", "medium", "low"]:
            if priority_groups[priority]:
                report += f"### {priority.capitalize()} Priority\n\n"
                
                for rec in priority_groups[priority]:
                    rec_type = rec.get("type", "").replace("_", " ").capitalize()
                    description = rec.get("description", "")
                    
                    report += f"**{rec_type}:** {description}\n\n"
                    
                    # Handle specific recommendation types
                    if rec_type.lower() == "parameter optimization":
                        best_value = rec.get("best_value", 0)
                        current_value = rec.get("current_value", 0)
                        improvement = rec.get("improvement_potential", 0)
                        
                        report += f"- Current Value: {current_value:.4f}\n"
                        report += f"- Best Value: {best_value:.4f}\n"
                        report += f"- Potential Improvement: {improvement:.4f}\n\n"
                        
                        params = rec.get("recommended_parameters", {})
                        
                        if params:
                            report += "Recommended parameters:\n\n"
                            report += "```\n"
                            
                            for param_name, param_value in params.items():
                                report += f"{param_name}: {param_value}\n"
                            
                            report += "```\n\n"
                    elif rec_type.lower() == "correlation analysis":
                        insights = rec.get("insights", [])
                        
                        if insights:
                            report += "Parameter insights:\n\n"
                            
                            for insight in insights:
                                report += f"- {insight}\n"
                            
                            report += "\n"
                    elif rec_type.lower() == "data collection":
                        current = rec.get("current_samples", 0)
                        recommended = rec.get("recommended_samples", 5)
                        
                        report += f"- Current Samples: {current}\n"
                        report += f"- Recommended Samples: {recommended}\n"
                        report += f"- Additional Needed: {max(0, recommended - current)}\n\n"
    else:
        report += "## Optimization Recommendations\n\n"
        report += "_No recommendations available. Continue collecting process data._\n\n"
    
    # Next steps
    report += "## Next Steps\n\n"
    
    if history_length < 5:
        report += "1. Continue collecting process performance data\n"
        report += "2. Run optimization analysis after collecting at least 5 samples\n"
    else:
        report += "1. Implement recommended optimizations\n"
        report += "2. Monitor performance after changes\n"
        report += "3. Run optimization analysis again to validate improvements\n"
    
    return report

def generate_parameter_optimization_report(
    model_id: str,
    model_type: str,
    optimization_target: str,
    current_params: Dict[str, Any],
    current_value: float,
    best_params: Dict[str, Any],
    best_value: float,
    recommendations: List[Dict[str, Any]],
    history_length: int
) -> str:
    """Generate a markdown report for parameter optimization"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"# Parameter Optimization Report: {model_id}\n\n"
    report += f"**Model Type:** {model_type}\n"
    report += f"**Report Date:** {timestamp}\n"
    report += f"**Optimization Target:** {optimization_target}\n"
    report += f"**Historical Samples:** {history_length}\n\n"
    
    # Current performance
    report += "## Current Performance\n\n"
    report += f"**Current {optimization_target}:** {current_value:.4f}\n"
    report += f"**Best {optimization_target}:** {best_value:.4f}\n"
    report += f"**Difference:** {abs(current_value - best_value):.4f}\n\n"
    
    # Current parameters
    report += "## Current Parameters\n\n"
    report += "```\n"
    
    for param_name, param_value in current_params.items():
        report += f"{param_name}: {param_value}\n"
    
    report += "```\n\n"
    
    # Best parameters (if different from current)
    if best_value != current_value:
        report += "## Best Performing Parameters\n\n"
        report += "```\n"
        
        for param_name, param_value in best_params.items():
            report += f"{param_name}: {param_value}\n"
        
        report += "```\n\n"
    
    # Parameter change suggestions
    for rec in recommendations:
        if rec.get("type") == "parameter_tuning":
            report += "## Parameter Adjustment Recommendations\n\n"
            
            suggestions = rec.get("suggestions", [])
            
            if suggestions:
                report += "| Parameter | Current Value | Suggested Value | Direction | Confidence |\n"
                report += "|-----------|---------------|-----------------|-----------|------------|\n"
                
                for suggestion in suggestions:
                    param = suggestion.get("parameter", "")
                    current = suggestion.get("current_value", "")
                    suggested = suggestion.get("suggested_value", "")
                    direction = suggestion.get("direction", "")
                    confidence = suggestion.get("confidence", "medium")
                    
                    report += f"| {param} | {current} | {suggested} | {direction} | {confidence} |\n"
                
                report += "\n"
                
                # Add correlation insights
                insights = rec.get("insights", [])
                
                if insights:
                    report += "**Correlation Insights:**\n\n"
                    
                    for insight in insights:
                        report += f"- {insight}\n"
                    
                    report += "\n"
    
    # Other recommendations
    other_recs = [r for r in recommendations if r.get("type") != "parameter_tuning"]
    
    if other_recs:
        report += "## Additional Recommendations\n\n"
        
        for rec in other_recs:
            rec_type = rec.get("type", "").replace("_", " ").capitalize()
            description = rec.get("description", "")
            
            report += f"**{rec_type}:** {description}\n\n"
    
    # Next steps
    report += "## Next Steps\n\n"
    
    if history_length < 5:
        report += "1. Continue collecting parameter performance data\n"
        report += "2. Run optimization analysis after collecting at least 5 samples\n"
    else:
        # Check if we have parameter tuning recommendations
        has_tuning = any(r.get("type") == "parameter_tuning" for r in recommendations)
        
        if has_tuning:
            report += "1. Implement suggested parameter adjustments\n"
            report += "2. Test model performance with new parameters\n"
            report += "3. Run optimization analysis again to validate improvements\n"
        else:
            report += "1. Maintain current parameters as they appear optimal\n"
            report += "2. Consider exploring new parameter combinations for further improvement\n"
            report += "3. Continue monitoring model performance\n"
    
    return report
