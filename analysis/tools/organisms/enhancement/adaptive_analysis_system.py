"""
Adaptive Analysis System

This module implements an adaptive analysis system that leverages performance feedback
and interactive visualizations to create a self-improving data analysis environment.
"""

import os
import json
import datetime
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
import logging
import uuid

from analysis.mcp_init import mcp
from analysis.tools.atoms.data_reader import read_data
from analysis.tools.molecules.exploratory_analysis import exploratory_analysis
from analysis.tools.molecules.predictive_modeling import build_predictive_model
from analysis.tools.molecules.enhancement.feedback_loop import (
    self_improving_workflow,
    process_optimization_workflow,
    parameter_optimization_workflow
)
from analysis.tools.molecules.enhancement.interactive_visualization import (
    create_interactive_dashboard,
    update_dashboard,
    export_dashboard
)

# Setup logging
logger = logging.getLogger(__name__)

@mcp.system(
    name="adaptive_analysis_system",
    description="Run an adaptive analysis system with feedback loops and visualizations"
)
async def adaptive_analysis_system(
    data_source: str,
    analysis_type: str,
    output_dir: Optional[str] = None,
    parameters: Optional[Dict[str, Any]] = None,
    optimization_target: Optional[str] = None,
    visualization_config: Optional[Dict[str, Any]] = None,
    history_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run an adaptive analysis system with feedback loops and visualizations.
    
    Parameters:
    -----------
    data_source : str
        Path to the data file to analyze
    analysis_type : str
        Type of analysis to perform ('eda', 'modeling', 'decision_support')
    output_dir : str, optional
        Directory to save output files
    parameters : Dict[str, Any], optional
        Analysis parameters (defaults to recommended settings based on history)
    optimization_target : str, optional
        Metric to optimize (e.g., 'execution_time', 'accuracy', 'r2_score')
    visualization_config : Dict[str, Any], optional
        Dashboard visualization configuration
    history_id : str, optional
        ID for tracking analysis history (generated if not provided)
        
    Returns:
    --------
    Dict[str, Any]
        Result including analysis results, optimization feedback, and visualization outputs
    """
    try:
        # Generate history ID if not provided
        if history_id is None:
            history_id = f"analysis_{uuid.uuid4().hex[:8]}"
        
        # Set default output directory if not provided
        if output_dir is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join("C:\\Users\\packr\\OneDrive\\palantir", "output", "adaptive", timestamp)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create subdirectories
        analysis_dir = os.path.join(output_dir, "analysis")
        feedback_dir = os.path.join(output_dir, "feedback")
        dashboard_dir = os.path.join(output_dir, "dashboard")
        
        os.makedirs(analysis_dir, exist_ok=True)
        os.makedirs(feedback_dir, exist_ok=True)
        os.makedirs(dashboard_dir, exist_ok=True)
        
        # Initialize result dictionary
        result = {
            "success": True,
            "history_id": history_id,
            "analysis_type": analysis_type,
            "timestamp": datetime.datetime.now().isoformat(),
            "output_dir": output_dir
        }
        
        # Check if parameters provided, otherwise load recommended parameters
        if parameters is None:
            parameters = await get_recommended_parameters(
                analysis_type=analysis_type,
                optimization_target=optimization_target,
                history_id=history_id
            )
        
        # Record start time for performance tracking
        start_time = datetime.datetime.now()
        
        # Perform analysis based on type
        analysis_result = None
        
        if analysis_type == "eda":
            # Perform exploratory data analysis
            analysis_result = await exploratory_analysis(
                file_path=data_source,
                output_dir=analysis_dir,
                **parameters
            )
        
        elif analysis_type == "modeling":
            # Make sure target column is specified
            if "target_column" not in parameters:
                return {
                    "success": False,
                    "error": "Target column must be specified for modeling analysis"
                }
            
            # Perform predictive modeling
            analysis_result = await build_predictive_model(
                file_path=data_source,
                output_dir=analysis_dir,
                **parameters
            )
        
        elif analysis_type == "decision_support":
            # Import here to avoid circular import
            from analysis.tools.organisms.decision_support_system import decision_support
            
            # Make sure question is specified
            if "question" not in parameters:
                return {
                    "success": False,
                    "error": "Question must be specified for decision support analysis"
                }
            
            # Perform decision support analysis
            question = parameters.pop("question")
            data_sources = parameters.pop("data_sources", [data_source])
            
            analysis_result = await decision_support(
                question=question,
                data_sources=data_sources,
                output_dir=analysis_dir,
                params=parameters
            )
        
        else:
            return {
                "success": False,
                "error": f"Unsupported analysis type: {analysis_type}"
            }
        
        # Record end time and calculate execution time
        end_time = datetime.datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        # Add analysis result to result dictionary
        result["analysis_result"] = analysis_result
        
        # Check if analysis was successful
        if not analysis_result.get("success", False):
            result["success"] = False
            result["error"] = analysis_result.get("error", "Analysis failed")
            return result
        
        # Extract performance metrics based on analysis type
        performance_metrics = {
            "execution_time": execution_time
        }
        
        if analysis_type == "modeling":
            # Extract model performance metrics
            best_model = analysis_result.get("best_model", {})
            model_performance = best_model.get("performance", {})
            
            # Add model metrics to performance metrics
            for metric_name, metric_value in model_performance.items():
                performance_metrics[metric_name] = metric_value
        
        # Process optimization if optimization target is specified
        if optimization_target:
            # Use process optimization for execution time
            if optimization_target == "execution_time":
                optimization_result = await process_optimization_workflow(
                    process_id=history_id,
                    process_type=analysis_type,
                    performance_metrics=performance_metrics,
                    process_params=parameters,
                    optimization_target=optimization_target,
                    feedback_path=feedback_dir
                )
            
            # Use parameter optimization for model metrics
            elif analysis_type == "modeling" and optimization_target in performance_metrics:
                optimization_result = await parameter_optimization_workflow(
                    model_id=history_id,
                    model_type=parameters.get("problem_type", "unknown"),
                    current_params=parameters,
                    performance_metrics=performance_metrics,
                    optimization_target=optimization_target,
                    feedback_path=feedback_dir
                )
            
            # Use self-improving workflow for other cases
            else:
                optimization_result = await self_improving_workflow(
                    model_id=history_id,
                    performance_metrics=performance_metrics,
                    dataset_id=os.path.basename(data_source),
                    model_type=analysis_type,
                    model_params=parameters,
                    feedback_path=feedback_dir
                )
            
            # Add optimization result to result dictionary
            result["optimization_result"] = optimization_result
        
        # Create visualization if configured
        if visualization_config:
            # Load data
            data_result = await read_data(data_source)
            
            if data_result.get("success", False):
                df = data_result.get("data")
                
                # Create components based on analysis type and results
                components = generate_dashboard_components(
                    analysis_type=analysis_type,
                    analysis_result=analysis_result,
                    optimization_result=optimization_result if optimization_target else None,
                    visualization_config=visualization_config
                )
                
                # Create dashboard
                dashboard_result = await create_interactive_dashboard(
                    data_source=df,
                    dashboard_title=f"{analysis_type.capitalize()} Analysis Dashboard",
                    components=components,
                    output_dir=dashboard_dir,
                    theme=visualization_config.get("theme", "light")
                )
                
                # Add dashboard result to result dictionary
                result["dashboard_result"] = dashboard_result
        
        # Generate report
        report = generate_adaptive_analysis_report(
            analysis_type=analysis_type,
            analysis_result=analysis_result,
            optimization_result=result.get("optimization_result"),
            dashboard_result=result.get("dashboard_result"),
            performance_metrics=performance_metrics,
            parameters=parameters
        )
        
        # Save report
        report_file = os.path.join(output_dir, "adaptive_analysis_report.md")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        result["report_file"] = report_file
        
        return result
    
    except Exception as e:
        logger.error(f"Error in adaptive analysis system: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

@mcp.system(
    name="web_dashboard_system",
    description="Create and serve a web-based interactive dashboard for data analysis"
)
async def web_dashboard_system(
    data_source: str,
    dashboard_title: str,
    dashboard_config: Dict[str, Any],
    output_dir: Optional[str] = None,
    port: int = 8050,
    open_browser: bool = True
) -> Dict[str, Any]:
    """
    Create and serve a web-based interactive dashboard for data analysis.
    
    Parameters:
    -----------
    data_source : str
        Path to the data file to visualize
    dashboard_title : str
        Title of the dashboard
    dashboard_config : Dict[str, Any]
        Dashboard configuration with layout and components
    output_dir : str, optional
        Directory to save dashboard files
    port : int, optional
        Port to serve the dashboard (default: 8050)
    open_browser : bool, optional
        Whether to open the browser automatically
        
    Returns:
    --------
    Dict[str, Any]
        Result including dashboard URL and process information
    """
    try:
        # Check if Plotly is available for web dashboard
        try:
            import plotly
            import plotly.express as px
            import plotly.graph_objects as go
        except ImportError:
            return {
                "success": False,
                "error": "Plotly is required for web dashboards. Install with 'pip install plotly dash'"
            }
        
        # Check if Dash is available
        try:
            import dash
            from dash import dcc, html
        except ImportError:
            return {
                "success": False,
                "error": "Dash is required for web dashboards. Install with 'pip install dash'"
            }
        
        # Set default output directory if not provided
        if output_dir is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join("C:\\Users\\packr\\OneDrive\\palantir", "output", "web_dashboard", timestamp)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data
        data_result = await read_data(data_source)
        
        if not data_result.get("success", False):
            return {
                "success": False,
                "error": f"Failed to read data: {data_result.get('error', 'Unknown error')}"
            }
        
        df = data_result.get("data")
        
        # Extract dashboard layout and components
        layout = dashboard_config.get("layout", {})
        components = dashboard_config.get("components", [])
        theme = dashboard_config.get("theme", "light")
        
        # Generate dashboard application code
        app_code = generate_dash_app_code(
            dashboard_title=dashboard_title,
            layout=layout,
            components=components,
            theme=theme
        )
        
        # Save dashboard application code
        app_file = os.path.join(output_dir, "app.py")
        with open(app_file, 'w', encoding='utf-8') as f:
            f.write(app_code)
        
        # Save data file for dashboard
        data_file = os.path.join(output_dir, "dashboard_data.csv")
        df.to_csv(data_file, index=False)
        
        # Generate run script
        run_script = f"""
import os
import sys
from app import app

if __name__ == '__main__':
    app.run_server(debug=False, port={port})
"""
        
        # Save run script
        run_file = os.path.join(output_dir, "run_dashboard.py")
        with open(run_file, 'w', encoding='utf-8') as f:
            f.write(run_script)
        
        # Generate batch file for Windows
        batch_content = f"""@echo off
echo Starting Dashboard Server on http://localhost:{port}
cd /d "%~dp0"
python run_dashboard.py
"""
        
        # Save batch file
        batch_file = os.path.join(output_dir, "run_dashboard.bat")
        with open(batch_file, 'w', encoding='utf-8') as f:
            f.write(batch_content)
        
        # Generate README file
        readme_content = f"""# {dashboard_title}

## Dashboard Overview
This is an interactive web dashboard created using Plotly Dash.

## How to Run the Dashboard
1. Make sure you have the required packages installed:
   ```
   pip install dash plotly pandas
   ```

2. Run the dashboard server:
   - On Windows: Double-click the `run_dashboard.bat` file
   - On any platform: Run `python run_dashboard.py` in this directory

3. Open your browser and navigate to:
   ```
   http://localhost:{port}
   ```

## Dashboard Contents
{len(components)} interactive components are included in this dashboard.

## Data Source
The dashboard uses data from the file: {os.path.basename(data_source)}
"""
        
        # Save README file
        readme_file = os.path.join(output_dir, "README.md")
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        # Start dashboard server process
        import subprocess
        import sys
        
        # Use Popen to start process without waiting
        dashboard_process = subprocess.Popen(
            [sys.executable, run_file],
            cwd=output_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Sleep briefly to allow server to start
        import time
        time.sleep(2)
        
        # Open browser if requested
        if open_browser:
            import webbrowser
            webbrowser.open(f"http://localhost:{port}")
        
        return {
            "success": True,
            "dashboard_title": dashboard_title,
            "output_dir": output_dir,
            "url": f"http://localhost:{port}",
            "app_file": app_file,
            "run_file": run_file,
            "batch_file": batch_file,
            "readme_file": readme_file,
            "process_id": dashboard_process.pid,
            "component_count": len(components)
        }
    
    except Exception as e:
        logger.error(f"Error in web dashboard system: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

@mcp.system(
    name="scheduled_update_system",
    description="Schedule automatic updates for analysis and dashboards"
)
async def scheduled_update_system(
    data_source: str,
    update_config: Dict[str, Any],
    dashboard_file: Optional[str] = None,
    report_email: Optional[str] = None,
    schedule: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create a scheduled update system for automatic analysis and dashboard updates.
    
    Parameters:
    -----------
    data_source : str
        Path to the data file or data source pattern
    update_config : Dict[str, Any]
        Configuration for updates (analysis type, parameters, etc.)
    dashboard_file : str, optional
        Path to dashboard configuration file to update
    report_email : str, optional
        Email address to send reports to
    schedule : Dict[str, Any], optional
        Schedule configuration (frequency, start time, etc.)
        
    Returns:
    --------
    Dict[str, Any]
        Result including scheduled task information
    """
    try:
        # Set default schedule if not provided
        if schedule is None:
            schedule = {
                "frequency": "daily",
                "time": "00:00",
                "days": ["Mon", "Tue", "Wed", "Thu", "Fri"],
            }
        
        # Create output directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("C:\\Users\\packr\\OneDrive\\palantir", "output", "scheduled", timestamp)
        os.makedirs(output_dir, exist_ok=True)
        
        # Create config file for scheduled task
        config = {
            "data_source": data_source,
            "update_config": update_config,
            "dashboard_file": dashboard_file,
            "report_email": report_email,
            "schedule": schedule,
            "created_at": datetime.datetime.now().isoformat(),
            "last_run": None,
            "next_run": calculate_next_run(schedule),
            "enabled": True
        }
        
        # Save config
        config_file = os.path.join(output_dir, "schedule_config.json")
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        # Generate update script
        update_script = generate_update_script(config)
        
        # Save update script
        script_file = os.path.join(output_dir, "run_update.py")
        with open(script_file, 'w', encoding='utf-8') as f:
            f.write(update_script)
        
        # Generate batch file for Windows
        batch_content = """@echo off
echo Running scheduled update
cd /d "%~dp0"
python run_update.py
"""
        
        # Save batch file
        batch_file = os.path.join(output_dir, "run_update.bat")
        with open(batch_file, 'w', encoding='utf-8') as f:
            f.write(batch_content)
        
        # Create scheduled task (Windows)
        task_name = f"AdaptiveAnalysisUpdate_{timestamp}"
        
        # Construct task schedule string
        if schedule["frequency"] == "daily":
            schedule_string = f'/sc DAILY /st {schedule["time"]}'
        elif schedule["frequency"] == "weekly":
            day_mapping = {
                "Mon": "MON", "Tue": "TUE", "Wed": "WED", "Thu": "THU",
                "Fri": "FRI", "Sat": "SAT", "Sun": "SUN"
            }
            days = ",".join([day_mapping.get(day, day) for day in schedule["days"]])
            schedule_string = f'/sc WEEKLY /d "{days}" /st {schedule["time"]}'
        else:
            schedule_string = f'/sc ONCE /st {schedule["time"]} /sd {datetime.datetime.now().strftime("%m/%d/%Y")}'
        
        # Generate task command (but don't execute - this is for reference)
        task_command = f'schtasks /create /tn "{task_name}" {schedule_string} /tr "{batch_file}"'
        
        # Generate instructions for manual scheduling
        instructions = f"""# Scheduled Update System

## Configuration
- Data source: {data_source}
- Update frequency: {schedule['frequency']}
- Update time: {schedule['time']}
- Days: {', '.join(schedule['days']) if 'days' in schedule else 'Every day'}

## How to Schedule
To create a scheduled task in Windows, run the following command in an Administrator Command Prompt:

```
{task_command}
```

## Manual Execution
To run the update manually, double-click the `run_update.bat` file or run:
```
python run_update.py
```

## Update Configuration
The update configuration is stored in `schedule_config.json`. You can edit this file to change the update settings.

## Dashboard Updates
{f'The dashboard at {dashboard_file} will be updated with new data.' if dashboard_file else 'No dashboard will be updated.'}

## Email Reports
{f'Reports will be sent to {report_email}.' if report_email else 'No email reports will be sent.'}
"""
        
        # Save instructions
        instructions_file = os.path.join(output_dir, "INSTRUCTIONS.md")
        with open(instructions_file, 'w', encoding='utf-8') as f:
            f.write(instructions)
        
        return {
            "success": True,
            "task_name": task_name,
            "config_file": config_file,
            "script_file": script_file,
            "batch_file": batch_file,
            "instructions_file": instructions_file,
            "output_dir": output_dir,
            "task_command": task_command,
            "next_run": config["next_run"],
            "message": "Scheduled update system created. Follow the instructions to set up the scheduled task."
        }
    
    except Exception as e:
        logger.error(f"Error in scheduled update system: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

# Helper functions

async def get_recommended_parameters(
    analysis_type: str,
    optimization_target: Optional[str] = None,
    history_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get recommended parameters based on historical performance.
    
    Parameters:
    -----------
    analysis_type : str
        Type of analysis
    optimization_target : str, optional
        Metric to optimize
    history_id : str, optional
        ID for tracking analysis history
        
    Returns:
    --------
    Dict[str, Any]
        Recommended parameters
    """
    # Set default parameters based on analysis type
    if analysis_type == "eda":
        return {
            "analysis_types": ["descriptive", "correlation", "distribution"],
            "columns": None,  # Use all columns
            "categorical_analysis": True,
            "numerical_analysis": True
        }
    
    elif analysis_type == "modeling":
        return {
            "problem_type": "regression",  # Default, should be overridden
            "test_size": 0.2,
            "random_state": 42,
            "model_types": ["linear", "random_forest", "xgboost"],
            "cv_folds": 5,
            "feature_engineering": True
        }
    
    elif analysis_type == "decision_support":
        return {
            "question": "What insights can be derived from this data?",
            "data_sources": None,  # Will be set by the adaptive system
            "include_visualization": True,
            "report_format": "markdown"
        }
    
    # Default empty parameters
    return {}

def generate_dashboard_components(
    analysis_type: str,
    analysis_result: Dict[str, Any],
    optimization_result: Optional[Dict[str, Any]],
    visualization_config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Generate dashboard components based on analysis results.
    
    Parameters:
    -----------
    analysis_type : str
        Type of analysis
    analysis_result : Dict[str, Any]
        Analysis results
    optimization_result : Dict[str, Any], optional
        Optimization results
    visualization_config : Dict[str, Any]
        Visualization configuration
        
    Returns:
    --------
    List[Dict[str, Any]]
        Dashboard component configurations
    """
    components = []
    
    # Extract custom components from config if provided
    custom_components = visualization_config.get("components", [])
    if custom_components:
        return custom_components
    
    # Generate components based on analysis type
    if analysis_type == "eda":
        # Add summary statistics component
        components.append({
            "type": "table",
            "title": "Data Summary",
            "description": "Summary statistics for numerical variables",
            "parameters": {
                "columns": None  # Use all columns
            }
        })
        
        # Add correlation heatmap
        components.append({
            "type": "chart",
            "title": "Correlation Heatmap",
            "description": "Correlation matrix for numerical variables",
            "parameters": {
                "chart_type": "heatmap",
                "x_column": "index",
                "y_columns": ["value", "pivot"]
            }
        })
        
        # Add distribution charts for selected variables
        numerical_columns = visualization_config.get("numerical_columns", [])
        for i, column in enumerate(numerical_columns[:3]):
            components.append({
                "type": "chart",
                "title": f"Distribution of {column}",
                "description": f"Histogram showing the distribution of {column}",
                "parameters": {
                    "chart_type": "histogram",
                    "x_column": column,
                    "y_columns": [""]
                }
            })
    
    elif analysis_type == "modeling":
        # Add model performance component
        components.append({
            "type": "table",
            "title": "Model Performance",
            "description": "Performance metrics for different models",
            "parameters": {
                "columns": ["model_name", "accuracy", "f1_score", "precision", "recall", "r2_score", "mae", "rmse"]
            }
        })
        
        # Add feature importance chart
        components.append({
            "type": "chart",
            "title": "Feature Importance",
            "description": "Importance of features in the best model",
            "parameters": {
                "chart_type": "bar",
                "x_column": "feature",
                "y_columns": ["importance"]
            }
        })
        
        # Add predictions vs actual scatter plot
        components.append({
            "type": "chart",
            "title": "Predictions vs Actual",
            "description": "Scatter plot of predicted vs actual values",
            "parameters": {
                "chart_type": "scatter",
                "x_column": "actual",
                "y_columns": ["predicted", "trendline"]
            }
        })
    
    elif analysis_type == "decision_support":
        # Add insights component
        components.append({
            "type": "insight",
            "title": "Key Insights",
            "description": "Important insights derived from the data",
            "parameters": {
                "insight_type": "text",
                "text": "Key insights from the decision support analysis."
            }
        })
        
        # Add recommendations component
        components.append({
            "type": "insight",
            "title": "Recommendations",
            "description": "Recommended actions based on the analysis",
            "parameters": {
                "insight_type": "text",
                "text": "Recommended actions based on the decision support analysis."
            }
        })
    
    # Add optimization components if available
    if optimization_result and optimization_result.get("success", False):
        # Add optimization insights
        components.append({
            "type": "insight",
            "title": "Optimization Insights",
            "description": "Insights from the optimization process",
            "parameters": {
                "insight_type": "text",
                "text": "Insights from the optimization process."
            }
        })
        
        # Add performance trend chart
        components.append({
            "type": "chart",
            "title": "Performance Trend",
            "description": "Trend of performance metrics over time",
            "parameters": {
                "chart_type": "line",
                "x_column": "timestamp",
                "y_columns": ["metric_value"]
            }
        })
    
    return components

def generate_adaptive_analysis_report(
    analysis_type: str,
    analysis_result: Dict[str, Any],
    optimization_result: Optional[Dict[str, Any]],
    dashboard_result: Optional[Dict[str, Any]],
    performance_metrics: Dict[str, float],
    parameters: Dict[str, Any]
) -> str:
    """
    Generate a markdown report for adaptive analysis.
    
    Parameters:
    -----------
    analysis_type : str
        Type of analysis
    analysis_result : Dict[str, Any]
        Analysis results
    optimization_result : Dict[str, Any], optional
        Optimization results
    dashboard_result : Dict[str, Any], optional
        Dashboard results
    performance_metrics : Dict[str, float]
        Performance metrics
    parameters : Dict[str, Any]
        Analysis parameters
        
    Returns:
    --------
    str
        Markdown report
    """
    # Format timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Start report
    report = f"# Adaptive Analysis Report: {analysis_type.capitalize()}\n\n"
    report += f"**Report Date:** {timestamp}\n\n"
    
    # Add performance metrics
    report += "## Performance Metrics\n\n"
    report += "| Metric | Value |\n"
    report += "|--------|-------|\n"
    
    for metric_name, metric_value in performance_metrics.items():
        if isinstance(metric_value, float):
            report += f"| {metric_name} | {metric_value:.4f} |\n"
        else:
            report += f"| {metric_name} | {metric_value} |\n"
    
    report += "\n"
    
    # Add analysis parameters
    report += "## Analysis Parameters\n\n"
    report += "```json\n"
    report += json.dumps(parameters, indent=2)
    report += "\n```\n\n"
    
    # Add analysis results
    report += "## Analysis Results\n\n"
    
    if analysis_type == "eda":
        # Add EDA results
        if "insights" in analysis_result:
            report += "### Key Insights\n\n"
            
            for insight in analysis_result["insights"]:
                report += f"- {insight}\n"
            
            report += "\n"
        
        if "statistics" in analysis_result:
            report += "### Summary Statistics\n\n"
            report += "Statistics for numerical variables are available in the output directory.\n\n"
        
        if "visualizations" in analysis_result:
            report += "### Visualizations\n\n"
            report += "The following visualizations were generated:\n\n"
            
            for viz_name, viz_path in analysis_result["visualizations"].items():
                report += f"- {viz_name}: {os.path.basename(viz_path)}\n"
            
            report += "\n"
    
    elif analysis_type == "modeling":
        # Add modeling results
        best_model = analysis_result.get("best_model", {})
        model_key = best_model.get("model_key", "")
        
        report += f"### Best Model: {model_key}\n\n"
        
        performance = best_model.get("performance", {})
        if performance:
            report += "#### Performance Metrics\n\n"
            report += "| Metric | Value |\n"
            report += "|--------|-------|\n"
            
            for metric_name, metric_value in performance.items():
                report += f"| {metric_name} | {metric_value:.4f} |\n"
            
            report += "\n"
        
        feature_importance = best_model.get("feature_importance", {})
        if feature_importance:
            report += "#### Feature Importance\n\n"
            report += "| Feature | Importance |\n"
            report += "|---------|------------|\n"
            
            for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]:
                report += f"| {feature} | {importance:.4f} |\n"
            
            report += "\n"
    
    elif analysis_type == "decision_support":
        # Add decision support results
        if "insights" in analysis_result:
            report += "### Key Insights\n\n"
            
            for insight in analysis_result["insights"]:
                insight_type = insight.get("type", "").capitalize()
                description = insight.get("description", "")
                
                report += f"**{insight_type}:** {description}\n\n"
        
        if "recommendations" in analysis_result:
            report += "### Recommendations\n\n"
            
            for rec in analysis_result["recommendations"]:
                rec_type = rec.get("type", "").replace("_", " ").capitalize()
                description = rec.get("description", "")
                
                report += f"**{rec_type}:** {description}\n\n"
                
                steps = rec.get("steps", [])
                if steps:
                    for i, step in enumerate(steps, 1):
                        report += f"{i}. {step}\n"
                    
                    report += "\n"
    
    # Add optimization results
    if optimization_result and optimization_result.get("success", False):
        report += "## Optimization Results\n\n"
        
        # Add recommendations from optimization
        if "recommendations" in optimization_result:
            report += "### Optimization Recommendations\n\n"
            
            for rec in optimization_result["recommendations"]:
                rec_type = rec.get("type", "").replace("_", " ").capitalize()
                description = rec.get("description", "")
                
                report += f"**{rec_type}:** {description}\n\n"
                
                # Add additional details for different recommendation types
                if rec_type.lower() == "parameter tuning":
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
    
    # Add dashboard information
    if dashboard_result and dashboard_result.get("success", False):
        report += "## Interactive Dashboard\n\n"
        report += f"An interactive dashboard was created with {dashboard_result.get('component_count', 0)} components.\n\n"
        
        dashboard_file = dashboard_result.get("html_file")
        if dashboard_file:
            report += f"Dashboard location: `{dashboard_file}`\n\n"
    
    # Add next steps
    report += "## Next Steps\n\n"
    
    if optimization_result and optimization_result.get("success", False):
        # Get high priority recommendations
        has_recommendations = False
        
        if "recommendations" in optimization_result:
            high_priority_recs = [rec for rec in optimization_result["recommendations"] if rec.get("priority") == "high"]
            
            if high_priority_recs:
                has_recommendations = True
                report += "1. Implement high priority optimization recommendations:\n"
                
                for i, rec in enumerate(high_priority_recs[:3]):
                    report += f"   - {rec.get('description')}\n"
        
        if not has_recommendations:
            report += "1. Continue monitoring performance metrics\n"
    else:
        report += "1. Enable optimization to receive performance improvement recommendations\n"
    
    report += "2. Review analysis results and insights\n"
    
    if dashboard_result and dashboard_result.get("success", False):
        report += "3. Explore the interactive dashboard for deeper insights\n"
    else:
        report += "3. Consider creating an interactive dashboard for visualization\n"
    
    return report

def generate_dash_app_code(
    dashboard_title: str,
    layout: Dict[str, Any],
    components: List[Dict[str, Any]],
    theme: str
) -> str:
    """
    Generate Dash app code for web dashboard.
    
    Parameters:
    -----------
    dashboard_title : str
        Dashboard title
    layout : Dict[str, Any]
        Layout configuration
    components : List[Dict[str, Any]]
        Component configurations
    theme : str
        Dashboard theme
        
    Returns:
    --------
    str
        Dash app code
    """
    # Convert theme to Dash Bootstrap Components theme
    theme_map = {
        "light": "BOOTSTRAP",
        "dark": "DARKLY",
        "corporate": "FLATLY",
        "sunset": "JOURNAL"
    }
    
    dash_theme = theme_map.get(theme, "BOOTSTRAP")
    
    # Generate imports
    code = f"""import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import os

# Load data
df = pd.read_csv('dashboard_data.csv')

# Initialize Dash app
app = dash.Dash(__name__, title="{dashboard_title}")

# Define app layout
app.layout = html.Div([
    html.H1("{dashboard_title}", style={{"textAlign": "center", "marginBottom": "30px"}}),
    
    # Main content
    html.Div([
"""
    
    # Generate components
    for i, component in enumerate(components):
        component_type = component.get("type", "chart")
        component_title = component.get("title", f"Component {i+1}")
        component_description = component.get("description", "")
        component_params = component.get("parameters", {})
        
        # Generate component code
        code += f"""
        # Component {i+1}: {component_title}
        html.Div([
            html.H3("{component_title}"),
            html.P("{component_description}"),
"""
        
        if component_type == "chart":
            chart_type = component_params.get("chart_type", "line")
            x_column = component_params.get("x_column", "")
            y_columns = component_params.get("y_columns", [""])
            
            if isinstance(y_columns, str):
                y_columns = [y_columns]
            
            # Generate chart code based on type
            if chart_type == "line":
                code += f"""
            dcc.Graph(
                id='chart-{i+1}',
                figure=px.line(df, x='{x_column}', y={y_columns}, title="{component_title}")
            )
"""
            
            elif chart_type == "bar":
                code += f"""
            dcc.Graph(
                id='chart-{i+1}',
                figure=px.bar(df, x='{x_column}', y={y_columns}, title="{component_title}")
            )
"""
            
            elif chart_type == "scatter":
                code += f"""
            dcc.Graph(
                id='chart-{i+1}',
                figure=px.scatter(df, x='{x_column}', y='{y_columns[0]}', title="{component_title}")
            )
"""
            
            elif chart_type == "pie":
                code += f"""
            dcc.Graph(
                id='chart-{i+1}',
                figure=px.pie(df, names='{x_column}', values='{y_columns[0]}', title="{component_title}")
            )
"""
            
            elif chart_type == "histogram":
                code += f"""
            dcc.Graph(
                id='chart-{i+1}',
                figure=px.histogram(df, x='{x_column}', title="{component_title}")
            )
"""
            
            elif chart_type == "box":
                code += f"""
            dcc.Graph(
                id='chart-{i+1}',
                figure=px.box(df, x='{x_column}', y={y_columns}, title="{component_title}")
            )
"""
            
            elif chart_type == "heatmap":
                code += f"""
            dcc.Graph(
                id='chart-{i+1}',
                figure=px.imshow(df.corr(), title="{component_title}")
            )
"""
        
        elif component_type == "table":
            columns = component_params.get("columns", None)
            
            if columns:
                code += f"""
            dash_table.DataTable(
                id='table-{i+1}',
                columns=[{{"name": i, "id": i}} for i in {columns}],
                data=df[{columns}].to_dict('records'),
                style_table={{"overflowX": "auto"}},
                style_header={{
                    'backgroundColor': 'rgb(230, 230, 230)',
                    'fontWeight': 'bold'
                }},
                style_cell={{
                    'padding': '10px',
                    'textAlign': 'left'
                }}
            )
"""
            else:
                code += f"""
            dash_table.DataTable(
                id='table-{i+1}',
                columns=[{{"name": i, "id": i}} for i in df.columns],
                data=df.to_dict('records'),
                style_table={{"overflowX": "auto"}},
                style_header={{
                    'backgroundColor': 'rgb(230, 230, 230)',
                    'fontWeight': 'bold'
                }},
                style_cell={{
                    'padding': '10px',
                    'textAlign': 'left'
                }}
            )
"""
        
        elif component_type == "metric":
            metric_column = component_params.get("metric_column", "")
            metric_name = component_params.get("metric_name", metric_column)
            
            if metric_column and metric_column in df.columns:
                code += f"""
            html.Div([
                html.Div("{metric_name}", style={{"fontSize": "18px", "color": "#666"}}),
                html.Div(f"{{df['{metric_column}'].mean():.2f}}", style={{
                    "fontSize": "36px", 
                    "fontWeight": "bold",
                    "margin": "15px 0"
                }})
            ], style={{"textAlign": "center", "padding": "20px", "backgroundColor": "#f8f9fa", "borderRadius": "10px"}})
"""
        
        elif component_type == "insight":
            insight_text = component_params.get("text", "")
            
            code += f"""
            html.Div([
                html.P("{insight_text}"),
            ], style={{"padding": "15px", "backgroundColor": "#f8f9fa", "borderLeft": "4px solid #4361ee", "borderRadius": "4px"}})
"""
        
        # Close component div
        code += """
        ], style={"backgroundColor": "white", "padding": "20px", "borderRadius": "10px", "boxShadow": "0px 2px 10px rgba(0,0,0,0.1)", "marginBottom": "20px"}),
"""
    
    # Close layout divs
    code += """
    ], style={"maxWidth": "1200px", "margin": "0 auto", "padding": "20px"}),
], style={"fontFamily": "Arial, sans-serif", "backgroundColor": "#f5f7fa", "minHeight": "100vh"})

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
"""
    
    return code

def generate_update_script(config: Dict[str, Any]) -> str:
    """
    Generate update script for scheduled updates.
    
    Parameters:
    -----------
    config : Dict[str, Any]
        Update configuration
        
    Returns:
    --------
    str
        Python update script
    """
    script = """#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import datetime
import pandas as pd
import logging
import traceback
import asyncio
import importlib.util
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Setup logging
log_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"update_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("scheduled_update")

def load_config():
    """Load the update configuration."""
    config_path = os.path.join(os.path.dirname(__file__), "schedule_config.json")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load config: {str(e)}")
        return None

def update_config(config):
    """Update the configuration file."""
    config_path = os.path.join(os.path.dirname(__file__), "schedule_config.json")
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logger.error(f"Failed to update config: {str(e)}")
        return False

def calculate_next_run(schedule):
    """Calculate the next run time based on the schedule."""
    now = datetime.datetime.now()
    
    if schedule["frequency"] == "daily":
        # Parse time
        hour, minute = map(int, schedule["time"].split(":"))
        next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        
        # If the time today has passed, move to tomorrow
        if next_run <= now:
            next_run += datetime.timedelta(days=1)
    
    elif schedule["frequency"] == "weekly":
        # Parse time
        hour, minute = map(int, schedule["time"].split(":"))
        
        # Map days to integers (0 = Monday, 6 = Sunday)
        day_mapping = {
            "Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6
        }
        
        # Convert days to integers
        days = [day_mapping.get(day, 0) for day in schedule["days"]]
        
        # Sort days
        days.sort()
        
        # Find the next scheduled day
        current_weekday = now.weekday()
        next_day = None
        
        for day in days:
            if day > current_weekday:
                next_day = day
                break
        
        # If no day found, use the first day of next week
        if next_day is None:
            next_day = days[0]
            days_ahead = 7 - current_weekday + next_day
        else:
            days_ahead = next_day - current_weekday
        
        # Calculate next run date
        next_run = now + datetime.timedelta(days=days_ahead)
        next_run = next_run.replace(hour=hour, minute=minute, second=0, microsecond=0)
        
        # If today is a scheduled day and the time has not passed, use today
        if current_weekday in days and now.replace(hour=hour, minute=minute, second=0, microsecond=0) > now:
            next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    
    else:  # once
        # Parse time
        hour, minute = map(int, schedule["time"].split(":"))
        next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        
        # If the time today has passed, don't run again
        if next_run <= now:
            next_run = None
    
    return next_run.isoformat() if next_run else None

def import_module_from_file(file_path, module_name):
    """Import a module from a file path."""
    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        logger.error(f"Failed to import module: {str(e)}")
        return None

def send_email_report(report_email, subject, body):
    """Send an email report."""
    if not report_email:
        return False
    
    try:
        # This is a placeholder for email sending
        # In a real implementation, you would use an SMTP server
        logger.info(f"Would send email to {report_email} with subject: {subject}")
        logger.info(f"Email body: {body[:100]}...")
        
        # Uncomment to implement actual email sending
        """
        msg = MIMEMultipart()
        msg['From'] = 'your_email@example.com'
        msg['To'] = report_email
        msg['Subject'] = subject
        
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP('smtp.example.com', 587)
        server.starttls()
        server.login("your_email@example.com", "your_password")
        server.send_message(msg)
        server.quit()
        """
        
        return True
    except Exception as e:
        logger.error(f"Failed to send email: {str(e)}")
        return False

async def update_dashboard(dashboard_file, data_source):
    """Update a dashboard with new data."""
    if not dashboard_file or not os.path.exists(dashboard_file):
        logger.warning(f"Dashboard file not found: {dashboard_file}")
        return False
    
    try:
        # Import modules
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
        from analysis.tools.molecules.enhancement.interactive_visualization import update_dashboard
        
        # Read data
        from analysis.tools.atoms.data_reader import read_data
        data_result = await read_data(data_source)
        
        if not data_result.get("success", False):
            logger.error(f"Failed to read data: {data_result.get('error', 'Unknown error')}")
            return False
        
        df = data_result.get("data")
        
        # Update dashboard
        result = await update_dashboard(
            dashboard_file=dashboard_file,
            data_source=df,
            update_existing=True
        )
        
        if result.get("success", False):
            logger.info(f"Dashboard updated successfully: {dashboard_file}")
            return True
        else:
            logger.error(f"Failed to update dashboard: {result.get('error', 'Unknown error')}")
            return False
    except Exception as e:
        logger.error(f"Error updating dashboard: {str(e)}")
        logger.error(traceback.format_exc())
        return False

async def run_analysis(data_source, update_config):
    """Run the analysis based on the configuration."""
    try:
        # Import modules
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
        from analysis.tools.organisms.enhancement.adaptive_analysis_system import adaptive_analysis_system
        
        # Get analysis configuration
        analysis_type = update_config.get("analysis_type", "eda")
        parameters = update_config.get("parameters", {})
        
        # Run analysis
        result = await adaptive_analysis_system(
            data_source=data_source,
            analysis_type=analysis_type,
            parameters=parameters,
            output_dir=os.path.join(os.path.dirname(__file__), "output", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        )
        
        if result.get("success", False):
            logger.info(f"Analysis completed successfully")
            return result
        else:
            logger.error(f"Analysis failed: {result.get('error', 'Unknown error')}")
            return None
    except Exception as e:
        logger.error(f"Error in analysis: {str(e)}")
        logger.error(traceback.format_exc())
        return None

async def main():
    """Main update function."""
    # Load configuration
    config = load_config()
    if not config:
        logger.error("Failed to load configuration. Exiting.")
        return
    
    # Check if enabled
    if not config.get("enabled", True):
        logger.info("Updates are disabled. Exiting.")
        return
    
    # Get data source
    data_source = config.get("data_source")
    if not data_source:
        logger.error("No data source specified. Exiting.")
        return
    
    # Run analysis
    logger.info(f"Running analysis for data source: {data_source}")
    analysis_result = await run_analysis(data_source, config.get("update_config", {}))
    
    # Update dashboard if configured
    dashboard_file = config.get("dashboard_file")
    if dashboard_file and analysis_result:
        logger.info(f"Updating dashboard: {dashboard_file}")
        await update_dashboard(dashboard_file, data_source)
    
    # Send email report if configured
    report_email = config.get("report_email")
    if report_email and analysis_result:
        logger.info(f"Sending email report to: {report_email}")
        
        # Extract report content
        report_file = analysis_result.get("report_file")
        report_content = "No report content available."
        
        if report_file and os.path.exists(report_file):
            with open(report_file, 'r', encoding='utf-8') as f:
                report_content = f.read()
        
        # Send email
        send_email_report(
            report_email,
            f"Scheduled Analysis Report - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
            report_content
        )
    
    # Update last run and next run
    config["last_run"] = datetime.datetime.now().isoformat()
    config["next_run"] = calculate_next_run(config.get("schedule", {}))
    
    # Save updated configuration
    update_config(config)
    
    logger.info(f"Update completed successfully")

if __name__ == "__main__":
    asyncio.run(main())
"""
    
    return script

def calculate_next_run(schedule: Dict[str, Any]) -> str:
    """
    Calculate the next run time based on the schedule.
    
    Parameters:
    -----------
    schedule : Dict[str, Any]
        Schedule configuration
        
    Returns:
    --------
    str
        ISO format timestamp for the next run
    """
    now = datetime.datetime.now()
    
    if schedule["frequency"] == "daily":
        # Parse time
        hour, minute = map(int, schedule["time"].split(":"))
        next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        
        # If the time today has passed, move to tomorrow
        if next_run <= now:
            next_run += datetime.timedelta(days=1)
    
    elif schedule["frequency"] == "weekly":
        # Parse time
        hour, minute = map(int, schedule["time"].split(":"))
        
        # Map days to integers (0 = Monday, 6 = Sunday)
        day_mapping = {
            "Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6
        }
        
        # Convert days to integers
        days = [day_mapping.get(day, 0) for day in schedule["days"]]
        
        # Sort days
        days.sort()
        
        # Find the next scheduled day
        current_weekday = now.weekday()
        next_day = None
        
        for day in days:
            if day > current_weekday:
                next_day = day
                break
        
        # If no day found, use the first day of next week
        if next_day is None:
            next_day = days[0]
            days_ahead = 7 - current_weekday + next_day
        else:
            days_ahead = next_day - current_weekday
        
        # Calculate next run date
        next_run = now + datetime.timedelta(days=days_ahead)
        next_run = next_run.replace(hour=hour, minute=minute, second=0, microsecond=0)
    
    else:  # once
        # Parse time
        hour, minute = map(int, schedule["time"].split(":"))
        next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        
        # If the time today has passed, don't run again
        if next_run <= now:
            next_run = None
    
    return next_run.isoformat() if next_run else None
