"""
Interactive Visualization Workflow Module

This module implements workflows for creating interactive visualizations and dashboards
to enhance the data analysis system's visualization capabilities.
"""

import os
import json
import datetime
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
import logging

from analysis.mcp_init import mcp
from analysis.tools.atoms.enhancement.visualization_enhancer import (
    create_interactive_chart,
    generate_dashboard_component,
    export_visualization
)
from analysis.tools.atoms.data_reader import read_data

# Setup logging
logger = logging.getLogger(__name__)

@mcp.workflow(
    name="create_interactive_dashboard",
    description="Create a complete interactive dashboard with multiple visualizations"
)
async def create_interactive_dashboard(
    data_source: Union[str, pd.DataFrame],
    dashboard_title: str,
    components: List[Dict[str, Any]],
    output_dir: Optional[str] = None,
    theme: str = "light",
    layout: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create a complete interactive dashboard with multiple visualizations.
    
    Parameters:
    -----------
    data_source : Union[str, pd.DataFrame]
        Data source - either a DataFrame or path to a data file
    dashboard_title : str
        Dashboard title
    components : List[Dict[str, Any]]
        List of dashboard component configurations
    output_dir : str, optional
        Directory to save dashboard files
    theme : str, optional
        Dashboard theme ('light', 'dark', 'corporate', etc.)
    layout : Dict[str, Any], optional
        Dashboard layout configuration
        
    Returns:
    --------
    Dict[str, Any]
        Result including success status and dashboard information
    """
    try:
        # Load data if it's a file path
        if isinstance(data_source, str):
            data_result = await read_data(data_source)
            
            if not data_result.get("success", False):
                return {
                    "success": False,
                    "error": f"Failed to read data source: {data_result.get('error', 'Unknown error')}"
                }
            
            df = data_result.get("data")
        else:
            df = data_source.copy()
        
        # Set default output directory if not provided
        if output_dir is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join("C:\\Users\\packr\\OneDrive\\palantir", "output", "dashboard", timestamp)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Dashboard configuration
        dashboard_config = {
            "title": dashboard_title,
            "created_at": datetime.datetime.now().isoformat(),
            "theme": theme,
            "layout": layout or {
                "type": "grid",
                "columns": 2,
                "gap": 16
            },
            "components": []
        }
        
        # Create each component
        component_results = []
        
        for i, comp_config in enumerate(components):
            component_type = comp_config.get("type", "chart")
            component_title = comp_config.get("title", f"Component {i+1}")
            
            # Set component output directory
            component_dir = os.path.join(output_dir, f"component_{i+1}")
            os.makedirs(component_dir, exist_ok=True)
            
            # Generate component
            component_result = await generate_dashboard_component(
                component_type=component_type,
                data=df,
                parameters=comp_config.get("parameters", {}),
                output_file=os.path.join(component_dir, f"{component_type}.json"),
                title=component_title,
                description=comp_config.get("description")
            )
            
            if component_result.get("success", False):
                # Add component to dashboard configuration
                dashboard_config["components"].append({
                    "id": f"component_{i+1}",
                    "type": component_type,
                    "title": component_title,
                    "description": comp_config.get("description"),
                    "data": component_result
                })
                
                component_results.append(component_result)
            else:
                logger.warning(f"Failed to create component {i+1}: {component_result.get('error', 'Unknown error')}")
        
        # Save dashboard configuration
        dashboard_file = os.path.join(output_dir, "dashboard_config.json")
        with open(dashboard_file, 'w', encoding='utf-8') as f:
            json.dump(dashboard_config, f, ensure_ascii=False, indent=2)
        
        # Generate the dashboard HTML
        html_content = generate_dashboard_html(dashboard_config)
        
        # Save the dashboard HTML
        html_file = os.path.join(output_dir, "dashboard.html")
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return {
            "success": True,
            "dashboard_title": dashboard_title,
            "output_dir": output_dir,
            "dashboard_file": dashboard_file,
            "html_file": html_file,
            "component_count": len(component_results),
            "components": component_results
        }
    
    except Exception as e:
        logger.error(f"Error creating interactive dashboard: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

@mcp.workflow(
    name="update_dashboard",
    description="Update an existing dashboard with new data or components"
)
async def update_dashboard(
    dashboard_file: str,
    data_source: Union[str, pd.DataFrame],
    update_existing: bool = True,
    new_components: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Update an existing dashboard with new data or add new components.
    
    Parameters:
    -----------
    dashboard_file : str
        Path to the dashboard configuration file
    data_source : Union[str, pd.DataFrame]
        New data source - either a DataFrame or path to a data file
    update_existing : bool, optional
        Whether to update existing components with new data
    new_components : List[Dict[str, Any]], optional
        List of new component configurations to add
        
    Returns:
    --------
    Dict[str, Any]
        Result including success status and updated dashboard information
    """
    try:
        # Check if dashboard file exists
        if not os.path.exists(dashboard_file):
            return {
                "success": False,
                "error": f"Dashboard file not found: {dashboard_file}"
            }
        
        # Load dashboard configuration
        with open(dashboard_file, 'r', encoding='utf-8') as f:
            dashboard_config = json.load(f)
        
        # Get output directory
        output_dir = os.path.dirname(dashboard_file)
        
        # Load data if it's a file path
        if isinstance(data_source, str):
            data_result = await read_data(data_source)
            
            if not data_result.get("success", False):
                return {
                    "success": False,
                    "error": f"Failed to read data source: {data_result.get('error', 'Unknown error')}"
                }
            
            df = data_result.get("data")
        else:
            df = data_source.copy()
        
        # Track changes
        changes = {
            "updated_components": [],
            "new_components": [],
            "failed_updates": []
        }
        
        # Update existing components
        if update_existing:
            for i, component in enumerate(dashboard_config.get("components", [])):
                component_id = component.get("id")
                component_type = component.get("type")
                component_title = component.get("title")
                
                # Get component data
                component_data = component.get("data", {})
                
                # Set component output directory
                component_dir = os.path.join(output_dir, component_id)
                os.makedirs(component_dir, exist_ok=True)
                
                # Update component with new data
                try:
                    # Extract original parameters
                    if component_type == "chart":
                        chart_data = component_data.get("chart", {})
                        
                        # Re-create chart with new data
                        chart_result = await create_interactive_chart(
                            data=df,
                            chart_type=chart_data.get("chart_type", "line"),
                            x_column=chart_data.get("x_column", ""),
                            y_columns=chart_data.get("y_columns", ""),
                            title=component_title,
                            color_column=chart_data.get("color_column"),
                            size_column=chart_data.get("size_column"),
                            facet_column=chart_data.get("facet_column"),
                            hover_data=chart_data.get("hover_data"),
                            output_file=os.path.join(component_dir, f"chart_updated.html"),
                            height=chart_data.get("dimensions", {}).get("height", 400),
                            width=chart_data.get("dimensions", {}).get("width", 600)
                        )
                        
                        if chart_result.get("success", False):
                            # Update component data
                            component["data"]["chart"] = chart_result
                            component["data"]["preview_image"] = chart_result.get("preview_image")
                            component["data"]["data_url"] = chart_result.get("data_url")
                            
                            changes["updated_components"].append({
                                "component_id": component_id,
                                "component_type": component_type,
                                "status": "updated"
                            })
                        else:
                            changes["failed_updates"].append({
                                "component_id": component_id,
                                "component_type": component_type,
                                "error": chart_result.get("error", "Unknown error")
                            })
                    else:
                        # For other component types, regenerate with new data
                        parameters = component_data.get("parameters", {})
                        
                        component_result = await generate_dashboard_component(
                            component_type=component_type,
                            data=df,
                            parameters=parameters,
                            output_file=os.path.join(component_dir, f"{component_type}_updated.json"),
                            title=component_title,
                            description=component.get("description")
                        )
                        
                        if component_result.get("success", False):
                            # Update component data
                            component["data"] = component_result
                            
                            changes["updated_components"].append({
                                "component_id": component_id,
                                "component_type": component_type,
                                "status": "updated"
                            })
                        else:
                            changes["failed_updates"].append({
                                "component_id": component_id,
                                "component_type": component_type,
                                "error": component_result.get("error", "Unknown error")
                            })
                
                except Exception as e:
                    logger.error(f"Error updating component {component_id}: {str(e)}")
                    changes["failed_updates"].append({
                        "component_id": component_id,
                        "component_type": component_type,
                        "error": str(e)
                    })
        
        # Add new components
        if new_components:
            for i, comp_config in enumerate(new_components):
                component_type = comp_config.get("type", "chart")
                component_title = comp_config.get("title", f"New Component {i+1}")
                
                # Generate component ID
                component_id = f"component_{len(dashboard_config.get('components', [])) + i + 1}"
                
                # Set component output directory
                component_dir = os.path.join(output_dir, component_id)
                os.makedirs(component_dir, exist_ok=True)
                
                # Generate component
                component_result = await generate_dashboard_component(
                    component_type=component_type,
                    data=df,
                    parameters=comp_config.get("parameters", {}),
                    output_file=os.path.join(component_dir, f"{component_type}.json"),
                    title=component_title,
                    description=comp_config.get("description")
                )
                
                if component_result.get("success", False):
                    # Add component to dashboard configuration
                    dashboard_config["components"].append({
                        "id": component_id,
                        "type": component_type,
                        "title": component_title,
                        "description": comp_config.get("description"),
                        "data": component_result
                    })
                    
                    changes["new_components"].append({
                        "component_id": component_id,
                        "component_type": component_type,
                        "status": "added"
                    })
                else:
                    logger.warning(f"Failed to create new component: {component_result.get('error', 'Unknown error')}")
        
        # Update dashboard metadata
        dashboard_config["updated_at"] = datetime.datetime.now().isoformat()
        
        # Save updated dashboard configuration
        with open(dashboard_file, 'w', encoding='utf-8') as f:
            json.dump(dashboard_config, f, ensure_ascii=False, indent=2)
        
        # Generate updated dashboard HTML
        html_content = generate_dashboard_html(dashboard_config)
        
        # Save the updated dashboard HTML
        html_file = os.path.join(output_dir, "dashboard.html")
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return {
            "success": True,
            "dashboard_file": dashboard_file,
            "html_file": html_file,
            "changes": changes,
            "component_count": len(dashboard_config.get("components", [])),
            "update_timestamp": dashboard_config.get("updated_at")
        }
    
    except Exception as e:
        logger.error(f"Error updating dashboard: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

@mcp.workflow(
    name="export_dashboard",
    description="Export dashboard components in various formats"
)
async def export_dashboard(
    dashboard_file: str,
    export_formats: List[str],
    components_to_export: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    include_full_dashboard: bool = True
) -> Dict[str, Any]:
    """
    Export dashboard components in various formats.
    
    Parameters:
    -----------
    dashboard_file : str
        Path to the dashboard configuration file
    export_formats : List[str]
        List of formats to export (png, jpg, pdf, svg, html)
    components_to_export : List[str], optional
        List of component IDs to export (default: all)
    output_dir : str, optional
        Directory to save exported files
    include_full_dashboard : bool, optional
        Whether to export the full dashboard as a single file
        
    Returns:
    --------
    Dict[str, Any]
        Result including success status and paths to exported files
    """
    try:
        # Check if dashboard file exists
        if not os.path.exists(dashboard_file):
            return {
                "success": False,
                "error": f"Dashboard file not found: {dashboard_file}"
            }
        
        # Load dashboard configuration
        with open(dashboard_file, 'r', encoding='utf-8') as f:
            dashboard_config = json.load(f)
        
        # Set default output directory if not provided
        dashboard_dir = os.path.dirname(dashboard_file)
        
        if output_dir is None:
            output_dir = os.path.join(dashboard_dir, "exports")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Track exported files
        exported_files = {
            "components": {},
            "dashboard": {}
        }
        
        # Export individual components
        components = dashboard_config.get("components", [])
        
        for component in components:
            component_id = component.get("id")
            
            # Skip if not in components_to_export
            if components_to_export and component_id not in components_to_export:
                continue
            
            component_type = component.get("type")
            component_data = component.get("data", {})
            
            if component_type == "chart":
                chart_data = component_data.get("chart", {})
                
                if "output_file" in chart_data:
                    chart_file = chart_data.get("output_file")
                    
                    if os.path.exists(chart_file):
                        # Export chart
                        export_result = await export_visualization(
                            chart_file=chart_file,
                            export_formats=export_formats,
                            output_dir=os.path.join(output_dir, component_id),
                            dpi=300
                        )
                        
                        if export_result.get("success", False):
                            exported_files["components"][component_id] = export_result.get("exported_files", {})
            
            # Special handling for other component types could be added here
        
        # Export full dashboard if requested
        if include_full_dashboard:
            dashboard_html = os.path.join(dashboard_dir, "dashboard.html")
            
            if os.path.exists(dashboard_html):
                # Export using different methods based on format
                dashboard_exported = {}
                
                for fmt in export_formats:
                    fmt = fmt.lower()
                    output_file = os.path.join(output_dir, f"dashboard.{fmt}")
                    
                    if fmt == "html":
                        # Copy HTML file
                        import shutil
                        shutil.copy2(dashboard_html, output_file)
                        dashboard_exported["html"] = output_file
                    elif fmt in ["png", "jpg", "jpeg", "pdf"]:
                        # Use screenshot capability if available
                        # This would require a browser automation tool like Playwright
                        # For now, just indicate that this requires additional tools
                        dashboard_exported[fmt] = f"ERROR: {fmt.upper()} export of full dashboard requires browser automation"
                
                exported_files["dashboard"] = dashboard_exported
        
        return {
            "success": True,
            "dashboard_title": dashboard_config.get("title"),
            "output_dir": output_dir,
            "export_formats": export_formats,
            "exported_files": exported_files
        }
    
    except Exception as e:
        logger.error(f"Error exporting dashboard: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

# Utility functions for HTML generation

def generate_dashboard_html(dashboard_config: Dict[str, Any]) -> str:
    """
    Generate HTML for an interactive dashboard.
    
    Parameters:
    -----------
    dashboard_config : Dict[str, Any]
        Dashboard configuration
        
    Returns:
    --------
    str
        HTML content for the dashboard
    """
    title = dashboard_config.get("title", "Interactive Dashboard")
    theme = dashboard_config.get("theme", "light")
    components = dashboard_config.get("components", [])
    
    # Set theme CSS variables
    theme_vars = get_theme_variables(theme)
    
    # Generate HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        :root {{
            {theme_vars}
        }}
        
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: var(--background);
            color: var(--text);
            line-height: 1.6;
            padding: 20px;
        }}
        
        .dashboard {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        
        .dashboard-header {{
            margin-bottom: 30px;
            text-align: center;
        }}
        
        .dashboard-title {{
            font-size: 28px;
            margin-bottom: 10px;
        }}
        
        .dashboard-subtitle {{
            font-size: 16px;
            color: var(--text-secondary);
        }}
        
        .dashboard-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
        }}
        
        .component {{
            background-color: var(--component-bg);
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            padding: 20px;
            transition: transform 0.2s;
        }}
        
        .component:hover {{
            transform: translateY(-5px);
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.15);
        }}
        
        .component-header {{
            margin-bottom: 15px;
        }}
        
        .component-title {{
            font-size: 18px;
            margin-bottom: 5px;
        }}
        
        .component-description {{
            font-size: 14px;
            color: var(--text-secondary);
            margin-bottom: 15px;
        }}
        
        .component-content {{
            width: 100%;
            min-height: 300px;
        }}
        
        .chart-container {{
            width: 100%;
            height: 100%;
        }}
        
        .chart-iframe {{
            width: 100%;
            height: 400px;
            border: none;
        }}
        
        .metric-value {{
            font-size: 36px;
            font-weight: bold;
            color: var(--primary);
            margin: 15px 0;
        }}
        
        .metric-comparison {{
            font-size: 14px;
            margin-top: 5px;
        }}
        
        .comparison-up {{
            color: var(--success);
        }}
        
        .comparison-down {{
            color: var(--danger);
        }}
        
        .table-container {{
            width: 100%;
            overflow-x: auto;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        
        th, td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}
        
        th {{
            background-color: var(--header-bg);
            color: var(--text);
        }}
        
        tr:nth-child(even) {{
            background-color: var(--row-alt-bg);
        }}
        
        .insight-box {{
            padding: 15px;
            background-color: var(--insight-bg);
            border-left: 4px solid var(--primary);
            margin-bottom: 15px;
        }}
    </style>
</head>
<body>
    <div class="dashboard">
        <div class="dashboard-header">
            <h1 class="dashboard-title">{title}</h1>
            <p class="dashboard-subtitle">Created: {dashboard_config.get("created_at", "")}</p>
        </div>
        
        <div class="dashboard-grid">
"""
    
    # Add components
    for component in components:
        component_id = component.get("id")
        component_type = component.get("type")
        component_title = component.get("title", "")
        component_description = component.get("description", "")
        component_data = component.get("data", {})
        
        html += f"""
            <div class="component" id="{component_id}">
                <div class="component-header">
                    <h2 class="component-title">{component_title}</h2>
                    <p class="component-description">{component_description}</p>
                </div>
                <div class="component-content">
"""
        
        # Add component content based on type
        if component_type == "chart":
            chart_data = component_data.get("chart", {})
            
            if "output_file" in chart_data:
                chart_file = chart_data.get("output_file")
                # Use relative path
                iframe_src = os.path.relpath(chart_file, os.path.dirname(os.path.dirname(chart_file)))
                
                html += f"""
                    <div class="chart-container">
                        <iframe class="chart-iframe" src="{iframe_src}" frameborder="0"></iframe>
                    </div>
"""
            elif "data_url" in chart_data:
                # Use image data URL
                html += f"""
                    <div class="chart-container">
                        <img src="{chart_data.get('data_url')}" alt="{component_title}" style="width: 100%;">
                    </div>
"""
        
        elif component_type == "metric":
            metric = component_data.get("metric", {})
            metric_name = metric.get("name", "")
            formatted_value = metric.get("formatted_value", "0")
            
            html += f"""
                    <div class="metric-content">
                        <div class="metric-name">{metric_name}</div>
                        <div class="metric-value">{formatted_value}</div>
"""
            
            # Add comparison if available
            if "comparison" in metric:
                comparison = metric.get("comparison", {})
                percent_change = comparison.get("percent_change", 0)
                direction = comparison.get("direction", "unchanged")
                
                if direction == "up":
                    html += f"""
                        <div class="metric-comparison comparison-up">
                            ↑ {percent_change:.2f}%
                        </div>
"""
                elif direction == "down":
                    html += f"""
                        <div class="metric-comparison comparison-down">
                            ↓ {percent_change:.2f}%
                        </div>
"""
            
            html += """
                    </div>
"""
        
        elif component_type == "table":
            table = component_data.get("table", {})
            columns = table.get("columns", [])
            data = table.get("data", [])
            
            html += """
                    <div class="table-container">
                        <table>
                            <thead>
                                <tr>
"""
            
            # Table headers
            for column in columns:
                html += f"""
                                    <th>{column}</th>
"""
            
            html += """
                                </tr>
                            </thead>
                            <tbody>
"""
            
            # Table rows
            for row in data:
                html += """
                                <tr>
"""
                
                for column in columns:
                    cell_value = row.get(column, "")
                    html += f"""
                                    <td>{cell_value}</td>
"""
                
                html += """
                                </tr>
"""
            
            html += """
                            </tbody>
                        </table>
                    </div>
"""
        
        elif component_type == "insight":
            insight = component_data.get("insight", {})
            insight_title = insight.get("title", "")
            insight_description = insight.get("description", "")
            
            html += f"""
                    <div class="insight-box">
                        <h3>{insight_title}</h3>
                        <p>{insight_description}</p>
                    </div>
"""
        
        # Close component content and component div
        html += """
                </div>
            </div>
"""
    
    # Close dashboard grid and dashboard div
    html += """
        </div>
    </div>
</body>
</html>
"""
    
    return html

def get_theme_variables(theme: str) -> str:
    """
    Get CSS variables for a dashboard theme.
    
    Parameters:
    -----------
    theme : str
        Theme name
        
    Returns:
    --------
    str
        CSS variables for the theme
    """
    # Theme definitions
    themes = {
        "light": """
            --background: #f5f7fa;
            --component-bg: #ffffff;
            --text: #333333;
            --text-secondary: #6c757d;
            --primary: #4361ee;
            --success: #2ecc71;
            --danger: #e74c3c;
            --border: #dee2e6;
            --header-bg: #f8f9fa;
            --row-alt-bg: #f8f9fa;
            --insight-bg: #f0f7ff;
        """,
        "dark": """
            --background: #1a1a2e;
            --component-bg: #16213e;
            --text: #e6e6e6;
            --text-secondary: #a0a0a0;
            --primary: #4361ee;
            --success: #2ecc71;
            --danger: #e74c3c;
            --border: #3a3a5a;
            --header-bg: #0f3460;
            --row-alt-bg: #1f1f3a;
            --insight-bg: #0f3460;
        """,
        "corporate": """
            --background: #f2f7fa;
            --component-bg: #ffffff;
            --text: #2c3e50;
            --text-secondary: #7f8c8d;
            --primary: #3498db;
            --success: #27ae60;
            --danger: #c0392b;
            --border: #ecf0f1;
            --header-bg: #f8fafc;
            --row-alt-bg: #f8fafc;
            --insight-bg: #edf2f7;
        """,
        "sunset": """
            --background: #ffe8d6;
            --component-bg: #ffffff;
            --text: #6d4c41;
            --text-secondary: #8d6e63;
            --primary: #ff7043;
            --success: #66bb6a;
            --danger: #f44336;
            --border: #ffe0b2;
            --header-bg: #fff3e0;
            --row-alt-bg: #fff8e1;
            --insight-bg: #ffecb3;
        """
    }
    
    # Return theme variables or default to light theme
    return themes.get(theme, themes["light"])
