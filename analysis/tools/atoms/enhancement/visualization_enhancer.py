"""
Visualization Enhancement Module

This module provides tools for creating interactive visualizations and dashboard components
to enhance the data analysis system's visualization capabilities.
"""

import os
import json
import datetime
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
import logging
import base64
from io import BytesIO

# For static visualization
import matplotlib.pyplot as plt
import seaborn as sns

# For interactive visualization
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from analysis.mcp_init import mcp

# Setup logging
logger = logging.getLogger(__name__)

@mcp.tool(
    name="create_interactive_chart",
    description="Create interactive chart using Plotly",
    tags=["visualization", "interactive", "enhancement"]
)
async def create_interactive_chart(
    data: Union[str, pd.DataFrame],
    chart_type: str,
    x_column: str,
    y_columns: Union[str, List[str]],
    title: str,
    color_column: Optional[str] = None,
    size_column: Optional[str] = None,
    facet_column: Optional[str] = None,
    hover_data: Optional[List[str]] = None,
    output_file: Optional[str] = None,
    height: int = 600,
    width: int = 900,
    template: str = "plotly_white"
) -> Dict[str, Any]:
    """
    Create an interactive chart using Plotly.
    
    Parameters:
    -----------
    data : Union[str, pd.DataFrame]
        Data source - either a DataFrame or path to a CSV file
    chart_type : str
        Type of chart: 'line', 'bar', 'scatter', 'pie', 'histogram', 'box', 'heatmap'
    x_column : str
        Column name for x-axis
    y_columns : Union[str, List[str]]
        Column name(s) for y-axis. Multiple columns for multi-series charts.
    title : str
        Chart title
    color_column : str, optional
        Column name for color encoding
    size_column : str, optional
        Column name for size encoding (scatter plots)
    facet_column : str, optional
        Column name for faceting/small multiples
    hover_data : List[str], optional
        Additional columns to show in hover tooltip
    output_file : str, optional
        Path to save the output HTML file
    height : int, optional
        Chart height in pixels
    width : int, optional
        Chart width in pixels
    template : str, optional
        Plotly template name
        
    Returns:
    --------
    Dict[str, Any]
        Result including success status and paths to output files
    """
    if not PLOTLY_AVAILABLE:
        return {
            "success": False,
            "error": "Plotly is not available. Install with 'pip install plotly'."
        }
    
    try:
        # Load data if it's a file path
        if isinstance(data, str):
            # Check file extension
            if data.endswith('.csv'):
                df = pd.read_csv(data)
            elif data.endswith('.xlsx') or data.endswith('.xls'):
                df = pd.read_excel(data)
            elif data.endswith('.json'):
                df = pd.read_json(data)
            else:
                return {
                    "success": False,
                    "error": f"Unsupported file format: {data}"
                }
        else:
            df = data.copy()
        
        # Convert y_columns to list if it's a string
        if isinstance(y_columns, str):
            y_columns = [y_columns]
        
        # Set default output file if not provided
        if output_file is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join("C:\\Users\\packr\\OneDrive\\palantir", "output", "viz", "interactive")
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"{chart_type}_{timestamp}.html")
        
        # Create chart based on type
        fig = None
        
        if chart_type == 'line':
            if len(y_columns) == 1:
                fig = px.line(
                    df, x=x_column, y=y_columns[0], 
                    color=color_column, facet_col=facet_column,
                    title=title, height=height, width=width, 
                    template=template, hover_data=hover_data
                )
            else:
                # Multiple lines
                fig = go.Figure()
                for y_col in y_columns:
                    fig.add_trace(go.Scatter(
                        x=df[x_column], y=df[y_col], mode='lines+markers', name=y_col
                    ))
                fig.update_layout(
                    title=title, height=height, width=width, 
                    template=template, xaxis_title=x_column, yaxis_title="Value"
                )
        
        elif chart_type == 'bar':
            if len(y_columns) == 1:
                fig = px.bar(
                    df, x=x_column, y=y_columns[0],
                    color=color_column, facet_col=facet_column,
                    title=title, height=height, width=width,
                    template=template, hover_data=hover_data
                )
            else:
                # Grouped bar chart
                fig = go.Figure()
                for y_col in y_columns:
                    fig.add_trace(go.Bar(
                        x=df[x_column], y=df[y_col], name=y_col
                    ))
                fig.update_layout(
                    title=title, height=height, width=width,
                    template=template, barmode='group',
                    xaxis_title=x_column, yaxis_title="Value"
                )
        
        elif chart_type == 'scatter':
            fig = px.scatter(
                df, x=x_column, y=y_columns[0],
                color=color_column, size=size_column,
                facet_col=facet_column, title=title,
                height=height, width=width,
                template=template, hover_data=hover_data
            )
            
            # Add trendline if requested
            if len(y_columns) > 1 and y_columns[1] == 'trendline':
                fig.update_traces(mode='markers')
                fig.update_layout(title=f"{title} with Trendline")
                # Add trendline
                z = np.polyfit(df[x_column], df[y_columns[0]], 1)
                p = np.poly1d(z)
                fig.add_trace(go.Scatter(
                    x=df[x_column], y=p(df[x_column]),
                    mode='lines', name='Trendline',
                    line=dict(color='red', dash='dash')
                ))
        
        elif chart_type == 'pie':
            fig = px.pie(
                df, names=x_column, values=y_columns[0],
                color=color_column, title=title,
                height=height, width=width,
                template=template, hover_data=hover_data
            )
        
        elif chart_type == 'histogram':
            fig = px.histogram(
                df, x=x_column, color=color_column,
                facet_col=facet_column, title=title,
                height=height, width=width,
                template=template, hover_data=hover_data
            )
            
            if len(y_columns) > 0 and y_columns[0] != '':
                # Use y for weights if provided
                fig = px.histogram(
                    df, x=x_column, y=y_columns[0],
                    color=color_column, facet_col=facet_column,
                    title=title, height=height, width=width,
                    template=template, hover_data=hover_data
                )
        
        elif chart_type == 'box':
            if len(y_columns) == 1:
                fig = px.box(
                    df, x=x_column, y=y_columns[0],
                    color=color_column, facet_col=facet_column,
                    title=title, height=height, width=width,
                    template=template, hover_data=hover_data
                )
            else:
                # Multiple box plots
                fig = go.Figure()
                for y_col in y_columns:
                    fig.add_trace(go.Box(
                        y=df[y_col], name=y_col,
                        boxmean=True  # adds mean to box plot
                    ))
                fig.update_layout(
                    title=title, height=height, width=width,
                    template=template
                )
        
        elif chart_type == 'heatmap':
            # For heatmap, we assume data is already in the right format
            # or we need to pivot it
            if len(y_columns) > 1 and y_columns[1] == 'pivot':
                # Need to pivot the data
                pivot_values = y_columns[0]
                pivot_df = df.pivot(index=x_column, columns=color_column, values=pivot_values)
                fig = px.imshow(
                    pivot_df, title=title,
                    height=height, width=width,
                    template=template
                )
            else:
                # Assume data is already in matrix format where x_column is the index
                heatmap_df = df.set_index(x_column)
                fig = px.imshow(
                    heatmap_df, title=title,
                    height=height, width=width,
                    template=template
                )
        
        else:
            return {
                "success": False,
                "error": f"Unsupported chart type: {chart_type}"
            }
        
        # Save to HTML file
        fig.write_html(output_file)
        
        # Generate a preview image (PNG)
        img_path = output_file.replace('.html', '.png')
        fig.write_image(img_path)
        
        # Create a data URL for preview image
        with open(img_path, "rb") as img_file:
            img_data = base64.b64encode(img_file.read()).decode('utf-8')
            img_data_url = f"data:image/png;base64,{img_data}"
        
        return {
            "success": True,
            "chart_type": chart_type,
            "output_file": output_file,
            "preview_image": img_path,
            "data_url": img_data_url,
            "chart_title": title,
            "dimensions": {
                "height": height,
                "width": width
            }
        }
    
    except Exception as e:
        logger.error(f"Error creating interactive chart: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

@mcp.tool(
    name="generate_dashboard_component",
    description="Generate a dashboard component for analytics",
    tags=["visualization", "dashboard", "enhancement"]
)
async def generate_dashboard_component(
    component_type: str,
    data: Union[str, pd.DataFrame],
    parameters: Dict[str, Any],
    output_file: Optional[str] = None,
    title: Optional[str] = None,
    description: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate a dashboard component for analytics dashboards.
    
    Parameters:
    -----------
    component_type : str
        Type of component ('chart', 'metric', 'table', 'insight', 'filter')
    data : Union[str, pd.DataFrame]
        Data source - either a DataFrame or path to data file
    parameters : Dict[str, Any]
        Component-specific parameters
    output_file : str, optional
        Path to save the output component file
    title : str, optional
        Component title
    description : str, optional
        Component description
        
    Returns:
    --------
    Dict[str, Any]
        Result including success status and component information
    """
    try:
        # Load data if it's a file path
        if isinstance(data, str):
            # Check file extension
            if data.endswith('.csv'):
                df = pd.read_csv(data)
            elif data.endswith('.xlsx') or data.endswith('.xls'):
                df = pd.read_excel(data)
            elif data.endswith('.json'):
                df = pd.read_json(data)
            else:
                return {
                    "success": False,
                    "error": f"Unsupported file format: {data}"
                }
        else:
            df = data.copy()
        
        # Set default output file if not provided
        if output_file is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join("C:\\Users\\packr\\OneDrive\\palantir", "output", "dashboard", "components")
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"{component_type}_{timestamp}.json")
        
        # Process based on component type
        result = {
            "success": True,
            "component_type": component_type,
            "title": title,
            "description": description,
            "created_at": datetime.datetime.now().isoformat(),
            "output_file": output_file
        }
        
        if component_type == 'chart':
            # Call create_interactive_chart with parameters
            chart_params = parameters.copy()
            
            # Add data to parameters
            chart_result = await create_interactive_chart(
                data=df,
                chart_type=chart_params.get("chart_type", "line"),
                x_column=chart_params.get("x_column", ""),
                y_columns=chart_params.get("y_columns", ""),
                title=title or chart_params.get("title", "Dashboard Chart"),
                color_column=chart_params.get("color_column"),
                size_column=chart_params.get("size_column"),
                facet_column=chart_params.get("facet_column"),
                hover_data=chart_params.get("hover_data"),
                height=chart_params.get("height", 400),
                width=chart_params.get("width", 600),
                template=chart_params.get("template", "plotly_white"),
                output_file=chart_params.get("output_file")
            )
            
            result["chart"] = chart_result
            result["preview_image"] = chart_result.get("preview_image")
            result["data_url"] = chart_result.get("data_url")
        
        elif component_type == 'metric':
            # Generate metric card
            metric_name = parameters.get("metric_name", "Value")
            metric_column = parameters.get("metric_column", "")
            
            if metric_column and metric_column in df.columns:
                metric_value = df[metric_column].mean()
                format_type = parameters.get("format", "number")
                
                if format_type == "currency":
                    formatted_value = f"${metric_value:,.2f}"
                elif format_type == "percent":
                    formatted_value = f"{metric_value:.2%}"
                else:
                    formatted_value = f"{metric_value:,.2f}"
                
                result["metric"] = {
                    "name": metric_name,
                    "value": metric_value,
                    "formatted_value": formatted_value,
                    "format": format_type
                }
                
                # Calculate comparison if requested
                if "comparison_column" in parameters:
                    comparison_column = parameters["comparison_column"]
                    if comparison_column in df.columns:
                        comparison_value = df[comparison_column].mean()
                        change = metric_value - comparison_value
                        percent_change = (change / comparison_value) * 100 if comparison_value != 0 else 0
                        
                        result["metric"]["comparison"] = {
                            "value": comparison_value,
                            "change": change,
                            "percent_change": percent_change,
                            "direction": "up" if change > 0 else "down" if change < 0 else "unchanged"
                        }
            else:
                result["metric"] = {
                    "name": metric_name,
                    "value": parameters.get("value", 0),
                    "formatted_value": parameters.get("formatted_value", "0")
                }
        
        elif component_type == 'table':
            # Generate table component
            columns = parameters.get("columns", df.columns.tolist())
            sort_by = parameters.get("sort_by")
            ascending = parameters.get("ascending", True)
            page_size = parameters.get("page_size", 10)
            
            # Filter columns
            table_df = df[columns].copy()
            
            # Apply sorting if specified
            if sort_by and sort_by in table_df.columns:
                table_df = table_df.sort_values(by=sort_by, ascending=ascending)
            
            # Get only the requested number of rows
            table_df = table_df.head(page_size)
            
            # Convert to records for JSON serialization
            table_data = table_df.to_dict(orient='records')
            
            result["table"] = {
                "columns": columns,
                "data": table_data,
                "row_count": len(table_df),
                "total_rows": len(df),
                "has_more": len(df) > page_size
            }
        
        elif component_type == 'insight':
            # Generate insight component
            insight_type = parameters.get("insight_type", "statistic")
            insight_text = parameters.get("text", "")
            
            if insight_type == "statistic" and "column" in parameters:
                column = parameters["column"]
                if column in df.columns:
                    statistic = parameters.get("statistic", "mean")
                    
                    if statistic == "mean":
                        value = df[column].mean()
                        statistic_text = f"Average {column}"
                    elif statistic == "median":
                        value = df[column].median()
                        statistic_text = f"Median {column}"
                    elif statistic == "sum":
                        value = df[column].sum()
                        statistic_text = f"Total {column}"
                    elif statistic == "min":
                        value = df[column].min()
                        statistic_text = f"Minimum {column}"
                    elif statistic == "max":
                        value = df[column].max()
                        statistic_text = f"Maximum {column}"
                    else:
                        value = df[column].mean()
                        statistic_text = f"Average {column}"
                    
                    result["insight"] = {
                        "type": "statistic",
                        "title": title or statistic_text,
                        "description": description or f"{statistic_text}: {value:,.2f}",
                        "value": value
                    }
                else:
                    result["insight"] = {
                        "type": "text",
                        "title": title or "Insight",
                        "description": description or insight_text
                    }
            else:
                result["insight"] = {
                    "type": "text",
                    "title": title or "Insight",
                    "description": description or insight_text
                }
        
        elif component_type == 'filter':
            # Generate filter component
            filter_column = parameters.get("column", "")
            filter_type = parameters.get("filter_type", "select")
            
            if filter_column and filter_column in df.columns:
                # Get unique values for select/multi-select filters
                if filter_type in ["select", "multi-select"]:
                    unique_values = df[filter_column].unique().tolist()
                    
                    result["filter"] = {
                        "column": filter_column,
                        "type": filter_type,
                        "options": unique_values,
                        "default": parameters.get("default")
                    }
                elif filter_type == "range":
                    min_value = df[filter_column].min()
                    max_value = df[filter_column].max()
                    
                    result["filter"] = {
                        "column": filter_column,
                        "type": filter_type,
                        "min": min_value,
                        "max": max_value,
                        "default_min": parameters.get("default_min", min_value),
                        "default_max": parameters.get("default_max", max_value)
                    }
                elif filter_type == "date":
                    min_date = df[filter_column].min()
                    max_date = df[filter_column].max()
                    
                    result["filter"] = {
                        "column": filter_column,
                        "type": filter_type,
                        "min_date": min_date,
                        "max_date": max_date,
                        "default_start": parameters.get("default_start", min_date),
                        "default_end": parameters.get("default_end", max_date)
                    }
                else:
                    result["filter"] = {
                        "column": filter_column,
                        "type": "text",
                        "default": parameters.get("default", "")
                    }
            else:
                result["filter"] = {
                    "column": filter_column,
                    "type": filter_type,
                    "default": parameters.get("default", "")
                }
        
        else:
            return {
                "success": False,
                "error": f"Unsupported component type: {component_type}"
            }
        
        # Save component specification to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        return result
    
    except Exception as e:
        logger.error(f"Error generating dashboard component: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

@mcp.tool(
    name="export_visualization",
    description="Export visualization in various formats",
    tags=["visualization", "export", "enhancement"]
)
async def export_visualization(
    chart_file: str,
    export_formats: List[str],
    output_dir: Optional[str] = None,
    dpi: int = 300,
    width: Optional[int] = None,
    height: Optional[int] = None
) -> Dict[str, Any]:
    """
    Export a visualization in various formats.
    
    Parameters:
    -----------
    chart_file : str
        Path to the chart file (HTML or JSON)
    export_formats : List[str]
        List of formats to export (png, jpg, pdf, svg, html)
    output_dir : str, optional
        Directory to save the exported files
    dpi : int, optional
        Resolution for raster formats (default: 300)
    width : int, optional
        Width of the exported visualization in pixels
    height : int, optional
        Height of the exported visualization in pixels
        
    Returns:
    --------
    Dict[str, Any]
        Result including success status and paths to exported files
    """
    try:
        if not PLOTLY_AVAILABLE:
            return {
                "success": False,
                "error": "Plotly is not available. Install with 'pip install plotly'."
            }
        
        # Validate input file
        if not os.path.exists(chart_file):
            return {
                "success": False,
                "error": f"Chart file not found: {chart_file}"
            }
        
        # Set default output directory if not provided
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(chart_file), "exports")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load the chart
        if chart_file.endswith('.html'):
            # For HTML files, we'll need to re-create the chart
            # This is a simplification as extracting a figure from HTML is complex
            from plotly.io import read_html
            try:
                fig = read_html(chart_file, output_type='figure')[0]
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Failed to extract figure from HTML: {str(e)}"
                }
        
        elif chart_file.endswith('.json'):
            # Load figure from JSON
            with open(chart_file, 'r', encoding='utf-8') as f:
                chart_data = json.load(f)
            
            if "chart" in chart_data and "data_url" in chart_data["chart"]:
                # This is a dashboard component JSON
                # We need to re-create the chart from parameters
                # This is a simplified approach
                return {
                    "success": False,
                    "error": "Exporting from dashboard component JSON is not fully supported yet."
                }
            else:
                # Try to load as a Plotly figure
                try:
                    fig = go.Figure(chart_data)
                except Exception:
                    return {
                        "success": False,
                        "error": "Failed to load figure from JSON"
                    }
        else:
            return {
                "success": False,
                "error": f"Unsupported chart file format: {chart_file}"
            }
        
        # Base filename (without extension)
        base_name = os.path.splitext(os.path.basename(chart_file))[0]
        
        # Export in requested formats
        exported_files = {}
        
        # Update figure dimensions if specified
        if width or height:
            fig.update_layout(
                width=width or fig.layout.width or 800,
                height=height or fig.layout.height or 600
            )
        
        for fmt in export_formats:
            fmt = fmt.lower()
            output_file = os.path.join(output_dir, f"{base_name}.{fmt}")
            
            if fmt == 'png':
                fig.write_image(output_file, scale=dpi/100)
                exported_files['png'] = output_file
            
            elif fmt == 'jpg' or fmt == 'jpeg':
                fig.write_image(output_file, scale=dpi/100)
                exported_files['jpg'] = output_file
            
            elif fmt == 'svg':
                fig.write_image(output_file)
                exported_files['svg'] = output_file
            
            elif fmt == 'pdf':
                fig.write_image(output_file)
                exported_files['pdf'] = output_file
            
            elif fmt == 'html':
                fig.write_html(output_file)
                exported_files['html'] = output_file
            
            elif fmt == 'json':
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(fig.to_dict(), f, ensure_ascii=False, indent=2)
                exported_files['json'] = output_file
            
            else:
                logger.warning(f"Unsupported export format: {fmt}")
        
        return {
            "success": True,
            "exported_files": exported_files,
            "output_dir": output_dir,
            "dpi": dpi,
            "dimensions": {
                "width": fig.layout.width,
                "height": fig.layout.height
            }
        }
    
    except Exception as e:
        logger.error(f"Error exporting visualization: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }
