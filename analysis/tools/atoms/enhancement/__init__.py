"""
Enhancement Atoms Initialization

This module initializes the enhancement atoms for the MCP workflow improvement project.
These atoms include feedback collection, performance analysis, and visualization enhancement tools.
"""

# Import all enhancement atoms for easy access
from analysis.tools.atoms.enhancement.feedback_collector import (
    collect_model_feedback,
    analyze_performance_trends,
    identify_improvement_areas
)

from analysis.tools.atoms.enhancement.visualization_enhancer import (
    create_interactive_chart,
    generate_dashboard_component,
    export_visualization
)

# Export all tools as module-level symbols
__all__ = [
    'collect_model_feedback',
    'analyze_performance_trends',
    'identify_improvement_areas',
    'create_interactive_chart',
    'generate_dashboard_component',
    'export_visualization'
]
