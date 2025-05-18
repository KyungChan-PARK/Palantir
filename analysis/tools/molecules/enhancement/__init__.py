"""
Enhancement Molecules Initialization

This module initializes the enhancement molecular workflows for the MCP workflow improvement project.
These molecules include feedback loops, interactive visualization, and dashboard generation workflows.
"""

# Import all enhancement molecular workflows for easy access
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

# Export all workflows as module-level symbols
__all__ = [
    'self_improving_workflow',
    'process_optimization_workflow',
    'parameter_optimization_workflow',
    'create_interactive_dashboard',
    'update_dashboard',
    'export_dashboard'
]
