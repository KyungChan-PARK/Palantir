"""
Enhancement Organisms Initialization

This module initializes the enhancement organism systems for the MCP workflow improvement project.
These organisms include adaptive analysis systems, web dashboards, and scheduled update systems.
"""

# Import all enhancement organism systems for easy access
from analysis.tools.organisms.enhancement.adaptive_analysis_system import (
    adaptive_analysis_system,
    web_dashboard_system,
    scheduled_update_system
)

# Export all systems as module-level symbols
__all__ = [
    'adaptive_analysis_system',
    'web_dashboard_system',
    'scheduled_update_system'
]
