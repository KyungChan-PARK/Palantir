# __init__.py
"""
분자 레벨 데이터 분석 워크플로우
"""

try:
    from analysis.tools.molecules.exploratory_analysis import exploratory_analysis
    from analysis.tools.molecules.predictive_modeling import build_predictive_model
except Exception:  # Optional dependencies may be missing
    exploratory_analysis = None
    build_predictive_model = None

# Import codebase_knowledge lazily
try:
    from . import codebase_knowledge
except Exception:  # pragma: no cover - optional
    codebase_knowledge = None

# 버전 정보
__version__ = '0.1.1'

# 사용 가능한 워크플로우 목록
__all__ = [
    'exploratory_analysis', 
    'build_predictive_model',
    'codebase_knowledge'
]
