# __init__.py
"""
원자 레벨 데이터 분석 도구
"""

try:
    from analysis.tools.atoms.data_reader import read_data
    from analysis.tools.atoms.data_processor import preprocess_data
    from analysis.tools.atoms.data_analyzer import analyze_data
except Exception:  # Optional dependencies may be missing
    read_data = None
    preprocess_data = None
    analyze_data = None

# Import codebase_knowledge lazily to avoid heavy imports
try:
    from . import codebase_knowledge
except Exception:  # pragma: no cover - optional
    codebase_knowledge = None

# 버전 정보
__version__ = '0.1.1'

# 사용 가능한 도구 목록
__all__ = [
    'read_data', 
    'preprocess_data', 
    'analyze_data',
    'codebase_knowledge'
]
