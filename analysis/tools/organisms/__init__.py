# __init__.py
"""
유기체 레벨 의사결정 지원 시스템
"""

try:
    from analysis.tools.organisms.decision_support_system import decision_support
except Exception:  # Optional dependencies may be missing
    decision_support = None

# Import codebase_knowledge lazily
try:
    from . import codebase_knowledge
except Exception:  # pragma: no cover - optional
    codebase_knowledge = None

# 버전 정보
__version__ = '0.1.1'

# 사용 가능한 시스템 목록
__all__ = [
    'decision_support',
    'codebase_knowledge'
]
