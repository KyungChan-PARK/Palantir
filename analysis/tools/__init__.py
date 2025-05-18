# __init__.py
"""
데이터 분석 도구 패키지
"""

# Import submodules lazily to avoid importing heavy dependencies
try:
    from . import atoms
except Exception:  # Optional dependencies may be missing
    atoms = None
try:
    from . import molecules
except Exception:  # Optional dependencies may be missing
    molecules = None
try:
    from . import organisms
except Exception:  # Optional dependencies may be missing
    organisms = None

# 버전 정보
__version__ = '0.1.1'

# All importable modules
__all__ = ['atoms', 'molecules', 'organisms']
