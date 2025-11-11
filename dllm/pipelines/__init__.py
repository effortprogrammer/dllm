# Import pipelines conditionally to avoid unnecessary dependencies
from . import llada, dream, rnd

# editflow requires flash_attn due to ModernBert dependency
# Import it only when needed to avoid import errors
try:
    from . import editflow
except ImportError:
    pass  # editflow not available (flash_attn not installed)
