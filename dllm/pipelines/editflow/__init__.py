from . import trainer, utils
from .models.dream.modelling_dream import (
    EditFlowDreamConfig,
    EditFlowDreamModel,
)
from .models.llada.modelling_llada import (
    EditFlowLLaDAConfig,
    EditFlowLLaDAModel,
)

# Lazy import ModernBert to avoid flash_attn dependency when not needed
# ModernBert requires flash_attn which may not be available
def _import_modernbert():
    from .models.bert.modelling_modernbert import (
        EditFlowModernBertConfig,
        EditFlowModernBertModel,
    )
    return EditFlowModernBertConfig, EditFlowModernBertModel

# Only import if explicitly needed
try:
    from .models.bert.modelling_modernbert import (
        EditFlowModernBertConfig,
        EditFlowModernBertModel,
    )
except ImportError:
    EditFlowModernBertConfig = None
    EditFlowModernBertModel = None

from dllm.pipelines.editflow.trainer import EditFlowTrainer
