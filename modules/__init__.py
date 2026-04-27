# Re-export original modules from modules.py
import sys
import importlib.util

# Import from the root-level modules.py file
_spec = importlib.util.spec_from_file_location("base_modules", "modules.py")
_base_modules = importlib.util.module_from_spec(_spec)
sys.modules["base_modules"] = _base_modules
_spec.loader.exec_module(_base_modules)

Embedding = _base_modules.Embedding
MultiHeadAttention = _base_modules.MultiHeadAttention
FeedForward = _base_modules.FeedForward
LayerNorm = _base_modules.LayerNorm
PositionalEncoding = _base_modules.PositionalEncoding

# Export MoE modules
from modules.moe_modules import (
    ScenarioVectorBuilder,
    AccountPrototype,
    ExpertNetwork,
    MoEPriceTower
)
