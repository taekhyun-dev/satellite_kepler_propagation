# simserver/core/hooks.py
from dataclasses import dataclass
from typing import Optional, Callable, Tuple, Dict, Any

@dataclass
class Hooks:
    upload_model_to_server: Optional[Callable[[int, str], None]] = None
    get_global_model_snapshot: Optional[Callable[[], tuple[int, dict]]] = None