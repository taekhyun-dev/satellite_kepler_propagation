# simserver/core/hooks.py
from dataclasses import dataclass
from typing import Optional, Callable, Tuple

@dataclass
class Hooks:
    upload_model_to_server: Optional[Callable[[int, str], None]] = None
    get_global_model_snapshot: Optional[Callable[[], Tuple[int, dict]]] = None