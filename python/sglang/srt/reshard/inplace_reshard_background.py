"""Background prepare + fast commit for experimental in-place TP reshard.

While the server is still serving at ``old_tp``, joining standby ranks can load a
dummy model shell, receive weight shards, allocate KV, and warm up attention
backends.  The pause window should then only rebuild process groups and flip
active ranks (plus rank0 KV teardown / narrow, which needs the serving drain).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def background_prep_enabled() -> bool:
    return os.environ.get("SGLANG_INPLACE_RESHARD_BACKGROUND_PREP", "1") == "1"


@dataclass
class InplaceReshardPrepState:
    old_tp: int = 0
    new_tp: int = 0
    weights_ready: bool = False
    runtime_ready: bool = False
    joiners_prepared: bool = False
    rank0_exported_ipc: bool = False
    timings_s: Dict[str, float] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return self.weights_ready and self.runtime_ready

    def reset(self) -> None:
        self.old_tp = 0
        self.new_tp = 0
        self.weights_ready = False
        self.runtime_ready = False
        self.joiners_prepared = False
        self.rank0_exported_ipc = False
        self.timings_s.clear()


def prep_state_key(old_tp: int, new_tp: int) -> tuple[int, int]:
    return (int(old_tp), int(new_tp))
