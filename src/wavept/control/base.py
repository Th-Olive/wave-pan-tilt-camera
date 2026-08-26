"""Controller protocol.

Controllers see only the observation dict (and nominal parameters at
construction) — never the simulator, its true parameters, or env info.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Controller(Protocol):
    name: str

    def act(self, obs: dict) -> np.ndarray:  # -> (2,) action [pan, tilt]
        ...

    def reset(self) -> None:
        ...
