from typing import Optional

import numpy as np  # type: ignore


class Policy:
    def choose_action(self, actions: np.ndarray, current_step: Optional[int] = None) -> int:
        raise NotImplementedError()
