from typing import Optional

import numpy as np  # type: ignore

from decuen.policies.policy import Policy


class GreedyPolicy(Policy):
    def choose_action(self, actions: np.ndarray, current_step: Optional[int] = None) -> int:
        action: int = np.argmax(actions)
        return action
