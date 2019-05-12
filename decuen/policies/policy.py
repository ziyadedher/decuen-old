"""Abstract policy for action."""
from typing import Optional

import numpy as np  # type: ignore


class Policy:
    """
    A policy for which a DQN Agent experiences new Experiences
    """

    def choose_action(self, actions: np.ndarray, current_step: Optional[int] = None) -> int:
        """
        Selects an action from a fixed set of possible moves
        """
        raise NotImplementedError()
