"""Greedy policy implementation."""

from typing import Optional

import numpy as np  # type: ignore

from decuen.policies.policy import Policy


class GreedyPolicy(Policy):
    """Implementation of greedy policy for DQN and online network exploration
    """

    def choose_action(self, actions: np.ndarray, current_step: Optional[int] = None) -> int:
        """Return the optimal action based on maximal reward value garnered
        """
        action: int = np.argmax(actions)
        return action
