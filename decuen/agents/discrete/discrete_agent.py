"""Agent operating on discrete action space."""
from typing import Tuple

import numpy as np  # type: ignore

from decuen.agents import Agent
from decuen.policies.policy import Policy
from decuen.experience.experience_manager import ExperienceManager


ActionType = int


class DiscreteAgent(Agent[ActionType]):
    """
    Discrete DQN agent operating on a discrete action space.
    """

    # the number of actions in action space
    num_actions: int

    def __init__(self, state_shape: Tuple[int, ...], policy: Policy, experience_manager: ExperienceManager[ActionType],
                 num_actions: int) -> None:
        """
        Construct state setup, policy, action number, and set experience manager
        """
        super().__init__(state_shape, policy, experience_manager)
        self.num_actions = num_actions

    def _act(self, state: np.ndarray) -> ActionType:
        raise NotImplementedError()

    def _learn(self) -> None:
        raise NotImplementedError()

    def _validate_action(self, action: ActionType) -> None:
        """
        Ensures that actions are chosen within range
        """
        if action < 0 or action >= self.num_actions:
            message = f"Got out of range [0, {self.num_actions - 1}] action index {action}."
            raise ValueError(message)
