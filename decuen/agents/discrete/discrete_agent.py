from typing import Tuple

import numpy as np  # type: ignore

from decuen.agents.agent import Agent
from decuen.policies.policy import Policy
from decuen.experience.experience_manager import ExperienceManager


ActionType = int


class DiscreteAgent(Agent[ActionType]):
    num_actions: int

    def __init__(self, state_shape: Tuple[int, ...], policy: Policy, experience_manager: ExperienceManager[ActionType],
                 num_actions: int) -> None:
        super().__init__(state_shape, policy, experience_manager)
        self.num_actions = num_actions

    def _act(self, state: np.ndarray) -> ActionType:
        raise NotImplementedError()

    def _learn(self) -> None:
        raise NotImplementedError()

    def _validate_action(self, action: ActionType) -> None:
        if action < 0 or action >= self.num_actions:  # type: ignore
            message = f"Got out of range [0, {self.num_actions - 1}] action index {action}."  # type: ignore
            raise ValueError(message)
