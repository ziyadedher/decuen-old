"""Common interface for all agents."""
from typing import Generic, TypeVar, Tuple

import numpy as np  # type: ignore

from decuen.policies.policy import Policy
from decuen.experience.experience import Experience
from decuen.experience.experience_manager import ExperienceManager


ActionType = TypeVar("ActionType")


class Agent(Generic[ActionType]):
    step: int
    state_shape: Tuple[int, ...]
    policy: Policy
    experience_manager: ExperienceManager[ActionType]

    def __init__(self, state_shape: Tuple[int, ...],
                 policy: Policy, experience_manager: ExperienceManager[ActionType]) -> None:
        self.step = 0
        self.state_shape = state_shape
        self.policy = policy
        self.experience_manager = experience_manager

    def wait_step(self) -> None:
        self.step += 1

    def act(self, state: np.ndarray) -> ActionType:
        self._validate_state(state)
        action = self._act(state)
        self._validate_action(action)
        self.wait_step()
        return action

    def experience(self, experience: Experience[ActionType]) -> None:
        self._validate_state(experience.original_state)
        self._validate_action(experience.action)
        self._validate_state(experience.new_state)
        self.experience_manager.experience(experience)
        self._learn()

    def _act(self, state: np.ndarray) -> ActionType:
        raise NotImplementedError()

    def _learn(self) -> None:
        raise NotImplementedError()

    def _validate_state(self, state: np.ndarray) -> None:
        if state.shape != self.state_shape:  # type: ignore
            message = f"Got state shape {state.shape} while agent expects {self.state_shape}."  # type: ignore
            raise ValueError(message)

    def _validate_action(self, action: ActionType) -> None:
        raise NotImplementedError()
