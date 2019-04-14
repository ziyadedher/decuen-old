from typing import Generic, TypeVar

import numpy as np  # type: ignore


ActionType = TypeVar("ActionType")


class Experience(Generic[ActionType]):
    original_state: np.ndarray
    action: ActionType
    new_state: np.ndarray
    reward: float
    done: bool

    def __init__(self, original_state: np.ndarray, action: ActionType,
                 new_state: np.ndarray, reward: float, done: bool) -> None:
        self.original_state = original_state
        self.action = action
        self.new_state = new_state
        self.reward = reward
        self.done = done
