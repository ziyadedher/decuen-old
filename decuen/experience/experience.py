from typing import Generic, TypeVar

import numpy as np  # type: ignore


ActionType = TypeVar("ActionType")


class Experience(Generic[ActionType]):
    """
    Experience object to store an Agent's experience at each time step
    """

    # the current state
    original_state: np.ndarray
    # corresponding new action
    action: ActionType
    # outcome of performing action
    new_state: np.ndarray
    # the Q-value gained
    reward: float
    # flag for if there are no more iterations to experience
    done: bool

    def __init__(self, original_state: np.ndarray, action: ActionType,
                 new_state: np.ndarray, reward: float, done: bool) -> None:
        """
        Initialize a new experience associated with a current state, new state, action, and reward.
        """
        self.original_state = original_state
        self.action = action
        self.new_state = new_state
        self.reward = reward
        self.done = done
