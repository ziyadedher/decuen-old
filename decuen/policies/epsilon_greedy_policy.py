from typing import Optional
from enum import Enum

import numpy as np  # type: ignore
import tensorflow as tf  # type: ignore

from decuen.policies.policy import Policy


class EpsilonGreedyPolicy(Policy):
    class AnnealingMethod(Enum):
        LINEAR = 0
        EXPONENTIAL = 1

    _annealing_method: AnnealingMethod
    _annealing_constant: float
    num_actions: int
    max_epsilon: float
    min_epsilon: float
    epsilon: float

    def __init__(self, num_actions: int, max_epsilon: float, min_epsilon: float,
                 annealing_method: AnnealingMethod, annealing_constant: float) -> None:
        self._annealing_method = annealing_method
        self._annealing_constant = annealing_constant
        self.num_actions = num_actions
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon = max_epsilon

    def choose_action(self, actions: np.ndarray, current_step: Optional[int] = None) -> int:
        action = np.argmax(actions) if np.random.rand() <= self.epsilon else np.random.randint(0, self.num_actions)
        self._anneal_epsilon(current_step)
        return action

    def _anneal_epsilon(self, current_step: Optional[int] = None) -> None:
        if self._annealing_method == EpsilonGreedyPolicy.AnnealingMethod.LINEAR:
            self.epsilon = max(self.min_epsilon, self.epsilon - self._annealing_constant)
        elif self._annealing_method == EpsilonGreedyPolicy.AnnealingMethod.EXPONENTIAL:
            self.epsilon = max(self.min_epsilon, self.epsilon * (1 - self._annealing_constant))
        else:
            raise ValueError(f"No such annealing method {self._anneal_epsilon} registered.")

        if current_step is not None:
            tf.summary.scalar("epsilon", self.epsilon, step=current_step)  # pylint: disable=E1101
