"""Implementation of prioritized experience replay."""

from typing import TypeVar, Optional, List

import numpy as np  # type: ignore

from decuen.experience.experience import Experience
from decuen.experience.experience_manager import ExperienceManager


ActionType = TypeVar("ActionType")


class PrioritizedExperienceManager(ExperienceManager[ActionType]):
    """Deep reinforcement learning ExperienceManager object used to apply Q-learning updates
    """

    # the memory cache of past experiences and corresponding actions
    memory: List[Experience[ActionType]]
    # the probability distribution hyperparameter
    omega: float

    def __init__(self, sample_size: int, omega: float, memory_capacity: Optional[int] = None) -> None:
        """Initialize memory double-ended queue of size memory_capacity and sampling size sample_size
        """
        super().__init__(sample_size, memory_capacity)
        # initialize a heap
        self.memory = []
        self.omega = omega

    def experience(self, experience: Experience[ActionType]) -> None:
        """Perform an experience and add it to the memory.
        """
        if experience.priority == -1:
            experience.priority = 1 if not self.memory else max(expr.priority for expr in self.memory)
        self.memory.append(experience)

    def sample(self, size: Optional[int] = None) -> Optional[List[Experience[ActionType]]]:
        """Select a size amount of experiences from the memory.
        """
        sample_size = size if size else self.sample_size
        if len(self.memory) < sample_size:
            return None

        total_priority = sum((expr.priority ** self.omega) for expr in self.memory)
        prob = [(expr.priority ** self.omega) / total_priority for expr in self.memory]

        return list(np.random.choice(self.memory, size=sample_size, p=prob))
