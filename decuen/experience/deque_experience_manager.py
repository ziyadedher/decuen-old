"""Implementation of a manager of experiences."""

from typing import TypeVar, Optional, List, Deque

import random
from collections import deque

from decuen.experience.experience import Experience
from decuen.experience.experience_manager import ExperienceManager


ActionType = TypeVar("ActionType")


class DequeExperienceManager(ExperienceManager[ActionType]):
    """Deep reinforcement learning ExperienceManager object used to apply Q-learning updates
    """

    # the memory cache of past experiences and corresponding actions
    memory: Deque[Experience[ActionType]]

    def __init__(self, sample_size: int, memory_capacity: Optional[int] = None) -> None:
        """Initialize memory double-ended queue of size memory_capacity and sampling size sample_size
        """
        super().__init__(sample_size, memory_capacity)
        self.memory = deque(maxlen=memory_capacity)

    def experience(self, experience: Experience[ActionType]) -> None:
        """Perform an experience and add it to the memory.
        """
        self.memory.append(experience)

    def sample(self, size: Optional[int] = None) -> Optional[List[Experience[ActionType]]]:
        """Select a size amount of experiences from the memory.
        """
        sample_size = size if size else self.sample_size
        if len(self.memory) < sample_size:
            return None
        return random.sample(self.memory, sample_size)
