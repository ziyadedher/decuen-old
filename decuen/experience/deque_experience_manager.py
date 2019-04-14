from typing import TypeVar, Optional, List, Deque

import random
from collections import deque

from decuen.experience.experience import Experience
from decuen.experience.experience_manager import ExperienceManager


ActionType = TypeVar("ActionType")


class DequeExperienceManager(ExperienceManager[ActionType]):
    memory: Deque[Experience[ActionType]]

    def __init__(self, sample_size: int, memory_capacity: Optional[int] = None) -> None:
        super().__init__(sample_size, memory_capacity)
        self.memory = deque(maxlen=memory_capacity)

    def experience(self, experience: Experience[ActionType]) -> None:
        self.memory.append(experience)

    def sample(self) -> Optional[List[Experience[ActionType]]]:
        if len(self.memory) < self.sample_size:
            return None
        return random.sample(self.memory, self.sample_size)
