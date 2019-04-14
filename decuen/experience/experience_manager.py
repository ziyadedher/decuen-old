from typing import Generic, TypeVar, Optional, Collection

from decuen.experience.experience import Experience


ActionType = TypeVar("ActionType")


class ExperienceManager(Generic[ActionType]):
    memory: Collection[Experience[ActionType]]
    memory_capacity: Optional[int]
    sample_size: int

    def __init__(self, sample_size: int, memory_capacity: Optional[int] = None) -> None:
        self.memory_capacity = memory_capacity
        self.sample_size = sample_size

    def experience(self, experience: Experience[ActionType]) -> None:
        raise NotImplementedError()

    def sample(self) -> Optional[Collection[Experience[ActionType]]]:
        raise NotImplementedError()
