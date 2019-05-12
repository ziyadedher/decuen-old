"""Experience manager for reinforcement learning agent."""

from typing import Generic, TypeVar, Optional, Collection

from decuen.experience.experience import Experience


ActionType = TypeVar("ActionType")


class ExperienceManager(Generic[ActionType]):
    """
    Wrapper class for processing Experiences and updates.
    """

    # cache of sample experiences
    memory: Collection[Experience[ActionType]]
    # replay capacity
    memory_capacity: Optional[int]
    # minibatch amount to sample from cache
    sample_size: int

    def __init__(self, sample_size: int, memory_capacity: Optional[int] = None) -> None:
        """
        Set memory replay capacity and minibatch sample
        """
        self.memory_capacity = memory_capacity
        self.sample_size = sample_size

    def experience(self, experience: Experience[ActionType]) -> None:
        """
        Initiate an experience for a new action
        """
        raise NotImplementedError()

    def sample(self, size: Optional[int] = None) -> Optional[Collection[Experience[ActionType]]]:
        """
        Return a subset of memory
        """
        raise NotImplementedError()
