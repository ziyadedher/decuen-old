"""Deep Q-Learning agent (DQN) based on the original DQN paper by Mnih et al. [1].

[1] https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
"""
from typing import Tuple

import numpy as np  # type: ignore
import tensorflow as tf  # type: ignore
from tensorflow.keras import models  # type: ignore

from decuen.policies.policy import Policy
from decuen.experience.experience_manager import ExperienceManager
from decuen.agents.discrete.discrete_agent import DiscreteAgent


ActionType = int


class DQNAgent(DiscreteAgent):
    """Classical DQN agent for a discrete action space."""
    _steps_since_update: int
    model: models.Model
    discount_factor: float
    target_model: models.Model
    target_update_rate: int

    def __init__(self, state_shape: Tuple[int, ...], policy: Policy, experience_manager: ExperienceManager[ActionType],
                 num_actions: int, model: models.Model, discount_factor: float, target_update_rate: int = 1) -> None:
        super().__init__(state_shape, policy, experience_manager, num_actions)
        self._steps_since_update = 0
        self.model = model
        self.discount_factor = discount_factor
        self.target_model = models.clone_model(model)
        self.target_update_rate = target_update_rate

        self._validate_model(model)

    def _act(self, state: np.ndarray) -> ActionType:
        return self.policy.choose_action(self.model.predict_on_batch(np.array([state])), self.step)

    def _learn(self) -> None:
        self._steps_since_update += 1
        self._train(self.model, self.target_model)

        if self._steps_since_update >= self.target_update_rate:
            self._steps_since_update = 0
            self.target_model.set_weights(self.model.get_weights())

    def _train(self, model: models.Model, target_model: models.Model) -> None:
        experience_sample = self.experience_manager.sample()
        if experience_sample is None:
            return

        original_states = np.array([experience.original_state for experience in experience_sample])
        new_states = np.array([experience.new_state for experience in experience_sample])

        value_predictions = target_model.predict_on_batch(new_states)
        discounted_state_values = np.zeros((len(experience_sample), self.num_actions))
        for i, experience in enumerate(experience_sample):
            discounted_state_values[i][experience.action] = experience.reward + self.discount_factor * (
                np.max(value_predictions[i]) if not experience.done else 0
            )

        loss = model.train_on_batch(original_states, discounted_state_values)
        tf.summary.scalar("loss", loss, step=self.step)  # pylint: disable=E1101

    def _validate_model(self, model: models.Model) -> None:
        input_shape = model.get_layer(index=0).input_shape[1:]
        if input_shape != self.state_shape:
            message = f"Given model has incorrect input shape {input_shape} while agent expects {self.state_shape}."
            raise ValueError(message)

        output_shape = model.get_layer(index=-1).output_shape[1:]
        if len(output_shape) > 1:
            message = f"Given model has a multidimentional output shape {output_shape}."

        output_size = output_shape[0]
        if output_size != self.num_actions:
            message = f"Given model has incorrect output size {output_size} while agent expects {self.num_actions}."
            raise ValueError(message)
