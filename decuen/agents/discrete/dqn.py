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

    # fixed number of steps per update
    _steps_since_update: int
    # the online (discounted) model
    model: models.Model
    # gamma, exploration vs exploitation rate
    discount_factor: float
    # the dynamic target model updated each iteration
    target_model: models.Model
    # rate of adjusting the parameters of target model
    target_update_rate: int

    def __init__(self, state_shape: Tuple[int, ...], policy: Policy, experience_manager: ExperienceManager[ActionType],
                 num_actions: int, model: models.Model, discount_factor: float, target_update_rate: int = 1) -> None:
        """
        Initialize online model, discount factor, target model, 
        ate of updates in time-steps and validates model
        """
        super().__init__(state_shape, policy, experience_manager, num_actions)
        self._steps_since_update = 0
        self.model = model
        self.discount_factor = discount_factor
        self.target_model = models.clone_model(model)
        self.target_update_rate = target_update_rate

        self._validate_model(model)

    def _act(self, state: np.ndarray) -> ActionType:
        """
        Select an action based on predictions on current state
        """
        return self.policy.choose_action(self.model.predict_on_batch(np.array([state])), self.step)

    def _learn(self) -> None:
        """
        Updates the steps update counter as well as the weights of
        the target network upon current state analysis
        """
        self._steps_since_update += 1
        self._train(self.model, self.target_model)

        if self._steps_since_update >= self.target_update_rate:
            self._steps_since_update = 0
            self.target_model.set_weights(self.model.get_weights())

    def _train(self, model: models.Model, target_model: models.Model) -> None:
        """
        Samples an experience from the EM's states and generates the discounted state values
        for what will be trained against iteratively per experience, new state, and action
        """
        experience_sample = self.experience_manager.sample()
        if experience_sample is None:
            return

        original_states = np.array([
            experience.original_state
            for experience in experience_sample
        ])
        new_states = np.array([
            experience.new_state
            for experience in experience_sample
        ])

        value_predictions = target_model.predict_on_batch(new_states)
        # generating the training target
        discounted_state_values = np.zeros((len(experience_sample), self.num_actions))
        for experience, value, pred in zip(experience_sample, discounted_state_values, value_predictions):
            value[experience.action] = experience.reward + self.discount_factor * (
                np.max(pred) if not experience.done else 0
            )

        loss = model.train_on_batch(original_states, discounted_state_values)
        tf.summary.scalar("loss", loss, step=self.step)  # pylint: disable=E1101

    def _validate_model(self, model: models.Model) -> None:
        """
        Ensures that the input and output shapes and sizes are in line with the agent's expectations
        """
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
