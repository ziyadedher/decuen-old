"""Deep Q-Learning agent (DQN) based on the original DQN paper by Mnih et al. [1].

[1] https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
"""
import numpy as np  # type: ignore
import tensorflow as tf  # type: ignore
from tensorflow.keras import models  # type: ignore

from decuen.agents.discrete.dqn import DQNAgent


ActionType = int


class DDQNAgent(DQNAgent):
    """Double DQN Agent implemenation."""

    def _learn(self) -> None:
        """
        Updates the steps update counter as well as the weights of
        the target network upon current state analysis
        """
        self._steps_since_update += 1
        self._train()
        if self._steps_since_update >= self.target_update_rate:
            self._steps_since_update = 0
            target_weights = self.target_model.get_weights()
            self.target_model.set_weights(self.model.get_weights())
            self.model.set_weights(target_weights)

    def _train(self) -> None:
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

        value_predictions = self.target_model.predict_on_batch(new_states)
        model_predictions = self.model.predict_on_batch(new_states)

        # generating the training target
        discounted_state_values = np.zeros((len(experience_sample), self.num_actions))
        itera = zip(experience_sample, discounted_state_values, value_predictions,
                    model_predictions)
        for experience, value, value_pred, model_pred in itera:
            value[experience.action] = experience.reward + self.discount_factor * (
                value_pred[np.argmax(model_pred)] if not experience.done else 0
            )

        loss = self.model.train_on_batch(original_states, discounted_state_values)
        tf.summary.scalar("loss", loss, step=self.step)  # pylint: disable=E1101
