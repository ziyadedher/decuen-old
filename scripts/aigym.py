import sys
import time

import gym  # type: ignore
import tensorflow as tf  # type: ignore
from tensorflow.keras import models, layers, activations, optimizers, losses  # type: ignore

from decuen.policies import EpsilonGreedyPolicy
from decuen.experience import Experience, DequeExperienceManager, PrioritizedExperienceManager
from decuen.agents import DQNAgent, DDQNAgent


def main() -> None:
    env = gym.make("CartPole-v1")

    logname = f"{time.time()}" if len(sys.argv) == 1 else sys.argv[1]
    summary_writer = tf.summary.create_file_writer(f"logs/{logname}")
    summary_writer.init()
    summary_writer.set_as_default()

    model = models.Sequential([
        layers.Dense(32, activation=activations.relu, input_shape=env.observation_space.shape),
        layers.Dense(64, activation=activations.relu),
        layers.Dense(128, activation=activations.relu),
        layers.Dense(env.action_space.n, activation=activations.relu),
    ])
    model.compile(optimizers.Adam(lr=0.0025), loss=losses.mean_squared_error)

    policy = EpsilonGreedyPolicy(
        env.action_space.n,
        max_epsilon=1, min_epsilon=0.05,
        annealing_method=EpsilonGreedyPolicy.AnnealingMethod.EXPONENTIAL, annealing_constant=1e-4,
    )
    # experience_manager: DequeExperienceManager[int] = DequeExperienceManager(

    experience_manager: PrioritizedExperienceManager[int] = PrioritizedExperienceManager(
        sample_size=64,
        memory_capacity=1 << 14,
        omega=1,
    )
    # DQNAgent
    agent = DDQNAgent(
        env.observation_space.shape, policy, experience_manager,
        env.action_space.n, model,
        discount_factor=0.99,
        target_update_rate=1000,
    )
    
    try:
        for i_episode in range(500):
            state = env.reset()
            total_reward = 0
            for timestep in range(500):
                if i_episode % 10 == 0:
                    env.render()
                action = agent.act(state)
                new_state, reward, done, _ = env.step(action)
                experience = Experience(state, action, new_state, reward, done)  # type: ignore
                state = new_state

                agent.experience(experience)

                total_reward += reward
                if done:
                    print(f"Episode {i_episode + 1} finished after {timestep + 1} timesteps.")
                    tf.summary.scalar("reward/episode", total_reward, agent.step)  # pylint: disable=E1101
                    break
    except KeyboardInterrupt:
        print("Stopping...")


if __name__ == '__main__':
    main()
