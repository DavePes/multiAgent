#!/usr/bin/env python3
import argparse
import collections

import gymnasium as gym
import numpy as np
import torch

import npfl139
npfl139.require_version("2425.4")

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--epsilon", default=0.5, type=float, help="Exploration factor.")
parser.add_argument("--epsilon_final", default=None, type=float, help="Final exploration factor.")
parser.add_argument("--epsilon_final_at", default=None, type=int, help="Training episodes.")
parser.add_argument("--gamma", default=1.0, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=50, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")
parser.add_argument("--target_update_freq", default=0, type=int, help="Target update frequency.")


class Network:
    # Use GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
        # TODO: Create a suitable model and store it as `self._model`.
        self._model = torch.nn.Sequential(
            torch.nn.Linear(env.observation_space.shape[0], args.hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(args.hidden_layer_size, env.action_space.n),
        ).to(self.device)

        # TODO: Define a suitable optimizer from `torch.optim`.
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=args.learning_rate)

        # TODO: Define the loss (most likely some `torch.nn.*Loss`).
        self._loss = torch.nn.MSELoss()

    # Define a training method. Generally you have two possibilities
    # - pass new q_values of all actions for a given state; all but one are the same as before
    # - pass only one new q_value for a given state, and include the index of the action to which
    #   the new q_value belongs
    # The code below implements the first option, but you can change it if you want.
    #
    # The `npfl139.typed_torch_function` automatically converts input arguments
    # to PyTorch tensors of given type, and converts the result to a NumPy array.
    @npfl139.typed_torch_function(device, torch.float32, torch.float32)
    def train(self, states: torch.Tensor, q_values: torch.Tensor) -> None:
        self._model.train()
        predictions = self._model(states)
        loss = self._loss(predictions, q_values)
        self._optimizer.zero_grad()
        loss.backward()
        with torch.no_grad():
            self._optimizer.step()

    @npfl139.typed_torch_function(device, torch.float32)
    def predict(self, states: torch.Tensor) -> np.ndarray:
        self._model.eval()
        with torch.no_grad():
            return self._model(states)

    # If you want to use target network, the following method copies weights from
    # a given Network to the current one.
    def copy_weights_from(self, other: "Network") -> None:
        self._model.load_state_dict(other._model.state_dict())


def main(env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl139.startup(args.seed, args.threads)
    npfl139.global_keras_initializers()  # Use Keras-style Xavier parameter initialization.

    # Construct the network
    network = Network(env, args)
    target = Network(env, args) if args.target_update_freq else network

    # Replay memory; the `max_length` parameter can be passed to limit its size.
    replay_buffer = npfl139.ReplayBuffer()
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])

    epsilon = args.epsilon
    training = True
    while training:
        # Perform episode
        state, done = env.reset()[0], False
        while not done:
            # TODO: Choose an action.
            # You can compute the q_values of a given state by
            #   q_values = network.predict(state[np.newaxis])[0]
            q_values = network.predict(state[np.newaxis])[0]
            action = np.argmax(q_values) if np.random.uniform() >= epsilon else env.action_space.sample()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Append state, action, reward, done and next_state to replay_buffer
            replay_buffer.append(Transition(state, action, reward, done, next_state))

            # TODO: If the `replay_buffer` is large enough, perform training using
            # a batch of `args.batch_size` uniformly randomly chosen transitions.
            #
            # The `replay_buffer` offers a method with signature
            #   sample(self, size, generator=np.random, replace=True) -> list[Transition]
            # which returns uniformly selected batch of `size` transitions, either with
            # replacement (which is much faster, and hence the default) or without.
            # By default, `np.random` is used to generate the random indices, but you can
            # pass your own `np.random.RandomState` instance.

            # After you compute suitable targets, you can train the network by
            #   network.train(...)
            if len(replay_buffer) >= 4 * args.batch_size:
                batch = replay_buffer.sample(args.batch_size)
                states, actions, rewards, dones, next_states = map(np.array, zip(*batch))

                targets = network.predict(states)
                estimates = rewards + args.gamma * (1 - dones) * target.predict(next_states).max(axis=-1)
                targets[np.arange(args.batch_size), actions] = estimates
                network.train(states, targets)

            state = next_state

        # Copy to target network
        if args.target_update_freq and env.episode % args.target_update_freq == 0:
            target.copy_weights_from(network)

        # End when reaching an average reward of 475
        if env.episode % 50 == 0:
            returns = 0
            for _ in range(10):
                state, done = env.reset(logging=False)[0], False
                while not done:
                    action = np.argmax(network.predict(state[np.newaxis])[0])
                    state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    returns += reward / 10
            print("Evaluation after episode {} returned {}".format(env.episode, returns))
            if returns >= 475:
                training = False

        if args.epsilon_final_at:
            epsilon = np.interp(env.episode + 1, [0, args.epsilon_final_at], [args.epsilon, args.epsilon_final])

    # Final evaluation
    while True:
        state, done = env.reset(start_evaluation=True)[0], False
        while not done:
            # TODO: Choose (greedy) action
            action = np.argmax(network.predict(state[np.newaxis])[0])
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    main_env = npfl139.EvaluationEnv(gym.make("CartPole-v1"), main_args.seed, main_args.render_each)

    main(main_env, main_args)