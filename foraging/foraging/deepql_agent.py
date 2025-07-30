import numpy as np
import random
import foraging
import torch
import argparse
import collections
import replay_buffer
parser = argparse.ArgumentParser()
parser.add_argument("--threads", default=4, type=int, help="Maximum number of threads to use.")
parser.add_argument("--batch_size", default=100, type=int, help="Batch size.")
parser.add_argument("--train_start", default=500, type=int, help="When to start training.")
parser.add_argument("--episodes", default=4000, type=int, help="Number of episodes.")
parser.add_argument("--epsilon", default=1.0, type=float, help="Exploration factor.")
parser.add_argument("--epsilon_final", default=0.05, type=float, help="Final exploration factor.")
parser.add_argument("--epsilon_final_at", default=2500, type=int, help="Training episodes.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=256, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=0.0001, type=float, help="Learning rate.")
parser.add_argument("--target_tau", default=0.001, type=float, help="Target network update weight.")
class Network(torch.nn.Module):
    def __init__(self, input_size, action_space,args):
        self._model = self.create_network(input_size, action_space,args)
        self._target = self.create_network(input_size, action_space,args)
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=args.learning_rate)

        self._loss = torch.nn.MSELoss()
    def create_network(self, input_size, action_space,args):
        """Create a neural network with the specified input size and action space."""
        return torch.nn.Sequential(
            torch.nn.Linear(input_size, args.hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(args.hidden_layer_size, args.hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(args.hidden_layer_size, action_space),
        ).to(self.device)
    
    def update_params_by_ema(target: torch.nn.Module, source: torch.nn.Module, tau: float) -> None:
        """Update target parameters using exponential moving average of the source parameters.

        Parameters:
        target: The target model whose parameters will be updated.
        source: The source model whose parameters will be used for the update.
        tau: The decay factor for the exponential moving average, e.g., 0.001.
        """
        with torch.inference_mode():
            for target_param, source_param in zip(target.parameters(), source.parameters()):
                target_param.mul_(1 - tau)
                target_param.add_(source_param, alpha=tau)

    def train(self,q_values,next_q_values):
        self._model.train()
        self._optimizer.zero_grad()
        loss = self._loss(q_values, next_q_values)
        loss.backward()
        with torch.no_grad():
            self._optimizer.step()
        self.update_params_by_ema(self._target, self._model,self.tau)


    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self._model(x)
    
class ExampleAgent:

    def __init__(self, w, h, a):
        self.width = w
        self.height = h
        self.agents = a
    # returns list of actions for all the agents
    def action(self, state):
        world_map, agent_locations = state # state is a tuple

        return [random.randrange(0,4) for _ in agent_locations]

    # is called to inform the agents about the reward from previous step
    def reward(self, reward):
        pass

def main(env: foraging.ForagingEnvironment, args: argparse.Namespace) -> None:

    # Construct the network
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])
    network = Network(env.h * env.w, 4, args)  # Assuming 4 possible actions (up, down, left, right)

    for _ in range(args.episodes):
        state = env.reset()
        R = 0
        while not env.done():
            if (np.random.rand() < args.epsilon):
                action = random.randrange(0, 4)
            else:
                action = np.argmax(network.predict(torch.tensor(state[0]).float().flatten()))
            reward, *next_state = env.perform_actions(agent.action(state))
            agent.reward(reward)
            R += reward
            
            replay_buffer.append(Transition(state, action, reward, done, next_state))
            state = next_state
        print(f'Finished with reward: {R}')
    

if __name__ == '__main__':

    WIDTH = 5
    HEIGHT = 5
    OBJECTS = 10
    AGENTS = 1

    env = foraging.ForagingEnvironment(WIDTH, HEIGHT, OBJECTS, AGENTS)

    agent = ExampleAgent(WIDTH, HEIGHT, AGENTS)
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(env, main_args)
    
    env.render_history()