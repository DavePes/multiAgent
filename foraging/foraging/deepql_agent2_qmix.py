import numpy as np
import random
import foraging
import torch
import argparse
import collections
import replay_buffer as rb
parser = argparse.ArgumentParser()
parser.add_argument("--threads", default=5, type=int, help="Maximum number of threads to use.")
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--train_start", default=5000, type=int, help="When to start training.")
parser.add_argument("--episodes", default=40000, type=int, help="Number of episodes.")
parser.add_argument("--epsilon", default=0.15, type=float, help="Exploration factor.")
parser.add_argument("--epsilon_final", default=0.05, type=float, help="Final exploration factor.")
parser.add_argument("--epsilon_final_at", default=2500, type=int, help="Training episodes.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=256, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=0.000_04, type=float, help="Learning rate.")
parser.add_argument("--target_tau", default=0.00_05, type=float, help="Target network update weight.")
class Network():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, input_size, ACTION_SPACE,args):
        self.args = args
        self._model = self.create_network(input_size, ACTION_SPACE)
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=args.learning_rate)
        self._loss = torch.nn.MSELoss()
        self.tau = args.target_tau

    def create_network(self, input_size, ACTION_SPACE):
        """Create a neural network with the specified input size and action space."""
        return torch.nn.Sequential(
            torch.nn.Linear(input_size, self.args.hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.args.hidden_layer_size, self.args.hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.args.hidden_layer_size, ACTION_SPACE),
        ).to(self.device)
    
    def update_params_by_ema(self,target: torch.nn.Module, source: torch.nn.Module) -> None:
        """Update target parameters using exponential moving average of the source parameters.

        Parameters:
        target: The target model whose parameters will be updated.
        source: The source model whose parameters will be used for the update.
        tau: The decay factor for the exponential moving average, e.g., 0.001.
        """
        with torch.inference_mode():
            for target_param, source_param in zip(target.parameters(), source.parameters()):
                target_param.mul_(1 - self.tau)
                target_param.add_(source_param, alpha=self.tau)

    def train(self,states,next_q_values,target,model):
        self._model.train()
        predictions = self._model(states)
        self._optimizer.zero_grad()
        loss = self._loss(predictions, next_q_values)
        loss.backward()
        with torch.no_grad():
            self._optimizer.step()
        self.update_params_by_ema(target,model)


    def predict(self, x):
        self._model.eval()
        with torch.no_grad():
            return self._model(x)
    
def prepare_state(state, max_size,maximum_agents,permuted_indices):
    # Flatten the grid
    flat_grid = [cell for row in state[0] for cell in row]
    # Convert to tensors
    grid_tensor = torch.tensor(flat_grid, dtype=torch.float32) / maximum_agents # normalize  
    # Create a tensor for agent locations
    agents_loc_tensor = -torch.ones(maximum_agents * 2, dtype=torch.float32)
    # Flatten the agent locations,normalize and permute agents location
    for i,pair in enumerate(state[1]):
        index = permuted_indices[i]
        agents_loc_tensor[index*2] = pair[0]
        agents_loc_tensor[index*2 + 1] = pair[1]
    agents_loc_tensor = agents_loc_tensor / max_size  # normalize
    # Create a tensor for the agent ID
    agent_id_one_hot = torch.nn.functional.one_hot(torch.tensor(0),num_classes = maximum_agents).float()
    return torch.cat([grid_tensor, agents_loc_tensor, agent_id_one_hot]),grid_tensor
    
def main(env: foraging.ForagingEnvironment, args: argparse.Namespace) -> None:

    # Construct the network
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "done", "next_state","agent_id"])
    MAXIMUM_AGENTS = 20
    ## normalize change to bigger something like 20*20
    MAX_SIZE = 5
    ACTION_SPACE = 4  # Number of actions
    # it is temporary env.h * env.w it should be changed to max_size * max_size
    networks = np.array([Network(env.h * env.w + MAXIMUM_AGENTS*2 + MAXIMUM_AGENTS, ACTION_SPACE, args) for _ in range(env.agents)],dtype=object)
    target_networks = np.array([Network(env.h * env.w + MAXIMUM_AGENTS*2 + MAXIMUM_AGENTS, ACTION_SPACE, args) for _ in range(env.agents)],dtype=object)
    replay_buffer = rb.MonolithicReplayBuffer(600_000)
    actions = [0] * env.agents
    f = open("output.txt",'w')
    for ep in range(args.episodes): 
        state = env.reset()
        permuted_indices = torch.arange(env.agents) #torch.randperm(env.agents)
        torch_state,grid_tensor = prepare_state(state, MAX_SIZE, MAXIMUM_AGENTS, permuted_indices)
        R = 0
        while not env.done():
            every_agent_state = []
            for i in range(env.agents):
                agent_id_one_hot = torch.nn.functional.one_hot(torch.tensor(i),num_classes = MAXIMUM_AGENTS)
                torch_state[-MAXIMUM_AGENTS:] = agent_id_one_hot.float()
                if np.random.uniform() < args.epsilon:
                    action = np.random.randint(ACTION_SPACE)
                else:
                    q_values = networks[i].predict(torch_state[np.newaxis])[0]
                    action = np.argmax(q_values)
                actions[permuted_indices[i]] = action
                every_agent_state.append(torch_state.clone())
                new_torch_state = env.perform_one_action(action,torch_state,grid_tensor.numel(),permuted_indices,i,MAX_SIZE)
                torch_state = new_torch_state
            reward, *next_state = env.perform_actions(actions)
            # compute next_torch_state
            next_permuted_indices = torch.arange(env.agents) #torch.randperm(env.agents)
            next_torch_state, next_grid_tensor = prepare_state(next_state, MAX_SIZE, MAXIMUM_AGENTS, next_permuted_indices)
            # Compute per-agent reward
            per_agent_reward = reward / env.agents
            ## REPLAY BUFFER APPEND
            ## add into transition next attribute (whole_game_state and grid)
            for i in range(env.agents):
                ## check if every_agent_state is right!!
                replay_buffer.append(Transition(state=every_agent_state[permuted_indices[i]], action=actions[permuted_indices[i]], reward=per_agent_reward, done=env.done(), next_state=next_torch_state,agent_id=permuted_indices[i]))
            if len(replay_buffer) >= args.train_start:
                states, actions, rewards, dones, next_states,agent_ids = replay_buffer.sample(args.batch_size)
                # Convert to tensors
                states      = torch.from_numpy(states).float().to(Network.device)
                actions     = torch.from_numpy(actions).long().to(Network.device)        # Use long for indexing
                rewards     = torch.from_numpy(rewards).float().to(Network.device)
                dones       = torch.from_numpy(dones).float().to(Network.device)
                next_states = torch.from_numpy(next_states).float().to(Network.device)
                next_q_values = target_networks[agent_ids].predict(next_states)
                targets = networks[agent_ids].predict(states)
                
                targets[torch.arange(args.batch_size),actions] = rewards + args.gamma * torch.max(next_q_values,dim=1)[0] * (1 - dones.int())
                networks[agent_ids].train(states, targets, target_networks[agent_ids]._model,networks[agent_ids]._model)
                # update state,grid,permuted_indices
                torch_state = next_torch_state
                grid_tensor = next_grid_tensor
                permuted_indices = next_permuted_indices
            R += reward
        log_message = f'Finished with reward: {R}, episode number: {ep+1}\n'
        f.write(log_message)
        f.flush()
        print(log_message.strip())  # Optional: Keep console output
    

if __name__ == '__main__':

    WIDTH = 5
    HEIGHT = 5
    OBJECTS = 10
    AGENTS = 10
    
    env = foraging.ForagingEnvironment(WIDTH, HEIGHT, OBJECTS, AGENTS)

    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(env, main_args)
    
