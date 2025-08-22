import numpy as np
import random
import foraging
import torch
import argparse
import collections
import replay_buffer as rb
import torch.nn.functional as F
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
class MixingNetwork(torch.nn.Module):
    def __init__(self, num_agents, state_dim, hidden_dim, args):
        super(MixingNetwork, self).__init__()
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.device = Network.device
        
        # Hypernetwork for weights (non-negative via abs)
        self.hyper_w1 = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, num_agents * hidden_dim)
        )
        self.hyper_b1 = torch.nn.Linear(state_dim, hidden_dim)

        self.hyper_w2 = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim * 1)
        )
        self.hyper_b2 = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)
        )
    def forward(self, q_values, global_state):  # q_values: (batch_size, maximum_agents), global_state: (batch_size, state_dim)
        batch_size = q_values.size(0)
        
        # Layer 1
        w1 = torch.abs(self.hyper_w1(global_state)).view(batch_size, self.num_agents, self.hidden_dim)
        b1 = self.hyper_b1(global_state).view(batch_size, 1, self.hidden_dim)
        hidden = F.elu(torch.bmm(q_values.unsqueeze(1), w1) + b1)  # (batch_size, 1, hidden_dim)
        
        # Layer 2
        w2 = torch.abs(self.hyper_w2(global_state)).view(batch_size, self.hidden_dim, 1)
        b2 = self.hyper_b2(global_state).view(batch_size, 1, 1)
        q_tot = torch.bmm(hidden, w2) + b2  # (batch_size, 1, 1)
        
        return q_tot.squeeze(2)  # (batch_size, 1)
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
    
def prepare_state(state, max_size,maximum_agents,agents):
    # Flatten the grid
    flat_grid = [cell for row in state[0] for cell in row]
    # Convert to tensors
    grid_tensor = torch.tensor(flat_grid, dtype=torch.float32) / maximum_agents # normalize  
    # Create a tensor for agent locations
    agents_loc_tensor = -torch.ones(maximum_agents * 2, dtype=torch.float32)
    # Flatten the agent locations,normalize and permute agents location
    for i,pair in enumerate(state[1]):
        agents_loc_tensor[i*2] = pair[0]
        agents_loc_tensor[i*2 + 1] = pair[1]
    agents_loc_tensor = agents_loc_tensor / max_size  # normalize
    # Create a tensor for the agent ID
    agent_id = torch.nn.functional.one_hot(torch.tensor(0),num_classes = maximum_agents).float()
    number_of_agents_one_hot = torch.nn.functional.one_hot(torch.tensor(agents), num_classes=maximum_agents).float()

    return torch.cat([grid_tensor, agents_loc_tensor, agent_id]),grid_tensor,torch.cat([grid_tensor, agents_loc_tensor, number_of_agents_one_hot])
    
def main(args: argparse.Namespace) -> None:
    WIDTH = 5
    HEIGHT = 5
    OBJECTS = 5
    AGENTS = 10
    
    env = foraging.ForagingEnvironment(WIDTH, HEIGHT, OBJECTS, AGENTS)
    # Construct the network
    # we suppose no permutation!!!
    # also we need to track all actions of each agent!!
    #every_agent_state = every_agent_state,
    #  "every_agent_state"
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "done", "next_state", "qmix_state","every_agent_state","agents"])
    MAXIMUM_AGENTS = 20
    ## normalize change to bigger something like 20*20
    MAX_SIZE = 5
    ACTION_SPACE = 4  # Number of actions
    # it is temporary env.h * env.w it should be changed to max_size * max_size
    STATE_DIM = env.h * env.w + MAXIMUM_AGENTS * 2 + MAXIMUM_AGENTS  # Grid + agent locations + agent ID (one hot)
    network = Network(STATE_DIM, ACTION_SPACE, args) 
    target =  Network(STATE_DIM, ACTION_SPACE, args)
    q_mix  =  MixingNetwork(MAXIMUM_AGENTS, STATE_DIM, args.hidden_layer_size, args) # Grid + agent locations + number of agents (one hot)
    q_mix_optimizer = torch.optim.Adam(q_mix.parameters(), lr=args.learning_rate) 

    replay_buffer = rb.MonolithicReplayBuffer(600_000)
    actions = [0] * env.agents
    f = open("output.txt",'w')
    for ep in range(args.episodes): 

        state = env.reset()
        ## prepare state 
        # torch_state (grid + agents_loc + agent_id one hot)
        # grid_tensor (grid)
        # qmix_state (grid + agents_loc + number_of_agents one hot)
        torch_state,grid_tensor,qmix_state = prepare_state(state, MAX_SIZE, MAXIMUM_AGENTS,env.agents)
        R = 0
        while not env.done():
            every_agent_state = torch.zeros((env.agents, STATE_DIM), dtype=torch.float32)
            for i in range(env.agents):
                agent_id = torch.nn.functional.one_hot(torch.tensor(i),num_classes = MAXIMUM_AGENTS)
                torch_state[-MAXIMUM_AGENTS:] = agent_id.float()
                q_values = network.predict(torch_state[np.newaxis])[0]
                if np.random.uniform() < args.epsilon:
                    action = np.random.randint(ACTION_SPACE)
                else:
                    action = np.argmax(q_values)
                actions[i] = action
                every_agent_state[i] = torch_state.clone()
                new_torch_state = env.perform_one_action(action,torch_state,grid_tensor.numel(),i,MAX_SIZE)
                torch_state = new_torch_state
            reward, *next_state = env.perform_actions(actions)
            # compute next_torch_state
            next_torch_state, next_grid_tensor,next_q_mix_state = prepare_state(next_state, MAX_SIZE, MAXIMUM_AGENTS,env.agents)
            # Compute per-agent reward
            per_agent_reward = reward / env.agents
            ## REPLAY BUFFER APPEND
            ## add into transition next attribute (whole_game_state and grid)
            for i in range(env.agents):
                # 1/env.agents - env.agents*0.03 + 0.7
                if (np.random.uniform() < 0.5):
                    replay_buffer.append(Transition(state=every_agent_state[i], action=actions[i], 
                                                    reward=per_agent_reward, done=env.done(), 
                                                    next_state=next_torch_state, 
                                                    qmix_state=qmix_state,
                                                    every_agent_state = every_agent_state,
                                                    agents=env.agents))
            if len(replay_buffer) >= args.train_start:
                states, actions, rewards, dones, next_states,qmix_state,every_agent_state,agents = replay_buffer.sample(args.batch_size)
                # Convert to tensors
                states      = torch.from_numpy(states).float().to(Network.device)
                actions     = torch.from_numpy(actions).long().to(Network.device)        # Use long for indexing
                rewards     = torch.from_numpy(rewards).float().to(Network.device)
                dones       = torch.from_numpy(dones).float().to(Network.device)
                next_states = torch.from_numpy(next_states).float().to(Network.device)
                every_agent_state = torch.from_numpy(every_agent_state).float().to(Network.device)
                qmix_state = torch.from_numpy(qmix_state).float().to(Network.device)
                number_of_agents = torch.from_numpy(agents).int().to(Network.device)
                # Compute q_mix
                
                # Compute
                q_values_every_agent = torch.zeros((args.batch_size, MAXIMUM_AGENTS), dtype=torch.float32).to(Network.device)
                q_values_every_agent[:,:number_of_agents[0]] = torch.max(network.predict(every_agent_state),dim=-1)[0]

                q_mix(q_values_every_agent, qmix_state)
                
                targets = network.predict(states)
                
                targets[torch.arange(args.batch_size),actions] = rewards + args.gamma * torch.max(target.predict(next_states),dim=1)[0] * (1 - dones.int())
                network.train(states, targets, target._model,network._model)
                # update state,grid
                torch_state = next_torch_state
                grid_tensor = next_grid_tensor
                qmix_state = next_q_mix_state
            R += reward
        log_message = f'Finished with reward: {R}, episode number: {ep+1}\n'
        f.write(log_message)
        f.flush()
        print(log_message.strip())  # Optional: Keep console output
    

if __name__ == '__main__':


    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
    
