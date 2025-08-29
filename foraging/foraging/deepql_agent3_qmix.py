import numpy as np
import random
import foraging
import torch
import argparse
import collections
import replay_buffer as rb
import torch.nn.functional as F
parser = argparse.ArgumentParser()
parser.add_argument("--threads", default=4, type=int, help="Maximum number of threads to use.")
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--train_start", default=500, type=int, help="When to start training.")
parser.add_argument("--episodes", default=40000, type=int, help="Number of episodes.")
parser.add_argument("--epsilon", default=0.15, type=float, help="Exploration factor.")
parser.add_argument("--epsilon_final", default=0.03, type=float, help="Final exploration factor.")
parser.add_argument("--epsilon_final_at", default=100, type=int, help="Final exploration episode.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=256, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=0.0004, type=float, help="Learning rate.")
parser.add_argument("--target_tau", default=0.001, type=float, help="Target network update weight.")
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

        return q_tot.squeeze(-1).squeeze(-1)  # (batch_size)
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
    # Flatten the agent locations,normalize
    for i,pair in enumerate(state[1]):
        agents_loc_tensor[i*2] = pair[0]
        agents_loc_tensor[i*2 + 1] = pair[1]
    agents_loc_tensor = agents_loc_tensor / max_size  # normalize
    rows = []
    for i in range(agents):
        # Create a tensor for the agent ID
        agent_id = torch.nn.functional.one_hot(torch.tensor(i),num_classes = maximum_agents).float()
        rows.append(torch.cat([grid_tensor, agents_loc_tensor, agent_id]))

    return torch.stack(rows),agents_loc_tensor
    
def main(args: argparse.Namespace) -> None:
    threads = args.threads
    if threads is not None and threads > 0:
        if torch.get_num_threads() != threads:
            torch.set_num_threads(threads)
        if torch.get_num_interop_threads() != threads:
            torch.set_num_interop_threads(threads)
    WIDTH = 5
    HEIGHT = 5
    OBJECTS = 5
    AGENTS = 1
    MAXIMUM_AGENTS = 20
    ## normalize change to bigger something like 20*20
    MAX_SIZE = 5
    ACTION_SPACE = 4  # Number of actions
    env = foraging.ForagingEnvironment(WIDTH, HEIGHT, OBJECTS, AGENTS)
    # Construct the network
    #every_agent_state = every_agent_state,
    #  "every_agent_state"
    Transition = collections.namedtuple("Transition", ["state", "actions", "reward", "done", "next_state","agents"])

    # it is temporary env.h * env.w it should be changed to max_size * max_size
    STATE_DIM = env.h * env.w + MAXIMUM_AGENTS * 2  + MAXIMUM_AGENTS # Grid + agent locations + one-hot agent ID
    network = Network(STATE_DIM, ACTION_SPACE, args) 
    target =  Network(STATE_DIM, ACTION_SPACE, args)
    q_mix  =  MixingNetwork(MAXIMUM_AGENTS, STATE_DIM, args.hidden_layer_size, args) # 1) q values 2) Grid + agent locations + number of agents (one hot)
    q_mix_optimizer = torch.optim.Adam(q_mix.parameters(), lr=args.learning_rate) 
    target_q_mix = MixingNetwork(MAXIMUM_AGENTS, STATE_DIM, args.hidden_layer_size, args)
    replay_buffer = rb.MonolithicReplayBuffer(30_000)
    actions = torch.zeros(MAXIMUM_AGENTS, dtype=torch.int64)
    f = open("output.txt",'w')
    epsilon = args.epsilon
    for ep in range(args.episodes): 
        state = env.reset()
        torch_states = prepare_state(state, MAX_SIZE, MAXIMUM_AGENTS,env.agents)[0]
        R = 0
        while not env.done():
            actions = torch.zeros(MAXIMUM_AGENTS, dtype=torch.int64)
            for i in range(env.agents):
                if np.random.uniform() < epsilon:
                    action = np.random.randint(ACTION_SPACE)
                else:
                    q_values = network.predict(torch_states[i][np.newaxis])[0]
                    action = np.argmax(q_values)
                actions[i] = action
            reward, *next_state = env.perform_actions(actions)
            # compute next_torch_state
            next_torch_states = prepare_state(next_state, MAX_SIZE, MAXIMUM_AGENTS,env.agents)[0]
            # Compute per-agent reward    
            per_agent_reward = reward / env.agents

            replay_buffer.append(Transition(state=torch_states.clone(), actions=actions,
                                            reward=per_agent_reward, done=env.done(), next_state=next_torch_states.clone(),
                                            agents=env.agents))
            if len(replay_buffer) >= args.train_start:
                states, actions, rewards, dones, next_states, agents = replay_buffer.sample(args.batch_size)
                # Convert to tensors
                states      = torch.from_numpy(states).float().to(Network.device)
                actions     = torch.from_numpy(actions).long().to(Network.device)        # Use long for indexing
                rewards     = torch.from_numpy(rewards).float().to(Network.device)
                dones       = torch.from_numpy(dones).float().to(Network.device)
                next_states = torch.from_numpy(next_states).float().to(Network.device)
                number_of_agents = torch.from_numpy(agents).long().to(Network.device)


                num_agents = number_of_agents[0]

                # Compute q_values   
                q_all = network.predict(states)
                # Gather Q values for the taken actions (shape: batch_size, num_agents)
                selected_q = q_all.gather(2, actions[:,:num_agents].unsqueeze(-1)).squeeze(-1)

                #selected_q = q_all.gather(2, actions.unsqueeze(2)).squeeze(2)
                # pad q_values we suppose number_of_agents is same
                padded_q_values = torch.zeros((args.batch_size, MAXIMUM_AGENTS), dtype=torch.float32).to(Network.device)
                padded_q_values[:,:num_agents] = selected_q
                # Compute global state for mixing:
                # (to get (batch_size, STATE_DIM)), then cat with one-hot num_agents
                num_agents_one_hot = torch.nn.functional.one_hot(num_agents-1, num_classes=MAXIMUM_AGENTS).float()
                num_agents_one_hot = num_agents_one_hot.unsqueeze(0)

                # shape 64,10,85 
                # we just need one state (all 10 values has same state only differ in agent_id)
                global_state = states[:,0,:].clone()
                #Expand the tensor to the target shape (batch_size, num_agents, 20) (batch_size, MAXIMUM_AGENTS)
                # This doesn't copy the data, making it memory-efficient.
                expanded_num_agents_one_hot = num_agents_one_hot.expand(args.batch_size, MAXIMUM_AGENTS)

                global_state[:, -MAXIMUM_AGENTS:] = expanded_num_agents_one_hot

                # Compute mixed Q_tot
                q_tot = q_mix(padded_q_values, global_state)  # (batch_size)
                with torch.no_grad():
                    next_q_values = target.predict(next_states)

                    selected_next_q = next_q_values.max(dim=2)[0]
                    padded_next_q_values = torch.zeros(args.batch_size, MAXIMUM_AGENTS, dtype=torch.float32).to(Network.device)
                    padded_next_q_values[:,:num_agents] = selected_next_q

                    next_global_state = next_states[:,0,:].clone()
                    next_global_state[:, -MAXIMUM_AGENTS:] = expanded_num_agents_one_hot
                    # Compute target mixed Q_tot
                    target_q_tot = target_q_mix(padded_next_q_values, next_global_state)  # (batch_size)
                # Compute TD target (y). Note: rewards in buffer is per-agent reward, so multiply by num_agents to get total reward.
                total_rewards = rewards * num_agents
                td_target = total_rewards + args.gamma * (1 - dones) * target_q_tot
                # Compute loss (MSE between predicted q_tot and td_target)
                loss = F.mse_loss(q_tot, td_target)

                # Backpropagation and optimization
                network._optimizer.zero_grad()
                q_mix_optimizer.zero_grad()
                loss.backward()
                network._optimizer.step()
                q_mix_optimizer.step()

                # Update target networks using EMA
                network.update_params_by_ema(target._model, network._model)
                network.update_params_by_ema(target_q_mix, q_mix)
                # update state,grid
            torch_states = next_torch_states
            R += reward
        log_message = f'Finished with reward: {R:.4f}, episode number: {ep+1} epsilon: {epsilon:.4f}\n'
        f.write(log_message)
        f.flush()
        print(log_message.strip())  # Optional: Keep console output
        if args.epsilon_final_at:
            epsilon = np.interp(ep + 1, [0, args.epsilon_final_at], [args.epsilon, args.epsilon_final])

if __name__ == '__main__':


    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
    
