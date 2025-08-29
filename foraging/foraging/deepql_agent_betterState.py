import numpy as np
import random
import foraging
import torch
import argparse
import collections
import replay_buffer as rb
import global_initilizers as global_init
parser = argparse.ArgumentParser()
parser.add_argument("--threads", default=5, type=int, help="Maximum number of threads to use.")
parser.add_argument("--batch_size", default=256, type=int, help="Batch size.")
parser.add_argument("--train_start", default=5000, type=int, help="When to start training.")
parser.add_argument("--episodes", default=40000, type=int, help="Number of episodes.")
parser.add_argument("--epsilon", default=0.7, type=float, help="Exploration factor.")
parser.add_argument("--epsilon_final", default=0.03, type=float, help="Final exploration factor.")
parser.add_argument("--epsilon_final_at", default=1000, type=int, help="Training episodes.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=512, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=0.00005, type=float, help="Learning rate.")
parser.add_argument("--target_tau", default=0.0001, type=float, help="Target network update weight.")
class Network():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, ACTION_SPACE,args):
        self.args = args
        self._model = self.create_network(ACTION_SPACE).to(self.device)
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=args.learning_rate)
        self._loss = torch.nn.MSELoss()
        self.tau = args.target_tau

    def create_network(self, ACTION_SPACE):
        """Create a neural network with the specified input size and action space."""
        return torch.nn.Sequential(
        torch.nn.LazyConv2d(32, kernel_size=3, stride=1, padding=1), 
        torch.nn.ReLU(),
        torch.nn.LazyConv2d(32, kernel_size=3, stride=1, padding=1),
        torch.nn.ReLU(),
        torch.nn.Flatten(),
        torch.nn.LazyLinear(self.args.hidden_layer_size),  # Lazy from flattened
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
        # Gradient clipping - prevents exploding gradients
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
        with torch.no_grad():
            self._optimizer.step()
        self.update_params_by_ema(target,model)


    def predict(self, x):
        self._model.eval()
        with torch.no_grad():
            return self._model(x)
    
def prepare_state(agents_loc,objects_loc,torch_state,maximum_agents):
    # clear torch_state
    torch_state.zero_()
    # 1 channel for active agents
    for ag_loc in agents_loc:
        ag_x,ag_y = ag_loc
        torch_state[0][ag_x][ag_y] += 1
    # 4 channel for objects
    for ob_loc in objects_loc:
        ob_x,ob_y,ob_val = ob_loc
        if (torch_state[3][ob_x][ob_y] != 0):
            print("overlapping objects")
        torch_state[3][ob_x][ob_y] = ob_val
    # normalize
    torch_state /= maximum_agents
    return torch_state

def are_we_closer(ag_x,ag_y,nag_x,nag_y,objs_loc):
    for obj_loc in objs_loc:
        old_d = np.abs(ag_x-obj_loc[0]) + np.abs(ag_y-obj_loc[1])
        new_d = np.abs(nag_x-obj_loc[0]) + np.abs(nag_y-obj_loc[1])
        if (new_d <= old_d):
            return True
    return False
def main(args: argparse.Namespace) -> None:
    global_init.global_keras_initializers()
    threads = args.threads
    if threads is not None and threads > 0:
        if torch.get_num_threads() != threads:
            torch.set_num_threads(threads)
        if torch.get_num_interop_threads() != threads:
            torch.set_num_interop_threads(threads)
    WIDTH = 5
    HEIGHT = 5
    OBJECTS = 5
    AGENTS = 5
    MAXIMUM_AGENTS = 20
    ## normalize change to bigger something like 20*20
    MAX_SIZE = 5
    ACTION_SPACE = 4  # Number of actions
    env = foraging.ForagingEnvironment(WIDTH, HEIGHT, OBJECTS, AGENTS)
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])
    # it is temporary env.h * env.w it should be changed to max_size * max_size
    network = Network(ACTION_SPACE, args) 
    target =  Network(ACTION_SPACE, args)
    replay_buffer = rb.MonolithicReplayBuffer(100_000)
    actions = [0] * env.agents
    epsilon = args.epsilon
    returns = torch.zeros(200,dtype=torch.float32).to(Network.device)
    f = open("output.txt",'w')
    # 0 channel for active agents
    # 1 channel for inactive agents
    # 2 channel for current agent
    # 3 channel for objects
    torch_state = torch.zeros(4, WIDTH, HEIGHT)
    started = False
    for ep in range(args.episodes): 
        state = env.reset()
        torch_state = prepare_state(env.agent_locations, env.object_locations, torch_state,MAXIMUM_AGENTS)
        R = 0
        while not env.done():
            every_agent_state = []
            for i in range(env.agents):
                obj_loc = env.object_locations
                ag_x,ag_y = env.agent_locations[i]
                playable_actions = []
                for a in range(ACTION_SPACE):
                    nag_x,nag_y = env.get_nag_xy(a,i)
                    if are_we_closer(ag_x,ag_y,nag_x,nag_y,obj_loc) == True:
                        playable_actions.append(a)
                if (len(playable_actions) == 0 and len(obj_loc) > 1):
                    print("no playable actions")
                    for a in range(ACTION_SPACE):
                        nag_x,nag_y = env.get_nag_xy(a,i)
                        if are_we_closer(ag_x,ag_y,nag_x,nag_y,obj_loc) == True:
                            playable_actions.append(a)
                if (len(playable_actions) == 0):
                    playable_actions = [0,1,2,3]
                # set current agent
                torch_state[2,ag_x,ag_y] = 1
                if np.random.uniform() < epsilon:
                    action_index = np.random.randint(len(playable_actions))
                    action = playable_actions[action_index]
                    #action = np.random.randint(ACTION_SPACE)
                else:
                    q_values = network.predict(torch_state[np.newaxis])[0]
                    ## old code
                    #action = torch.argmax(q_values[playable_actions])
                    # actions_argsort = torch.argsort(input=q_values,descending=True)
                    # for act in actions_argsort:
                    #     if act in playable_actions:
                    #         action = act
                    #         actions[i] = action
                    #         break
                    # ## old code
                    # q_values[playable_actions] += np.max(q_values)
                    
                    action = playable_actions[np.argmax(q_values[playable_actions])]
                actions[i] = action
                every_agent_state.append(torch_state.clone())

                nag_x, nag_y = env.get_nag_xy(action, i)
                # unset current agent
                torch_state[2, ag_x, ag_y] = 0
                # unset active agents
                torch_state[0, ag_x, ag_y] -= (1 / MAXIMUM_AGENTS)
                # set inactive agents
                torch_state[1, nag_x, nag_y] += (1 / MAXIMUM_AGENTS)
            reward, *next_state = env.perform_actions(actions)
            ## end of agent loop ##
            #update obj loc
            torch_state[3] = 0
            for ob_x, ob_y, ob_val in env.object_locations:
                torch_state[3, ob_x, ob_y] = ob_val / MAXIMUM_AGENTS
            # move inactive agents to active
            for nag_x,nag_y in env.agent_locations:
                torch_state[1, nag_x, nag_y] -= (1 / MAXIMUM_AGENTS)
                torch_state[0, nag_x, nag_y] += (1 / MAXIMUM_AGENTS)
            ## add as current agent 0
            ag_x,ag_y = env.agent_locations[0]
            torch_state[2, ag_x, ag_y] = 1
            # compute next_torch_state
            next_torch_state = torch_state
            # Compute per-agent reward
            per_agent_reward = reward / env.agents
            ## REPLAY BUFFER APPEND
            ## add into transition next attribute (whole_game_state and grid)
            for i in range(env.agents):
                replay_buffer.append(Transition(state=every_agent_state[i].clone(), action=actions[i], reward=per_agent_reward, done=env.done(), next_state=next_torch_state.clone()))
            if len(replay_buffer) >= args.train_start:
                if not started:
                    started = True
                    print("start training")
                states, actions, rewards, dones, next_states = replay_buffer.sample(args.batch_size)
                # Convert to tensors
                states      = torch.from_numpy(states).float().to(Network.device)
                actions     = torch.from_numpy(actions).long().to(Network.device)        # Use long for indexing
                rewards     = torch.from_numpy(rewards).float().to(Network.device)
                dones       = torch.from_numpy(dones).int().to(Network.device)
                next_states = torch.from_numpy(next_states).float().to(Network.device)


                next_q_values = target.predict(next_states)
                targets = network.predict(states)
                
                targets[torch.arange(args.batch_size),actions] = rewards + args.gamma * torch.max(next_q_values,dim=1)[0] * (1 - dones)
                network.train(states, targets, target._model,network._model)
            # update state,grid,
            torch_state = next_torch_state
            R += reward
        returns[ep % 200] = R
        average_return = returns.mean().item()
        if args.epsilon_final_at:
            epsilon = np.interp(ep + 1, [0, args.epsilon_final_at], [args.epsilon, args.epsilon_final])
        log_message = f'Finished with reward: {R:.4f}, average_return: {average_return:.4f} episode number: {ep+1} epsilon: {epsilon:.4f}\n'
        f.write(log_message)
        f.flush()
        print(log_message.strip())  # Optional: Keep console output
    

if __name__ == '__main__':




    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
    
