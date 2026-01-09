from collections import deque
import random
import numpy as np
import torch

from MCTS import MCTSPlanner

class ReplayBuffer:
    def __init__(self, max_size):
        self.data = deque(maxlen=max_size)

    def add(self, sample):
        self.data.append(sample)

    def sample(self, batch_size):
        #TODO: Using Random, maybe want to se seed?
        return random.sample(self.data, batch_size)

    def __len__(self):
        return len(self.data)

def collect_self_play_games(replay_buffer, env, net, gamma = 0.99,num_episodes = 100):
    for episode in range(num_episodes):

        s = env.reset()
        done = False
        episode_steps = []

        while not done:
            #Now making a new planner everytime. This might not be the most efficient
            # It is called root shifting, might be interesting
            # Can’t reuse if state changes externally (stochastic env).
            # Yeah we have stochacicity so not doable
            planner = MCTSPlanner(net = net) #Give it the right imputs
            #THE PLANNER CAN NOT MAKE ACTIONS THE REAL ENV env, ONLY IN A COPY,
            # s IS THE OBSERVATION/STATE AND THE INFORMATION THAT NEEDS TO BE COPIED INTO THE SIMULATION ENV
            root = planner.search(s)
            pi, actions = planner.policy_from_root(root)
            a = planner.act(root, training=True)

            s_next, r, done, info = env.step(a)

            episode_steps.append((s, pi, actions, r))
            s = s_next

        # compute discounted returns
        G = 0.0
        for s, pi, actions, r in reversed(episode_steps):
            #TODO: Set gamma
            # Probably try different values of Gamma between 0.95-0.995, 0.95 for short episodes around length 50
            # Check how long our episodes are
            G = r + gamma * G
            replay_buffer.add((s, pi, actions, G))

#Same function as in MCTSPlanner, still quick check why to use this
def log_prob_diag_gaussian(mu: np.ndarray, log_std: np.ndarray, a: np.ndarray) -> float:
    # log N(a; mu, std) summed over dims
    std = np.exp(log_std)
    var = std * std
    # constant term: -0.5 * log(2π) per dim
    log2pi = np.log(2.0 * np.pi)
    return float(np.sum(-0.5 * (((a - mu) ** 2) / var + 2.0 * log_std + log2pi)))

def train_step(net, optimizer, batch, c_v = 1.0):
    optimizer.zero_grad()

    total_loss = 0.0

    #TODO: this can apparantly be stacked for faster GPU computation
    for state, pi, actions, G in batch:
        state_t = torch.from_numpy(state).float().unsqueeze(0)

        mu, log_std, v = net(state_t)

        # value loss
        value_loss = (v.squeeze() - G) ** 2

        # policy loss
        log_probs = []
        for a in actions:
            a_t = torch.from_numpy(a).float()
            log_p = log_prob_diag_gaussian(mu, log_std, a_t)
            log_probs.append(log_p)

        log_probs = torch.stack(log_probs)
        pi_t = torch.from_numpy(pi).float()

        policy_loss = -(pi_t * log_probs).sum()

        #TODO: set c_v
        # c_v <0 for more better policy control, c_v>1 for more value control.
        # between 0.5-1.0 for continouos control with dense rewards
        total_loss += policy_loss + c_v * value_loss

    total_loss /= len(batch)
    total_loss.backward()
    optimizer.step()

def run_training_loop(environment, net, optimizer = None, replay_buffer_size = 50000,num_iterations = 10000, num_episodes_per_iteration = 100, num_gradient_steps = 500, batch_size = 32):
    if optimizer is None:
        #TODO: set learning rate
        # Optionally look in to adding weight decay and gradient clipping
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

    #TODO: set the replay_buffer_size
    # probably between 50000-200000
    # Larger buffer → more stability, smaller buffer → more “fresh” targets
    replay_buffer = ReplayBuffer(replay_buffer_size)
    for iteration in range(num_iterations):

        # collect data
        collect_self_play_games(replay_buffer= replay_buffer, env = environment, net = net, num_episodes = num_episodes_per_iteration)

        # train
        #TODO: set num_gradients_steps (probably between 100-1000)
        for _ in range(num_gradient_steps):
            #TODO: set batch_size
            batch = replay_buffer.sample(batch_size)
            train_step(net, optimizer, batch)
