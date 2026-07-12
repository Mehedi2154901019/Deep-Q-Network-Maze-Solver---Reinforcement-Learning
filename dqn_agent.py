import random
from turtle import done
import copy
import torch
import torch.nn as nn
import torch.optim as optim

from dqn import DQN
from replay_buffer import ReplayBuffer

ACTIONS = [
    "UP",
    "DOWN",
    "LEFT",
    "RIGHT"
]

def action_name(action_index):

    return ACTIONS[action_index]

class DQNAgent:

    def __init__(self):

        self.state_size = 2

        self.action_size = 4

        self.gamma = 0.99

        self.epsilon = 1.0

        self.epsilon_min = 0.01

        self.epsilon_decay = 0.995

        self.batch_size = 32

        self.update_count = 0

        self.target_update_frequency = 100

        self.model = DQN(
            self.state_size,
            self.action_size
        )

        self.target_model = copy.deepcopy(self.model)

        self.target_model.eval()

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=0.001
        )

        self.criterion = nn.MSELoss()

        self.memory = ReplayBuffer(
            capacity=10000
        )

    def choose_action(self, state):

        # Exploration
        if random.random() < self.epsilon:

            return random.randint(0, self.action_size - 1)

        # Exploitation
        state = torch.tensor(
            state,
            dtype=torch.float32
        )

        with torch.no_grad():

            q_values = self.model(state)

        return torch.argmax(q_values).item()
    
    def choose_best_action(self, state):

        state = torch.tensor(
            state,
            dtype=torch.float32
        )

        with torch.no_grad():

            q_values = self.model(state)

        return torch.argmax(q_values).item()
    
    def remember(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):

        self.memory.push(
            state,
            action,
            reward,
            next_state,
            done
        )
    
    def memory_size(self):

        return len(self.memory)

    def train_step(self):

        if len(self.memory) < self.batch_size:
            return

        batch = self.memory.sample(self.batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(
            states,
            dtype=torch.float32
        )

        actions = torch.tensor(
            actions,
            dtype=torch.long
        )

        rewards = torch.tensor(
            rewards,
            dtype=torch.float32
        )

        next_states = torch.tensor(
            next_states,
            dtype=torch.float32
        )

        dones = torch.tensor(
            dones,
            dtype=torch.float32
        )

        # Current Q-values
        current_q = self.model(states)

        current_q = current_q.gather(
            1,
            actions.unsqueeze(1)
        ).squeeze(1)

        # Target Q-values
        with torch.no_grad():

            next_q = self.target_model(next_states)

            max_next_q = next_q.max(1)[0]

            target_q = rewards + (1 - dones) * self.gamma * max_next_q

        loss = self.criterion(
            current_q,
            target_q
        )

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        self.update_count += 1

        if self.update_count % self.target_update_frequency == 0:

            self.target_model.load_state_dict(
                self.model.state_dict()
            )