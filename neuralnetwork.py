import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Transformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        # Input embedding layer
        self.input_embedding = nn.Linear(input_dim, hidden_dim)
        
        # Multi-head attention layers
        self.attention_layers = nn.ModuleList()
        for _ in range(num_layers):
            attention_layer = nn.MultiheadAttention(hidden_dim, num_heads)
            self.attention_layers.append(attention_layer)
            
        # Feedforward layers
        self.feedforward_layers = nn.ModuleList()
        for _ in range(num_layers):
            feedforward_layer = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.ReLU(),
                nn.Linear(hidden_dim * 4, hidden_dim)
            )
            self.feedforward_layers.append(feedforward_layer)
            
    def forward(self, x):
        # Input embedding
        x = self.input_embedding(x)
        
        # Multi-head attention layers
        for attention_layer in self.attention_layers:
            x = F.layer_norm(x, (self.hidden_dim,))
            x, _ = attention_layer(x, x, x)
            
        # Feedforward layers
        for feedforward_layer in self.feedforward_layers:
            x = F.layer_norm(x, (self.hidden_dim,))
            x = feedforward_layer(x)
            
        return x

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)

    def forward(self, x, h=None):
        batch_size = x.size(0)

        if h is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        else:
            h0, c0 = h

        out, (h, c) = self.rnn(x, (h0, c0))

        return self.dropout(h[-1])


class DQNAgent(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128):
        super(DQNAgent, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNAgentTrainer:
    def __init__(self, input_size, output_size, learning_rate=0.001, discount_factor=0.99, replay_memory_size=10000, batch_size=32):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.agent = DQNAgent(input_size, output_size).to(self.device)
        self.target_agent = DQNAgent(input_size, output_size).to(self.device)
        self.target_agent.load_state_dict(self.agent.state_dict())
        self.target_agent.eval()

        self.optimizer = optim.Adam(self.agent.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        self.discount_factor = discount_factor
        self.replay_memory_size = replay_memory_size
        self.batch_size = batch_size

        self.replay_memory = []

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_memory.append((state, action, reward, next_state, done))
        if len(self.replay_memory) > self.replay_memory_size:
            self.replay_memory.pop(0)

    def sample_replay_memory(self):
        return random.sample(self.replay_memory, min(self.batch_size, len(self.replay_memory)))

    def update_target_agent(self):
        self.target_agent.load_state_dict(self.agent.state_dict())

    def train(self):
        if len(self.replay_memory) < self.batch_size:
            return

        transitions = self.sample_replay_memory()
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)

        batch_state = torch.tensor(batch_state, dtype=torch.float32, device=self.device)
        batch_action = torch.tensor(batch_action, dtype=torch.long, device=self.device)
        batch_reward = torch.tensor(batch_reward, dtype=torch.float32, device=self.device)
        batch_next_state = torch.tensor(batch_next_state, dtype=torch.float32, device=self.device)
        batch_done = torch.tensor(batch_done, dtype=torch.float32, device=self.device)

        q_values = self.agent(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        target_q_values = self.target_agent(batch_next_state).max(1)[0]
        target_q_values = (1 - batch_done) * self.discount_factor * target_q_values

        loss = self.loss_fn(q_values, target_q_values.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

class TFTRecommender:
    def __init__(self, n_champions, n_items, n_actions, device='cpu'):
        self.transformer = Transformer(n_champions, n_items).to(device)
        self.rnn = RNN(n_champions + n_items).to(device)
        self.dqn = DQNAgent(self.rnn.hidden_size, n_actions).to(device)
        self.device = device

    def recommend_action(self, game_state):
        game_tensor = self.transformer(game_state)
        rnn_hidden = self.rnn.init_hidden()
        for i in range(game_tensor.size(0)):
            rnn_out, rnn_hidden = self.rnn(game_tensor[i].unsqueeze(0), rnn_hidden)
        q_values = self.dqn(rnn_out.squeeze(0))
        action = torch.argmax(q_values)
        return action.item()

    def update(self, batch_size, discount_factor, optimizer):
        if len(self.replay_buffer) < batch_size:
            return
        
        batch = self.replay_buffer.sample(batch_size)
        state_batch = []
        action_batch = []
        next_state_batch = []
        reward_batch = []
        done_batch = []

        for transition in batch:
            state, action, next_state, reward, done = transition
            state_batch.append(state)
            action_batch.append(action)
            next_state_batch.append(next_state)
            reward_batch.append(reward)
            done_batch.append(done)

        state_batch = torch.stack(state_batch)
        action_batch = torch.tensor(action_batch)
        next_state_batch = torch.stack(next_state_batch)
        reward_batch = torch.tensor(reward_batch)
        done_batch = torch.tensor(done_batch)

        q_values = self.dqn(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
        next_q_values = self.dqn(next_state_batch).max(1)[0]
        expected_q_values = reward_batch + (1 - done_batch) * discount_factor * next_q_values

        loss = torch.nn.functional.mse_loss(q_values, expected_q_values.detach())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        self.replay_buffer.clear()