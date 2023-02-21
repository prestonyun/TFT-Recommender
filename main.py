import torch
import torch.nn as nn
import numpy as np

from env import *
from neuralnetwork import *
from _utils import *

def main():
    # Set up the environment and network
    env = TFTEnvironment(get_game_state())
    recommender = TFTRecommender(1, 1, 1000)
    num_episodes = 1000

    # Set up hyperparameters
    learning_rate = 0.001
    batch_size = 32
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.1
    epsilon_decay = 0.999
    target_update_frequency = 10

    # Set up optimizer and loss function
    optimizer = torch.optim.Adam(recommender.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()

    # Set up replay buffer
    replay_buffer = ReplayBuffer(10000)

    # Initialize variables
    total_steps = 0
    total_rewards = []
    epsilon = epsilon_start

    # Start training loop
    for i in range(num_episodes):
        # Reset environment and get initial state
        state = env.reset()
        done = False
        total_reward = 0
        
        # Loop through steps in the episode
        while not done:
            # Choose action
            if np.random.uniform() < epsilon:
                action = env.get_random_action(state)
            else:
                action = recommender.get_action()
            
            # Take action and get next state, reward, and done flag
            next_state, reward, done = env.step(action)
            total_reward += reward
            
            # Add transition to replay buffer
            replay_buffer.add_transition(state, action, reward, next_state, done)
            
            # Update state
            state = next_state
            
            # Update network weights
            if total_steps % target_update_frequency == 0:
                recommender.update_target_network()
            
            if len(replay_buffer) > batch_size:
                batch = replay_buffer.sample(batch_size)
                state_batch, action_batch, reward_batch, next_state_batch, done_batch = batch
                
                # Calculate Q values and target values
                q_values = recommender.get_q_values(state_batch, action_batch)
                with torch.no_grad():
                    next_q_values = recommender.get_target_q_values(next_state_batch)
                    target_values = reward_batch + gamma * next_q_values * (1 - done_batch)
                
                # Calculate loss and update weights
                loss = loss_function(q_values, target_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            total_steps += 1
        
        # Decay epsilon
        if epsilon > epsilon_end:
            epsilon *= epsilon_decay
        
        total_rewards.append(total_reward)
        
        # Print progress every 10 episodes
        if i % 10 == 0:
            print("Episode {}: Total reward = {}".format(i, total_reward))

if __name__ == '__main__':
    main()